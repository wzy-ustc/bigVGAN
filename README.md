# bigVGAN

# Anti-Alias Activation 算子完整公式逻辑

## 概述

这个算子对输入序列 $x[t]$ 执行：**上采样卷积（×2）→ Snake激活 → 下采样卷积（stride=2）**，实现抗混叠的非线性激活。

---

## 符号约定

| 符号 | 对应代码常量 | 值 |
|------|-------------|-----|
| $F$ | `FILTER_SIZE` | 12 |
| $F_h$ | `half_FILTER_SIZE` | 6 |
| $N$ | `BUFFER_SIZE` | 32 |
| $P_u$ | `UPSAMPLE_REPLICATION_PAD` | 5 |
| $P_L$ | `DOWNSAMPLE_REPLICATION_PAD_LEFT` | 5 |
| $P_R$ | `DOWNSAMPLE_REPLICATION_PAD_RIGHT` | 6 |

---

## 输入输出

- **输入**： $x[t] \in \mathbb{R}^{B \times C \times T}$ ，其中 $B$=batch, $C$=channels, $T$=seq_len
- **上采样滤波器**： $h_u[k] \in \mathbb{R}^{F}$ （所有通道共享）
- **下采样滤波器**： $h_d[k] \in \mathbb{R}^{F}$ （所有通道共享）
- **Snake激活参数**（按通道）：
  - $\alpha_c = \exp(\text{alpha}[c]) > 0$
  - $\beta_c = \exp(\text{beta}[c]) > 0$
- **输出**： $y[t] \in \mathbb{R}^{B \times C \times T}$ （长度不变）

---

## 阶段1：Replication Padding + 插零上采样准备

### 复制补边函数

定义复制补边算子 $\text{rep}(x, t)$：

$$
\text{rep}(x, t) = \begin{cases}
x[0] & \text{if } t < 0 \\\\
x[T-1] & \text{if } t \geq T \\\\
x[t] & \text{otherwise}
\end{cases}
$$

### 构造上采样序列 $u[n]$

对每个线程负责的段，构造一个"×2 上采样后的稀疏序列"：

$$
u[2 \cdot (F_h + i)] = 2 \cdot \text{rep}(x, s_0 + i)
$$

其中：
- $i \in [-F_h,\; N + F_h)$
- $s_0$ 为当前线程的 `seq_offset`
- 奇数位置 $u[2k+1]$ 默认为 0（或由实现保证）
- 系数 $2.0$ 用于能量归一化

**代码对应**：
```cpp
elements[2 * (half_FILTER_SIZE + it)] = 2.0f * rep(src, seq_offset + it);
```

---

## 阶段2：上采样卷积（FIR滤波）

对 $u[n]$ 做 1D FIR 卷积，得到中间序列 $z[n]$：

$$
z[n] = \sum_{k=0}^{F-1} h_u[k] \cdot u[n + k]
$$

其中 $n \in [0,\; 2N + 2F)$。

**代码对应**：
```cpp
acc += up_filter[f_idx] * elements[it + f_idx];
intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] = acc;
```

**边界处理**：当 $n + k < 0$ 时，该项不累加（等价于 $u[n+k] = 0$）。

---

## 阶段3：Snake激活函数

对每个位置 $n$，应用 Snake 激活：

$$
z'[n] = z[n] + \frac{1}{\beta_c + \epsilon} \cdot \sin^2(\alpha_c \cdot z[n])
$$

其中：
- $\epsilon = 10^{-8}$（防止除零）
- $\alpha_c, \beta_c$ 是当前通道 $c$ = `blockIdx.y` 的参数

**代码对应**：
```cpp
intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] += 
    (1.0f / (beta_val + eps)) * 
    sinf(intermediates[...] * alpha_val) * 
    sinf(intermediates[...] * alpha_val);
```

---

## 阶段4：Replication Padding（为下采样准备）

对 $z'[n]$ 做左右复制补边：

$$
\tilde{z}[n] = \begin{cases}
z'[P_L] & \text{if } n < P_L \\\\
z'[n] & \text{if } P_L \leq n < E \\\\
z'[E-1] & \text{if } n \geq E
\end{cases}
$$

其中 $E = P_L + 2N + 2F$。

**代码对应**：
```cpp
// 左padding
intermediates[it] = intermediates[DOWNSAMPLE_REPLICATION_PAD_LEFT];

// 右padding
intermediates[it] = intermediates[end - 1];
```

---

## 阶段5：下采样卷积（stride=2）

对 $\tilde{z}[n]$ 做 stride=2 的卷积，得到输出 $y[t]$：

$$
y[t] = \sum_{k=0}^{F-1} h_d[k] \cdot \tilde{z}[2t + k + P_R]
$$

其中：
- $t \in [0, N)$
- $P_R$ = `DOWNSAMPLE_REPLICATION_PAD_RIGHT`（用于对齐 PyTorch 实现）

**代码对应**：
```cpp
acc += down_filter[f_idx] * intermediates[it * 2 + f_idx + DOWNSAMPLE_REPLICATION_PAD_RIGHT];
output[it] = acc;
```

---

## 阶段6：写回输出

将 `output[it]` 写回 `dst[it]`（仅当索引在有效范围内）：

$$
y[s_0 + t] = \text{output}[t], \quad \text{if } s_0 + t < T
$$

---

## 完整公式（组合算子）

$$
y = \text{DownConv}_{s2}\Big(\text{RepPad}\big(\text{Snake}_c(\text{UpConv}(\text{InsertZero}_2(x)))\big)\Big)
$$

其中：
- $\text{InsertZero}_2$：插零上采样（×2）
- $\text{UpConv}$：上采样 FIR 滤波
- $\text{Snake}_c$：按通道的 Snake 激活
- $\text{RepPad}$：复制补边
- $\text{DownConv}_{s2}$：stride=2 下采样卷积

---

## 并行化索引映射

### Grid/Block 布局

- **blockDim/threadIdx** = `(128, 1, 1)`
- **gridDim/blockIdx** = `(seq_blocks, channels, batches)`

### 内存偏移计算

对每个 `(batch=b, channel=c, time_chunk=k)`：

$$
\text{offset} = k \cdot (128 \cdot N) + T \cdot (c + C \cdot b)
$$

其中：
- $k$ = `blockIdx.x`（时间段索引）
- $c$ = `blockIdx.y`（通道索引）
- $b$ = `blockIdx.z`（batch索引）
- $C$ = `gridDim.y` = channels

### 线程内偏移

$$
l = \text{threadIdx.x} \cdot N
$$

$$
s_0 = k \cdot (128 \cdot N) + l
$$

其中 $l$ 为 `local_offset`，$s_0$ 为 `seq_offset`。

---

## 关键常量

- `BUFFER_SIZE = 32`：每个线程处理的输出点数
- `FILTER_SIZE = 12`：卷积核长度
- `half_FILTER_SIZE = 6`：卷积核半径
- `UPSAMPLE_REPLICATION_PAD = 5`：上采样阶段的padding范围
- `DOWNSAMPLE_REPLICATION_PAD_LEFT = 5`：下采样左padding
- `DOWNSAMPLE_REPLICATION_PAD_RIGHT = 6`：下采样右padding（对齐PyTorch）

---

## 注意事项

1. **滤波器共享**：所有通道使用相同的 $h_u, h_d$（不是 per-channel）
2. **激活参数按通道**：`alpha_c, beta_c` 每个通道不同
3. **能量归一化**：上采样阶段乘以 2.0 用于补偿插零导致的能量损失
4. **边界对齐**：`DOWNSAMPLE_REPLICATION_PAD_RIGHT` 用于匹配 PyTorch 的索引行为
