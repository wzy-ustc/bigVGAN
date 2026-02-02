#include "anti_alias_activation_cuda.h"
#include <cuda.h>
#include <cude_runtime.h>
#include <cuda_profiler_api.h>
// #include <cuda_fp16.h>
#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
// #include <c10/macros/Macros.h>

constexpr int ELEMENTS_PER_LDG_STG = 1;
constexpr int BUFFER_SIZE = 32;
constexpr int FILTER_SIZE = 12;
// half_FILTER_SIZE is computed as FILTER_SIZE / 2 in the kernel
constexpr int UPSAMPLE_REPLICATION_PAD = 5;
constexpr int DOWNSAMPLE_REPLICATION_PAD_LEFT = 5;
constexpr int DOWNSAMPLE_REPLICATION_PAD_RIGHT = 6;

template<class T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

template<class input_t>
__global__ void anti_alias_activation_forward(
    input_t *dst,
    const input_t *src,
    const input_t *up_ftr,
    const input_t *down_ftr,
    const input_t *alpha,
    const input_t *beta,
    int batch_size,
    int channels,
    int seq_len)
{
    // Up and downsamples filters
    input_t up_filter[FILTER_SIZE];
    input_t down_filter[FILTER_SIZE];

    // Load data from GM including extra indices reserved for replication paddings
    input_t elements[2 * FILTER_SIZE + 2 * BUFFER_SIZE + 2 * UPSAMPLE_REPLICATION_PAD + FILTER_SIZE] = {0};
    input_t intermediates[2 * FILTER_SIZE + 2 * BUFFER_SIZE + DOWNSAMPLE_REPLICATION_PAD_LEFT + DOWNSAMPLE_REPLICATION_PAD_RIGHT] = {0};

    // Ouput stores downsampled output before writing to dst
    input_t output[BUFFER_SIZE];

    // blockDim/thredIdx = (128,1,1)
    // gridDim/blockIdx = (seq, channels, batches)
    int block_offset = (blockIdx.x * 128 * BUFFER_SIZE + seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
    int local_offset = threadIdx.x * BUFFER_SIZE;
    int seq_offset = blockIdx.x * 128 * BUFFER_SIZE + local_offset;

    //intermediate have double the seq_len
    int intermediate_local_offset = threadIdx.x * BUFFER_SIZE * 2;
    int intermediate_seq_offset = blockIdx.x * 128 * BUFFER_SIZE * 2 + intermediate_local_offset;

    //Get values needed for replication padding before moving pointr
    const input_t *right_most_pntr = src + (seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
    input_t seq_left_most_value = right_most_pntr[0];
    input_t seq_right_most_value = right_most_pntr[seq_len - 1];

    // Move src and dst pointers
    src += block_offset + local_offset;
    dst += block_offset + local_offset;

    // Alpha and beta values for snake activations. Applies exp by default
    alpha = alpha + blockIdx.y;
    input_t alpha_val = expf(alpha[0]);
    beta = beta + blockIdx.y;
    input_t beta_val = expf(beta[0]);

    #pragma unroll
    for(int it = 0; it < FILTER_SIZE; it += 1){
        up_filter[it] = up_ftr[it];
        down_filter[it] = down_ftr[it];
    }

    // Apply replication padding for upsampling, matching torch impl
    constexpr int half_FILTER_SIZE = FILTER_SIZE / 2;
    #pragma unroll
    for (int it = -half_FILTER_SIZE; it < BUFFER_SIZE + half_FILTER_SIZE; it += 1)
    {
        int elements_index = seq_offset + it;
        if((elements_index < 0) && (elements_index >= -UPSAMPLE_REPLICATION_PAD))
        {
            elements[2 * (half_FILTER_SIZE + it)] = (input_t)(2.0f) * seq_left_most_value;
        }
        if((elements_index >= seq_len) && (elements_index < seq_len + UPSAMPLE_REPLICATION_PAD))
        {
            elements[2 * (half_FILTER_SIZE + it)] = (input_t)(2.0f) * seq_right_most_value;
        }
        if((elements_index >= 0) && (elements_index < seq_len))
        {
            elements[2 * (half_FILTER_SIZE + it)] = (input_t)(2.0f) * src[it];
        }
    }

    // Apply unsampling strided convolution and write to intermediates. It reserves DOWNSAMPLE_REPLICTION_PAD_LEFT for replication paddiong of the downsampiling conv later
    #pragma unroll
    for (int it = 0; it < (2 * BUFFER_SIZE + 2 * FILTER_SIZE); it += 1)
    {
        input_t acc = 0.0;
        int elements_index = intermediate_seq_offset + it; // index for intermediates
        #pragma unroll
        for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1)
        {
            if((elements_index + f_idx) >= 0)
            {
                acc += up_filter[f_idx] * elements[it + f_idx];
            }
        }
        intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] = acc;
    }

    // Apply activation function. It reserves DOWNSAMPLE_REPLICATION_PAD_LEFT and DOWNSAMPLE_REPLICATION_PAD_RIGHT for replication padding of the downsampiling conv later
    double no_div_by_zero = 0.00000001;
    #pragma unroll
    for(int it = 0; it < 2 * BUFFER_SIZE + 2 * FILTER_SIZE; it += 1){
        intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] += ((input_t)(1.0f) / (beta_val + (input_t)(no_div_by_zero)))
                                                              * (input_t)(sinf(intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] * alpha_val))
                                                              * (input_t)(sinf(intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] * alpha_val));
    }

    // Apply replication padding before downsampling conv from intermediates
    #pragma unroll
    for(int it = 0; it < DOWNSAMPLE_REPLICATION_PAD_LEFT; it += 1){
        intermediates[it] = intermediates[DOWNSAMPLE_REPLICATION_PAD_LEFT];
    }

    #pragma unroll
    for(int it = DOWNSAMPLE_REPLICATION_PAD_LEFT + 2 * BUFFER_SIZE + 2 * FILTER_SIZE;
        it < DOWNSAMPLE_REPLICATION_PAD_LEFT + 2 * BUFFER_SIZE + 2 * FILTER_SIZE + DOWNSAMPLE_REPLICATION_PAD_RIGHT; it += 1)
    {
        intermediates[it] = intermediates[DOWNSAMPLE_REPLICATION_PAD_LEFT + 2 * BUFFER_SIZE + 2 * FILTER_SIZE - 1];
    }

    // Apply downsample strided convolution (assuming stride=2) from intermediates
    #pragma unroll
    for(int it = 0; it < BUFFER_SIZE; it+=1){
        input_t acc = 0.0;
        #pragma unroll
        for(int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1)
        {
            // Add constant DOWNSAMPLE_REPLICATION_PAD_RIGHT to match torch implemention
            acc += down_filter[f_idx] * intermediates[it * 2 + f_idx + DOWNSAMPLE_REPLICATION_PAD_RIGHT];
        }
        output[it] = acc;
    }

    // Write output to dst
    #pragma unroll
    for(int it = 0; it < BUFFER_SIZE; it += ELEMENTS_PER_LDG_STG){
        int elements_index = seq_offset + it;
        if (elements_index < seq_len){
            dst[it] = output[it];
        }
    }

}  


template<class input_t>
void dispatch_anti_alias_activation_forward(
    input_t *dst,
    const input_t *src,
    const input_t *up_ftr,
    const input_t *down_ftr,
    const input_t *alpha,
    const input_t *beta,
    const int batch_size,
    int channels,
    int seq_len
){
    if(seq_len == 0){
        return;
    }else {
        //Use 128 threads per block to maximize gpu utilization
        constexpr int threads_per_block = 128;
        constexpr int seq_len_per_block = 4096;
        int blocks_per_seq_len = (seq_len + seq_len_per_block - 1) / seq_len_per_block;
        dim3 blocks(blocks_per_seq_len, channels, batch_size);
        dim3 threads(threads_per_block, 1);

        anti_alias_activation_forward<input_t>
            <<<blocks, threads, 0, stream>>>(dst, src, up_ftr, down_ftr, alpha, beta, batch_size, channels, seq_len);
    }
}