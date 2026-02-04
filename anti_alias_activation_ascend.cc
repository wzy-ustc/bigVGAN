#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr int FILTER_SIZE = 12;
constexpr int HALF_FILTER_SIZE = 6;
constexpr int UPSAMPLE_REPLICATION_PAD = 5;
constexpr int DOWNSAMPLE_REPLICATION_PAD_LEFT = 5;
constexpr int DOWNSAMPLE_REPLICATION_PAD_RIGHT = 6;

template <typename dType> class AntiAliasActivation {
public:
    __aicore__ inline AntiAliasActivation() {}

    __aicore__ inline void Init(const GM_ADDR src, GM_ADDR dst, GM_ADDR up_filter, GM_ADDR down_filter,
                                GM_ADDR alpha, GM_ADDR beta, uint32_t batch_size, uint32_t channels,
                                uint32_t seq_len, uint32_t channel_batch_size, AscendC::TPipe *pipeIn) 
    {
        auto totalLength = batch_size * channels * seq_len;
        this->blockLength = totalLength / AscendC::GetBlockNum();

        // Dynamic BUFFER_SIZE calculation based on seq_len
        this->BUFFER_SIZE = 2048;
        if (seq_len <= 480) {
            if (batch_size < 3) {
                this->BUFFER_SIZE = 3872;
            }
        } else if (seq_len == 2400) {
            this->BUFFER_SIZE = 1280;
        } else if (seq_len == 7200) {
            this->BUFFER_SIZE = 1216;
        } else if (seq_len == 14400) {
            this->BUFFER_SIZE = 1824;
        }

        // Calculate derived parameters
        this->MASK_SIZE = this->BUFFER_SIZE / 8 + 4;
        this->tileLength = this->BUFFER_SIZE + FILTER_SIZE;
        this->tileNum = (this->blockLength + this->BUFFER_SIZE - 1) / this->BUFFER_SIZE / BUFFER_NUM;
        this->loopCount = this->tileNum * BUFFER_NUM;
        
        if (this->tileNum == 0) {
            this->tileNum = 1;
            this->loopCount = 1;
        }

        this->seq_len = seq_len;
        this->channel_id = (AscendC::GetBlockIdx() * channel_batch_size) % channels;
        this->channels = channels;
        this->channel_batch_size = channel_batch_size;
        pipe = pipeIn;

        // Set global buffers
        srcGm.SetGlobalBuffer((__gm__ dType *)src + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        dstGm.SetGlobalBuffer((__gm__ dType *)dst + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        upFilterGm.SetGlobalBuffer((__gm__ dType *)up_filter, FILTER_SIZE);
        downFilterGm.SetGlobalBuffer((__gm__ dType *)down_filter, FILTER_SIZE);
        alphaGm.SetGlobalBuffer((__gm__ dType *)alpha, channels);
        betaGm.SetGlobalBuffer((__gm__ dType *)beta, channels);

        // Calculate array sizes
        // elem_size: total declared size of elements array
        this->elem_size = 2 * (FILTER_SIZE + this->BUFFER_SIZE + UPSAMPLE_REPLICATION_PAD) + FILTER_SIZE;
        
        // inter_size: total declared size of intermediates array
        this->inter_size = 2 * (FILTER_SIZE + this->BUFFER_SIZE) + 
                          DOWNSAMPLE_REPLICATION_PAD_LEFT + DOWNSAMPLE_REPLICATION_PAD_RIGHT;

        // Initialize pipe buffers
        pipe->InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(dType));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->BUFFER_SIZE * sizeof(dType));
        pipe->InitBuffer(inQueueAlpha, 1, this->channels * sizeof(dType));
        pipe->InitBuffer(inQueueBeta, 1, this->channels * sizeof(dType));
        pipe->InitBuffer(inQueueUp, 1, FILTER_SIZE * sizeof(dType));
        pipe->InitBuffer(inQueueDown, 1, FILTER_SIZE * sizeof(dType));

        pipe->InitBuffer(tmpBufferAlphaSrcFloat, this->channels * sizeof(float));
        pipe->InitBuffer(tmpBufferAlphaDst, this->channels * sizeof(float));
        pipe->InitBuffer(tmpBufferBetaSrcFloat, this->channels * sizeof(float));
        pipe->InitBuffer(tmpBufferBetaDst, this->channels * sizeof(float));
        pipe->InitBuffer(tmpBufferUp, FILTER_SIZE * sizeof(dType));
        pipe->InitBuffer(tmpBufferDown, FILTER_SIZE * sizeof(dType));
        // Conditional size assignments based on channel_batch_size
        if (channel_batch_size == 1) {
            this->acc_size = this->elem_size;
            this->i_size = this->inter_size;
            this->if_size = this->BUFFER_SIZE;
            this->offset1_size = this->elem_size;
            this->offset2_size = this->elem_size;
            this->offset3_size = this->BUFFER_SIZE;
            this->offset4_size = this->elem_size;
        } else {
            const uint32_t e_size = 2 * (FILTER_SIZE + seq_len + UPSAMPLE_REPLICATION_PAD);
            const uint32_t i_size = e_size + DOWNSAMPLE_REPLICATION_PAD_LEFT + DOWNSAMPLE_REPLICATION_PAD_RIGHT;
            this->acc_size = e_size;
            this->i_size = i_size;
            this->if_size = seq_len;
            this->offset1_size = e_size;
            this->offset2_size = e_size;
            this->offset3_size = seq_len;
            this->offset4_size = e_size;
        }

        pipe->InitBuffer(tmpBufferAcc, this->acc_size * sizeof(float));
        pipe->InitBuffer(tmpBufferTmp2, this->i_size * sizeof(float));
        pipe->InitBuffer(tmpBufferSin, this->i_size * sizeof(float));
        pipe->InitBuffer(tmpBufferInterFilter, this->if_size * sizeof(float));
        pipe->InitBuffer(tmpBufferOffset1, this->offset1_size * sizeof(uint32_t));
        pipe->InitBuffer(tmpBufferOffset2i, this->offset2_size * sizeof(int32_t));
        pipe->InitBuffer(tmpBufferOffset3i, this->offset3_size * sizeof(int32_t));
        pipe->InitBuffer(tmpBufferOffset4i, this->offset4_size * sizeof(int32_t));
        pipe->InitBuffer(tmpBufferElem, this->elem_size * sizeof(float));
        pipe->InitBuffer(tmpBufferTmp, this->inter_size * sizeof(float));
        pipe->InitBuffer(tmpBufferInter, this->inter_size * sizeof(float));
        pipe->InitBuffer(tmpBufferSelMask, this->MASK_SIZE * sizeof(uint16_t));
    }

    __aicore__ inline void Process() {
        // Process alpha and beta parameters
        AscendC::LocalTensor<dType> alphaLocal = inQueueAlpha.AllocTensor<dType>();
        AscendC::LocalTensor<dType> betaLocal = inQueueBeta.AllocTensor<dType>();
        
        AscendC::LocalTensor<float> alphaSrcFloat = tmpBufferAlphaSrcFloat.Get<float>();
        AscendC::LocalTensor<float> alphaDst = tmpBufferAlphaDst.Get<float>();
        AscendC::LocalTensor<float> betaSrcFloat = tmpBufferBetaSrcFloat.Get<float>();
        AscendC::LocalTensor<float> betaDst = tmpBufferBetaDst.Get<float>();

        uint32_t copy_size = this->channels * sizeof(dType);
        AscendC::DataCopyExtParams copyParams{1, copy_size, 0, 0, 0};
        AscendC::DataCopyPadExtParams<dType> padParams{true, 0, 0, 0};
        
        AscendC::DataCopyPad(alphaLocal, alphaGm, copyParams, padParams);
        AscendC::DataCopyPad(betaLocal, betaGm, copyParams, padParams);
        
        inQueueAlpha.EnQue(alphaLocal);
        inQueueBeta.EnQue(betaLocal);

        AscendC::LocalTensor<dType> alphaSrc = inQueueAlpha.DeQue<dType>();
        AscendC::LocalTensor<dType> betaSrc = inQueueBeta.DeQue<dType>();

        AscendC::Cast(alphaSrcFloat, alphaSrc, AscendC::RoundMode::CAST_NONE, channels);
        AscendC::Cast(betaSrcFloat, betaSrc, AscendC::RoundMode::CAST_NONE, channels);
        
        inQueueAlpha.FreeTensor(alphaSrc);
        inQueueBeta.FreeTensor(betaSrc);

        // Apply exponential to alpha and beta
        AscendC::Exp(alphaDst, alphaSrcFloat, channels);
        AscendC::Exp(betaDst, betaSrcFloat, channels);

        // Add small value to beta to prevent division by zero
        float scalar = 0.000000001f;
        AscendC::Adds(betaDst, betaDst, scalar, channels);

        // Calculate 1.0 / beta
        if (channels < 8) {
            for (int i = 0; i < channels; i++) {
                betaSrcFloat(i) = 1.0f / betaDst(i);
            }
        } else {
            AscendC::Duplicate<float>(alphaSrcFloat, 1.0f, this->channels);
            AscendC::Div(betaSrcFloat, alphaSrcFloat, betaDst, this->channels);
        }
        
        AscendC::LocalTensor<dType> upLocal = inQueueUp.AllocTensor<dType>();
        AscendC::LocalTensor<dType> downLocal = inQueueDown.AllocTensor<dType>();
        
        // Load filters
        copy_size = FILTER_SIZE * sizeof(dType);
        AscendC::DataCopyExtParams copyParams2{1, copy_size, 0, 0, 0};

        AscendC::DataCopyPad(upLocal, upFilterGm, copyParams2, padParams);
        AscendC::DataCopyPad(downLocal, downFilterGm, copyParams2, padParams);
        
        inQueueUp.EnQue(upLocal);
        inQueueDown.EnQue(downLocal);

        AscendC::LocalTensor<dType> upData = inQueueUp.DeQue<dType>();
        AscendC::LocalTensor<dType> downData = inQueueDown.DeQue<dType>();

        AscendC::LocalTensor<dType> upTmp = tmpBufferUp.Get<dType>();
        AscendC::LocalTensor<dType> downTmp = tmpBufferDown.Get<dType>();
        upTmp = upData;
        downTmp = downData;

        inQueueUp.FreeTensor(upData);
        inQueueDown.FreeTensor(downData);

        // Get boundary values for replication padding
        auto leftvalue = srcGm[0];
        auto rightvalue = srcGm[this->seq_len * this->channel_batch_size - 1];

        // Generate index offsets
        // [0,0,1,1,2,2,...,N,N] * sizeof(float)
        AscendC::LocalTensor<uint32_t> indexOffset1 = tmpBufferOffset1.Get<uint32_t>();
        for (int i = 0; i < this->offset1_size; i++) {
            indexOffset1(i) = (i / 2) * sizeof(float);
        }

        // [0,1,2,...,N] * sizeof(float)
        AscendC::LocalTensor<int32_t> indexOffset2i = tmpBufferOffset2i.Get<int32_t>();
        AscendC::CreateVecIndex(indexOffset2i, 0, this->offset2_size);
        AscendC::Muls(indexOffset2i, indexOffset2i, (int32_t)sizeof(float), this->offset2_size);
        auto indexOffset2 = indexOffset2i.ReinterpretCast<uint32_t>();

        // [0,2,4,...,2N] * sizeof(float)
        AscendC::LocalTensor<int32_t> indexOffset3i = tmpBufferOffset3i.Get<int32_t>();
        AscendC::CreateVecIndex(indexOffset3i, 0, this->offset3_size);
        AscendC::Muls(indexOffset3i, indexOffset3i, (int32_t)(2 * sizeof(float)), this->offset3_size);
        auto indexOffset3 = indexOffset3i.ReinterpretCast<uint32_t>();

        // [0,0,0,0,0,1,2,3,...,N] * sizeof(float)
        AscendC::LocalTensor<int32_t> indexOffset4i = tmpBufferOffset4i.Get<int32_t>();
        AscendC::CreateVecIndex(indexOffset4i, 0, this->offset4_size);
        AscendC::Muls(indexOffset4i, indexOffset4i, (int32_t)sizeof(float), this->offset4_size);
        AscendC::Adds(indexOffset4i, indexOffset4i, -20, this->offset4_size);
        for (int i = 0; i < 5; i++) {
            indexOffset4i(i) = 0;
        }
        auto indexOffset4 = indexOffset4i.ReinterpretCast<uint32_t>();

        // Main processing loop
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, leftvalue, rightvalue);
            Compute(i, this->seq_len, upTmp, downTmp, alphaDst, betaSrcFloat, indexOffset1, indexOffset2, indexOffset3, indexOffset4);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t i, AscendC::GlobalTensor<dType> &leftvalue, AscendC::GlobalTensor<dType> &rightvalue) {
        AscendC::LocalTensor<dType> inputLocal = inQueue.AllocTensor<dType>();
        
        auto start_idx = i * this->BUFFER_SIZE - HALF_FILTER_SIZE;
        auto copy_size = tileLength;
        
        if (start_idx < 0) {
            start_idx = 0;
            copy_size = tileLength - HALF_FILTER_SIZE;
            copy_size = copy_size * sizeof(dType);
            AscendC::DataCopyExtParams copyParams{1, copy_size, 0, 0, 0};
            AscendC::DataCopyPadExtParams<dType> padParams{true, HALF_FILTER_SIZE, 0, leftvalue(0)};
            AscendC::DataCopyPad(inputLocal, srcGm[start_idx], copyParams, padParams);
        } else if (start_idx + tileLength - HALF_FILTER_SIZE > this->blockLength) {
            copy_size = this->blockLength - start_idx;
            copy_size = copy_size * sizeof(dType);
            AscendC::DataCopyExtParams copyParams{1, copy_size, 0, 0, 0};
            AscendC::DataCopyPadExtParams<dType> padParams{true, 0, HALF_FILTER_SIZE, rightvalue(0)};
            AscendC::DataCopyPad(inputLocal, srcGm[start_idx], copyParams, padParams);
        } else {
            copy_size = copy_size * sizeof(dType);
            AscendC::DataCopyExtParams copyParams{1, copy_size, 0, 0, 0};
            AscendC::DataCopyPadExtParams<dType> padParams{true, 0, 0, 0};
            AscendC::DataCopyPad(inputLocal, srcGm[start_idx], copyParams, padParams);
        }
        
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int32_t i, int32_t seq_len, AscendC::LocalTensor<dType> &upF,
                                   AscendC::LocalTensor<dType> &downF, AscendC::LocalTensor<float> &alphaVal, 
                                   AscendC::LocalTensor<float> &betaVal, 
                                   AscendC::LocalTensor<uint32_t> &indexOffset1,
                                   AscendC::LocalTensor<uint32_t> &indexOffset2,
                                   AscendC::LocalTensor<uint32_t> &indexOffset3,
                                   AscendC::LocalTensor<uint32_t> &indexOffset4) {
        AscendC::LocalTensor<dType> inputLocal = inQueue.DeQue<dType>();
        AscendC::LocalTensor<dType> outputLocal = outQueue.AllocTensor<dType>();

        AscendC::LocalTensor<float> elements = tmpBufferElem.Get<float>();
        AscendC::LocalTensor<float> intermediates = tmpBufferInter.Get<float>();
        AscendC::LocalTensor<float> acc = tmpBufferAcc.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBufferTmp.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBufferTmp2.Get<float>();

        float inputVal(0);
        AscendC::Duplicate<float>(elements, inputVal, this->elem_size);
        AscendC::Duplicate<float>(intermediates, inputVal, this->inter_size);

        AscendC::LocalTensor<uint16_t> selMask = tmpBufferSelMask.Get<uint16_t>();
        const uint32_t cnt = 2 * tileLength;
        
        // Multiply input by 2 and cast to float
        AscendC::Muls(inputLocal, inputLocal, (dType)(2), tileLength);
        AscendC::Cast(tmp, inputLocal, AscendC::RoundMode::CAST_NONE, tileLength);

        if (channel_batch_size == 1) {
            // Gather elements with indexOffset1 for upsampling
            AscendC::Gather(elements, tmp, indexOffset1, (uint32_t)0, cnt);
            // Create selection mask and apply to elements
            AscendC::Duplicate<uint16_t>(selMask, 21845, this->MASK_SIZE);
            AscendC::Select(elements, selMask, elements, (float)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, cnt);

            // Initialize accumulator
            AscendC::Duplicate<float>(acc, inputVal, this->elem_size);

            // Upsampling convolution
            for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1) {
                uint32_t srcBaseAddr = f_idx * sizeof(float);
                AscendC::Gather(tmp, elements, indexOffset2, srcBaseAddr, cnt);
                AscendC::Muls(tmp, tmp, (float)upF(f_idx), cnt);
                AscendC::Add(acc, acc, tmp, cnt);
            }

            // Gather to intermediates with left padding offset
            AscendC::Gather(intermediates, acc, indexOffset4, (uint32_t)0, cnt + DOWNSAMPLE_REPLICATION_PAD_LEFT);
            AscendC::Duplicate<float>(intermediates, inputVal, DOWNSAMPLE_REPLICATION_PAD_LEFT);

            // Snake activation
            AscendC::LocalTensor<float> sin_tmp = tmpBufferSin.Get<float>();
            const uint32_t cnt2 = this->inter_size;
            AscendC::Muls(tmp, intermediates, (float)alphaVal(this->channel_id), cnt2);
            AscendC::Sin(sin_tmp, tmp, cnt2);
            AscendC::Mul(tmp, sin_tmp, sin_tmp, cnt2);
            AscendC::Muls(tmp, tmp, (float)betaVal(this->channel_id), cnt2);
            AscendC::Add(intermediates, intermediates, tmp, cnt2);

            // Replication padding for downsampling
            auto left = intermediates(DOWNSAMPLE_REPLICATION_PAD_LEFT);
            auto right = intermediates(cnt2 - DOWNSAMPLE_REPLICATION_PAD_LEFT - 1);
            for (int it = 0; it < DOWNSAMPLE_REPLICATION_PAD_LEFT; it += 1) {
                intermediates(it) = left;
            }
            for (int it = cnt2 - DOWNSAMPLE_REPLICATION_PAD_RIGHT; it < cnt2; it += 1) {
                intermediates(it) = right;
            }

            // Downsampling convolution
            AscendC::LocalTensor<float> interFilter = tmpBufferInterFilter.Get<float>();
            AscendC::Duplicate<float>(acc, inputVal, this->BUFFER_SIZE);
            for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1) {
                uint32_t srcBaseAddr = (f_idx + DOWNSAMPLE_REPLICATION_PAD_RIGHT) * sizeof(float);
                AscendC::Gather(interFilter, intermediates, indexOffset3, srcBaseAddr, this->BUFFER_SIZE);
                AscendC::Muls(interFilter, interFilter, (float)downF(f_idx), this->BUFFER_SIZE);
                AscendC::Add(acc, acc, interFilter, this->BUFFER_SIZE);
            }
            AscendC::Cast(outputLocal, acc, AscendC::RoundMode::CAST_NONE, this->BUFFER_SIZE);
        } else {
            // Multi-channel/batch processing path
            for (int ci = 0; ci < channel_batch_size; ci++) {
                if (ci == 0) {
                    AscendC::Gather(tmp2, tmp, indexOffset2, (uint32_t)0, seq_len + HALF_FILTER_SIZE);
                    for (int si = seq_len + HALF_FILTER_SIZE; si < seq_len + FILTER_SIZE - 1; si++) {
                        tmp2(si) = tmp2(seq_len + HALF_FILTER_SIZE - 1);
                    }
                    tmp2(seq_len + FILTER_SIZE - 1) = 0;
                } else {
                    AscendC::Gather(tmp2, tmp, indexOffset2, (uint32_t)((seq_len * ci) * sizeof(float)), seq_len + FILTER_SIZE);
                    tmp2(0) = 0;
                    for (int si = 1; si < HALF_FILTER_SIZE; si++) {
                        tmp2(si) = tmp2(HALF_FILTER_SIZE);
                    }
                    for (int si = seq_len + HALF_FILTER_SIZE; si < seq_len + FILTER_SIZE - 1; si++) {
                        tmp2(si) = tmp2(seq_len + HALF_FILTER_SIZE - 1);
                    }
                    tmp2(seq_len + FILTER_SIZE - 1) = 0;
                }

                const uint32_t cnt = 2 * (seq_len + FILTER_SIZE);
                AscendC::Gather(elements, tmp2, indexOffset1, (uint32_t)0, cnt);
                AscendC::Duplicate<uint16_t>(selMask, 21845, this->MASK_SIZE);
                AscendC::Select(elements, selMask, elements, (float)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, cnt);

                const uint32_t elem_size = cnt + 2 * UPSAMPLE_REPLICATION_PAD;
                const uint32_t inter_size = cnt + DOWNSAMPLE_REPLICATION_PAD_LEFT + DOWNSAMPLE_REPLICATION_PAD_RIGHT;
                
                AscendC::Duplicate<float>(acc, inputVal, elem_size);

                // Upsampling convolution
                for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1) {
                    uint32_t srcBaseAddr = f_idx * sizeof(float);
                    AscendC::Gather(tmp2, elements, indexOffset2, srcBaseAddr, cnt);
                    AscendC::Muls(tmp2, tmp2, (float)upF(f_idx), cnt);
                    AscendC::Add(acc, acc, tmp2, cnt);
                }

                // Gather to intermediates
                AscendC::Gather(intermediates, acc, indexOffset4, (uint32_t)0, cnt + DOWNSAMPLE_REPLICATION_PAD_LEFT);
                AscendC::Duplicate<float>(intermediates, inputVal, DOWNSAMPLE_REPLICATION_PAD_LEFT);

                // Snake activation
                AscendC::LocalTensor<float> sin_tmp = tmpBufferSin.Get<float>();
                const uint32_t cnt2 = inter_size;
                AscendC::Muls(tmp2, intermediates, (float)alphaVal(this->channel_id + ci), cnt2);
                AscendC::Sin(sin_tmp, tmp2, cnt2);
                AscendC::Mul(tmp2, sin_tmp, sin_tmp, cnt2);
                AscendC::Muls(tmp2, tmp2, (float)betaVal(this->channel_id + ci), cnt2);
                AscendC::Add(intermediates, intermediates, tmp2, cnt2);

                // Replication padding
                auto left = intermediates(DOWNSAMPLE_REPLICATION_PAD_LEFT);
                auto right = intermediates(cnt2 - DOWNSAMPLE_REPLICATION_PAD_LEFT - 1);
                for (int it = 0; it < DOWNSAMPLE_REPLICATION_PAD_LEFT; it += 1) {
                    intermediates(it) = left;
                }
                for (int it = cnt2 - DOWNSAMPLE_REPLICATION_PAD_RIGHT; it < cnt2; it += 1) {
                    intermediates(it) = right;
                }

                // Downsampling convolution
                AscendC::LocalTensor<float> interFilter = tmpBufferInterFilter.Get<float>();
                AscendC::Duplicate<float>(acc, inputVal, seq_len);
                for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1) {
                    uint32_t srcBaseAddr = (f_idx + DOWNSAMPLE_REPLICATION_PAD_RIGHT) * sizeof(float);
                    AscendC::Gather(interFilter, intermediates, indexOffset3, srcBaseAddr, seq_len);
                    AscendC::Muls(interFilter, interFilter, (float)downF(f_idx), seq_len);
                    AscendC::Add(acc, acc, interFilter, seq_len);
                }
                AscendC::Cast(outputLocal[seq_len * ci], acc, AscendC::RoundMode::CAST_NONE, seq_len);
            }
        }

        inQueue.FreeTensor(inputLocal);
        outQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(int32_t i)
    {
        AscendC::LocalTensor<dType> outputLocal = outQueue.DeQue<dType>();
        if (i == loopCount - 1) {
            uint32_t copy_size =
                (this->blockLength - (this->tileNum * BUFFER_NUM - 1) * this->BUFFER_SIZE) * sizeof(dType);
            if (loopCount == 1) {
                copy_size = this->blockLength * sizeof(dType);
            }
            AscendC::DataCopyExtParams copyParams{1, copy_size, 0, 0, 0};
            AscendC::DataCopyPad(dstGm[i * this->BUFFER_SIZE], outputLocal, copyParams);
        } else {
            /*
            uint32_t copy_size = BUFFER_SIZE * sizeof(dType);
            AscendC::DataCopyExtParams copyParams{1, copy_size, 0, 0, 0};
            AscendC::DataCopyPad(dstGm[i * BUFFER_SIZE], outputLocal, copyParams);
            */
            AscendC::DataCopy(dstGm[i * this->BUFFER_SIZE], outputLocal, this->BUFFER_SIZE);
        }
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe *pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueAlpha;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueBeta;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueUp;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueDown;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferAlphaSrcFloat, tmpBufferAlphaDst;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferBetaSrcFloat, tmpBufferBetaDst;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferUp, tmpBufferDown;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferAcc, tmpBufferTmp, tmpBufferTmp2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferElem, tmpBufferInter, tmpBufferInterFilter;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferSin;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferOffset1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferOffset2i, tmpBufferOffset3i, tmpBufferOffset4i;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferSelMask;

    AscendC::GlobalTensor<dType> srcGm;
    AscendC::GlobalTensor<dType> dstGm;
    AscendC::GlobalTensor<dType> upFilterGm;
    AscendC::GlobalTensor<dType> downFilterGm;
    AscendC::GlobalTensor<dType> alphaGm;
    AscendC::GlobalTensor<dType> betaGm;

    uint32_t blockLength;
    uint32_t seq_len;
    uint32_t channel_batch_size;
    uint32_t channel_id;
    uint32_t channels;
    uint32_t tileNum;
    uint32_t loopCount;
    uint32_t tileLength;
    uint32_t elem_size;
    uint32_t inter_size;
    uint32_t offset1_size;
    uint32_t offset2_size;
    uint32_t offset3_size;
    uint32_t offset4_size;
    uint32_t acc_size;
    uint32_t tmp2_size;
    uint32_t i_size;
    uint32_t if_size;
    int32_t BUFFER_SIZE;
    int32_t MASK_SIZE;
};

extern "C" __global__ __aicore__ void anti_alias_activation(GM_ADDR src, GM_ADDR dst, GM_ADDR up_filter,
                                                            GM_ADDR down_filter, GM_ADDR alpha, GM_ADDR beta,
                                                            uint32_t batch_size, uint32_t channels, uint32_t seq_len,
                                                            uint32_t channel_batch_size)
{
    AscendC::TPipe pipe;
    AntiAliasActivation<half> op;
    op.Init(src, dst, up_filter, down_filter, alpha, beta, batch_size, channels, seq_len, channel_batch_size, &pipe);
    op.Process();
    pipe.Destroy();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void anti_alias_activation_launch(uint32_t blockDim, void* stream,
                                  uint8_t* src, uint8_t* up_ftr, uint8_t* down_ftr,
                                  uint8_t* alpha, uint8_t* beta, uint8_t* out_tensor,
                                  uint32_t batch_size, uint32_t channels, uint32_t seq_len,
                                  uint32_t channel_batch_size)
{
    anti_alias_activation<<<blockDim, nullptr, stream>>>(src, out_tensor, up_ftr, down_ftr, alpha, beta, batch_size, channels, seq_len, channel_batch_size);
}
#endif
