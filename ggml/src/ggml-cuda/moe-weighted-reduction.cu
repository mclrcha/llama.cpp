#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

#if defined(GGML_USE_HIP)
template<int width>
static __global__ void moe_reduce_local(const float * __restrict__ experts,
        const float * __restrict__ weights, float * __restrict__ dst, int n_embd) {
#pragma clang fp contract(off)
    const int token = blockIdx.y;
    const int col = width * (blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= n_embd) { return; }
    float sum[width];
#pragma unroll
    for (int i=0; i<width; ++i) { sum[i] = experts[(size_t)token*8*n_embd+col+i] * weights[token*8]; }
#pragma unroll
    for (int e=1; e<8; ++e) {
#pragma unroll
        for (int i=0; i<width; ++i) {
            sum[i] = fmaf(experts[((size_t)token*8+e)*n_embd+col+i], weights[token*8+e], sum[i]);
        }
    }
#pragma unroll
    for (int i=0; i<width; ++i) { dst[(size_t)token*n_embd+col+i] = sum[i]; }
}
#endif

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
#if defined(GGML_USE_HIP)
    static const int mode=[] { const char * v=getenv("GGML_HIP_PREFILL_REDUCE"); return v ? atoi(v) : 0; }();
    if (mode && GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) &&
            n_expert_used==8 && !expert_scale && n_embd%512==0 && n_tokens>=32 && n_tokens<=65535) {
        moe_reduce_local<1><<<dim3(n_embd/256,n_tokens),256,0,stream>>>(experts,weights,dst,n_embd);
        return;
    }
#endif
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
