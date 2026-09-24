#include "common.cuh"

// sum over the used experts of experts[row, col] * scale[row] * weights[row], rows first_row..first_row+n_expert_used-1;
// shared by the reduction kernel and fused consumers so that both round the same way
static __device__ __forceinline__ float moe_weighted_sum(const float * __restrict__ experts, const float * __restrict__ expert_scale,
        const float * __restrict__ weights, const uint64_t first_row, const int64_t n_embd, const int64_t col, const int n_expert_used) {
    const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float       sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    return sum;
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);
