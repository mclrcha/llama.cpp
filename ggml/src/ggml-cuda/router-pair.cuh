#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)
// MoE router (256 experts, 2048 inputs, f32) plus the shared expert gate (row 256) for up to 4 tokens: one block of 128
// threads per row, the reduction of mul_mat_vec_f_router_exact. Shared by router_pair_f32 and the fused router + top-k.
template <int n_tokens>
static __device__ __forceinline__ void router_pair_block(const float * x, const float * y, float * dst, int nrows,
        const float * gate, float * gate_out) {
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int row = blockIdx.x;
    const float2 * x2 = reinterpret_cast<const float2 *>(row==256 ? gate : x + row * 2048);
    const float2 * y2 = reinterpret_cast<const float2 *>(y);
    float sums[2][n_tokens]{};
    for (int col = tid; col < 1024; col += 256) {
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const float2 weight = x2[col + group * 128];
#pragma unroll
            for (int t = 0; t < n_tokens; ++t) {
                const float2 input = y2[t * 1024 + col + group * 128];
                ggml_cuda_mad(sums[group][t], weight.x, input.x);
                ggml_cuda_mad(sums[group][t], weight.y, input.y);
            }
        }
    }
    __shared__ float partial[n_tokens][8];
#pragma unroll
    for (int group = 0; group < 2; ++group) {
#pragma unroll
        for (int t = 0; t < n_tokens; ++t) {
            sums[group][t] = warp_reduce_sum<32>(sums[group][t]);
            if (lane == 0) {
                partial[t][tid / 32 + group * 4] = sums[group][t];
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
#pragma unroll
        for (int t = 0; t < n_tokens; ++t) {
            float sum = tid < 8 ? partial[t][tid] : 0.0f;
            sum = warp_reduce_sum<32>(sum);
            if (tid == t) {
                if(row==256) { gate_out[t]=sum; } else { dst[t*nrows+row]=sum; }
            }
        }
    }
}
#endif
