#pragma once

#include "common.cuh"

// Dense K-quant / IQ4_XS x f32 matrix multiplication for RDNA4 batches that use dequantization + hipBLAS instead of MMQ
// (GGML_HIP_BLAS_TYPES): the weights are dequantized to f16 inside the kernel (GGML_HIP_MMDQ=0 uses hipBLAS).
bool ggml_cuda_mmdq_rdna4_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mmdq_rdna4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Tile-major F16 scratch; partial token tiles use the original layout.
int ggml_cuda_mmdq_layout(const ggml_tensor * src);

static __host__ __device__ __forceinline__ int64_t ggml_cuda_mmdq_x_index(int64_t n, int64_t k, int64_t K, int layout) {
    return layout ? ((n/16)*(K/16) + k/16)*256 + (n%16)*16 + k%16 : n*K + k;
}

bool ggml_cuda_mmdq_gate_up_supported(const ggml_tensor * gate, const ggml_tensor * up);
void ggml_cuda_mmdq_gate_up(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
        half * dst, int output_layout);
