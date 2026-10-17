#pragma once

#include "common.cuh"

// Dense K-quant / IQ4_XS x f32 matrix multiplication for RDNA4 batches that use dequantization + hipBLAS instead of MMQ
// (GGML_HIP_BLAS_TYPES): the weights are dequantized to f16 inside the kernel (GGML_HIP_MMDQ=0 uses hipBLAS).
bool ggml_cuda_mmdq_rdna4_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mmdq_rdna4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
