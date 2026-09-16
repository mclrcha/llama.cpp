#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if defined(GGML_USE_HIP)
void ggml_cuda_cpy_f32_batch(ggml_backend_cuda_context & ctx, const ggml_tensor * const * sources,
                            const ggml_tensor * const * destinations, int count);
#endif
