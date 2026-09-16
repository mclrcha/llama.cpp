#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_can_fuse_mixed_mmvq(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * src, const ggml_tensor * dst);

bool ggml_cuda_can_fuse_shared_q8(const ggml_tensor * up, const ggml_tensor * gate,
                                const ggml_tensor * src, const ggml_tensor * dst);

#if defined(GGML_USE_HIP)
void ggml_cuda_gdn_gates_q8(ggml_backend_cuda_context & ctx,
        const ggml_tensor * alpha, const ggml_tensor * beta, const ggml_tensor * input,
        const ggml_tensor * bias, const ggml_tensor * scale, float * output, float * beta_output);
#endif

#if defined(GGML_USE_HIP)
void ggml_cuda_moe_down_reduce(ggml_backend_cuda_context & ctx, const ggml_tensor * down,
                              const ggml_tensor * routing, ggml_tensor * output);
#endif

#if defined(GGML_USE_HIP)
void ggml_cuda_q8_pair(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);
#endif
