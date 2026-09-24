#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if defined(GGML_USE_HIP)
void ggml_cuda_op_rms_norm_scale_128(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, ggml_tensor * dst);
#endif

#if defined(GGML_USE_HIP)
void ggml_cuda_op_rms_norm_scale_pair_128(ggml_backend_cuda_context & ctx,
        const ggml_tensor * norm_a, ggml_tensor * dst_a, const ggml_tensor * norm_b, ggml_tensor * dst_b);
#endif

#if defined(GGML_USE_HIP)
void ggml_cuda_op_residual_rms(ggml_backend_cuda_context & ctx, ggml_tensor * add,
                             const ggml_tensor * norm, const ggml_tensor * weight, ggml_tensor * dst,
                             const ggml_tensor * first, const ggml_tensor * gated_mul, void * quantized = nullptr,
                             void * mmq_d4 = nullptr, void * mmq_ds4 = nullptr,
                             const ggml_tensor * moe_experts = nullptr, const ggml_tensor * moe_weights = nullptr);
#endif

#if defined(GGML_USE_HIP)
void ggml_cuda_op_rms_norm_gate_128(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
        const ggml_tensor * weight, const ggml_tensor * gate, ggml_tensor * dst, void * quantized = nullptr);
#endif
