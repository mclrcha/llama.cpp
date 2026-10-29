#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#if defined(GGML_USE_HIP)
// Conv state windows written by the fused convolution: window `keep[i]` (tokens consumed) goes to dst[i].
struct ggml_cuda_conv_states { float * dst[8]; int keep[8]; int n; };

void ggml_cuda_op_conv_prepare(ggml_backend_cuda_context & ctx,const ggml_tensor * concat,
        const ggml_tensor * weight,const ggml_tensor * state,ggml_tensor * output,const ggml_tensor * gathered,
        const ggml_tensor *norm,ggml_tensor *q,ggml_tensor *k,const ggml_cuda_conv_states * states=nullptr,bool staged=false);
#endif

#if defined(GGML_USE_HIP)
void ggml_cuda_conv_prefill(ggml_backend_cuda_context & ctx,const ggml_tensor * cat,
    const ggml_tensor * weight,const ggml_tensor * state,ggml_tensor * output,bool scratch);
#endif

#if defined(GGML_USE_HIP)
// DFlash2 local convolution fused by ggml_cuda_try_dflash_conv. Strides of coeff in floats.
void ggml_cuda_dflash_conv(ggml_backend_cuda_context & ctx, const float * x, const float * coeff, int64_t s_tap,
        int64_t s_tok, const float * base, float * dst, int hidden, int group, int n_tokens, int block, int n_taps);
#endif
