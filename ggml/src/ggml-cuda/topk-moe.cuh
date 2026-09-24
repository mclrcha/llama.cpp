#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

#if defined(GGML_USE_HIP)
// router (256 experts) + shared expert gate followed by the softmax top-k of the router logits, in one launch
void ggml_cuda_router_topk(ggml_backend_cuda_context & ctx, ggml_tensor * router, ggml_tensor * gate, ggml_tensor * weights,
        ggml_tensor * ids, const ggml_tensor * clamp, const ggml_tensor * scale, const ggml_cuda_topk_moe_args & args);
#endif

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
