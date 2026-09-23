#pragma once

#include "common.cuh"

// RDNA4 FlashAttention for prefill with head size 256 (Qwen3.x): DV split between wave pairs, f32 accumulators.
bool ggml_cuda_flash_attn_ext_rdna4_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_rdna4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
