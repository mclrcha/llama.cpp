#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

#if defined(GGML_USE_HIP)
void ggml_cuda_op_gdn_indexed(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gathered, const ggml_cuda_gated_delta_net_fused_cache * cache);

// Experimental single-sequence journal. Commit must precede reuse of graph inputs or state buffers.
bool ggml_cuda_gdn_journal_enable(ggml_backend_cuda_context & ctx, bool enabled);
bool ggml_cuda_gdn_journal_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, bool all_fused);
void ggml_cuda_gdn_journal_finish(ggml_backend_cuda_context & ctx);
bool ggml_cuda_gdn_journal_commit(ggml_backend_cuda_context & ctx, int rollback, bool all_snapshots);
bool ggml_cuda_gdn_journal_pending(const ggml_backend_cuda_context & ctx);
#endif
