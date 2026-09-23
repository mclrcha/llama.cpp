#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "mma.cuh"

template <int S_v, bool KDA, bool keep_rs_t, int columns_per_block = 4, int column_lanes = 0, bool indexed = false>
__global__ void __launch_bounds__((column_lanes ? column_lanes : (ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v)) * columns_per_block, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_indices, int64_t state_row_stride) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // Each lane group owns one column.
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = (indexed ? int64_t(state_indices[sequence]) * state_row_stride : sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = column_lanes ? column_lanes : (ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v);
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
            if constexpr (column_lanes == 8) {
                static_assert(S_v == 128, "Virtual wave reduction requires S_v=128");
                float partial[4] = {};
#pragma unroll
                for (int r = 0; r < 4; ++r) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        partial[v] += s_shard[4 * r + v] * k_reg[4 * r + v];
                    }
                }
                // Preserve the original XOR-16 and XOR-8 reduction stages.
                kv_shard = (partial[0] + partial[2]) + (partial[1] + partial[3]);
            } else {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_reg[r];
                }
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
            float partial[4] = {};
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                if constexpr (column_lanes == 8) {
                    partial[r % 4] += s_shard[r] * q_reg[r];
                } else {
                    attn_partial += s_shard[r] * q_reg[r];
                }
            }
            if constexpr (column_lanes == 8) {
                attn_partial = (partial[0] + partial[2]) + (partial[1] + partial[3]);
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

#if defined(GGML_USE_HIP)
// Chunked prefill (RDNA4, S_v = 128, scalar log decay g <= 0, final state only). Per chunk of C = 64 tokens and head:
//   gamma_t = exp(b_t), b = inclusive cumsum of g in the chunk; A[t][s] = beta_t gamma_t/gamma_s k_t.k_s (s < t).
//   U = (I + A)^-1 [beta gamma K | beta V] = [-W' | U0] (forward substitution, f32).
//   Per column group of the state: U = U0 - W S, O = (gamma Q) S + P U with P[t][s] = gamma_t/gamma_s q_t.k_s (s <= t),
//   S' = gamma_C S + Kt^T U with Kt[s] = gamma_C/gamma_s k_s. Matrix products use f16 WMMA with f32 accumulation.
namespace gdn_chunk {
constexpr int C  = 64;
constexpr int D  = 128;
// Per (chunk, head) scratch, halves unless noted.
constexpr size_t off_W   = 0;                         // -W'   [C][D]  f16 row-major
constexpr size_t off_Q   = off_W + C*D*2;             // gamma Q [C][D] f16 row-major
constexpr size_t off_KT  = off_Q + C*D*2;             // Kt^T  [D][C]  f16 row-major
constexpr size_t off_P   = off_KT + D*C*2;            // P     [C][C]  f16 row-major
constexpr size_t off_U0  = off_P + C*C*2;             // U0^T  [D][C]  f32
constexpr size_t off_G   = off_U0 + D*C*4;            // gamma_C f32
constexpr size_t stride  = off_G + 256;
}

static __global__ void __launch_bounds__(256)
gated_delta_net_chunk_prepare(const float * q, const float * k, const float * v, const float * g, const float * beta,
        char * scratch, int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic, int n_chunks, int * growth_flag) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    using namespace ggml_cuda_mma;
    using namespace gdn_chunk;
    typedef tile<16,  8, half2, DATA_LAYOUT_I_MAJOR> tile_AB;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_D;
    constexpr int LK = D + 8; // row stride of [C][D] f16 tiles
    constexpr int LT = C + 8; // row stride of [D][C] and [C][C] f16 tiles

    const int chunk = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
    // The mma tile helpers index lanes by threadIdx.x: 32 x 8 block, one warp per threadIdx.y.
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp*32 + lane;
    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(seq, rq3_magic);
    const int64_t t0 = int64_t(chunk)*C;

    // Region 0: K [C][LK] f16, later V^T [D][LT] f16.   Region 1: Q [C][LK] f16, later K^T [D][LT] f16.
    // Region 2: A [C][C+4] f32, later T diag(beta gamma) and T diag(beta) as [C][LT] f16.
    constexpr int bytes0 = (C*LK > D*LT ? C*LK : D*LT) * 2;
    constexpr int bytes2 = (C*(C + 4)*4 > 2*C*LT*2 ? C*(C + 4)*4 : 2*C*LT*2);
    __shared__ __attribute__((aligned(16))) char smem[2*bytes0 + bytes2];
    half  * sK  = (half  *) smem;
    half  * sVT = (half  *) smem;
    half  * sQ  = (half  *) (smem + bytes0);
    half  * sKT = (half  *) (smem + bytes0);
    float * sA  = (float *) (smem + 2*bytes0);
    half  * sT1 = (half  *) (smem + 2*bytes0);
    half  * sT2 = sT1 + C*LT;
    __shared__ float sb[C], sgam[C], sbeta[C];

    char * out = scratch + (int64_t(seq)*H + h)*n_chunks*stride + int64_t(chunk)*stride;

    if (tid < C) {
        const int64_t t = t0 + tid;
        const bool valid = t < n_tokens;
        sbeta[tid] = valid ? beta[seq*sb3 + t*sb2 + h*sb1] : 0.0f;
        sb[tid]    = valid ? g   [seq*sb3 + t*sb2 + h*sb1] : 0.0f;
    }
    // Rows past the end read the last token and are zeroed: branch free loads the compiler can batch.
    const int64_t t_last = n_tokens - 1;
#pragma unroll 8
    for (int e = tid; e < C*D; e += 256) {
        const int t = e / D, i = e % D;
        const int64_t tt = t0 + t;
        const int64_t tc = tt < n_tokens ? tt : t_last;
        const float m  = tt < n_tokens ? 1.0f : 0.0f;
        const float kv = k[iq3*sq3 + tc*sq2 + iq1*sq1 + i];
        const float qv = q[iq3*sq3 + tc*sq2 + iq1*sq1 + i];
        sK[t*LK + i] = __float2half(m*kv);
        sQ[t*LK + i] = __float2half(m*qv);
    }
    __syncthreads();
    if (tid == 0) {
        // The chunked form multiplies by exp(b_t - b_s), s < t: it is only used for decaying gates (g <= 0, as in the models).
        // A head whose gates grow in any chunk takes the recurrent fallback in the scan kernel.
        float acc = 0.0f, lowest = 0.0f, growth = 0.0f;
        for (int t = 0; t < C; ++t) {
            acc += sb[t];
            sb[t] = acc;
            growth = fmaxf(growth, acc - lowest);
            lowest = fminf(lowest, acc);
        }
        if (growth > 0.0f) {
            atomicOr(growth_flag + seq*H + h, 1);
        }
    }
    __syncthreads();
    if (tid < C) {
        sgam[tid] = expf(sb[tid]);
    }
    __syncthreads();

    // KK -> A (f32, SRAM) and QK -> P (f16, global): 16 tiles each, 4 per warp.
    half * P = (half *) (out + off_P);
    for (int tile_idx = warp; tile_idx < 32; tile_idx += 8) {
        const bool is_q = tile_idx >= 16;
        const int tb = (tile_idx % 16) / 4, sbk = (tile_idx % 16) % 4;
        tile_D acc;
#pragma unroll
        for (int kb = 0; kb < D/16; ++kb) {
            tile_AB a, bt;
            const half2 * pa = (const half2 *) ((is_q ? sQ : sK) + (16*tb + lane % 16)*LK + 16*kb + 8*(lane / 16));
            const half2 * pb = (const half2 *) (sK + (16*sbk + lane % 16)*LK + 16*kb + 8*(lane / 16));
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                a.x[l]  = pa[l];
                bt.x[l] = pb[l];
            }
            mma(acc, a, bt);
        }
#pragma unroll
        for (int l = 0; l < tile_D::ne; ++l) {
            const int t = 16*tb + tile_D::get_i(l), s = 16*sbk + tile_D::get_j(l);
            const float decay = s <= t ? expf(sb[t] - sb[s]) : 0.0f;
            if (is_q) {
                P[t*C + s] = __float2half(decay * acc.x[l]);
            } else {
                sA[t*(C + 4) + s] = s < t ? sbeta[t] * decay * acc.x[l] : 0.0f;
            }
        }
    }
    // gamma Q to global while Q is still in SRAM.
    half * Qg = (half *) (out + off_Q);
#pragma unroll 8
    for (int e = tid; e < C*D; e += 256) {
        const int t = e / D, i = e % D;
        Qg[t*D + i] = __float2half(sgam[t] * __half2float(sQ[t*LK + i]));
    }
    __syncthreads();

    // T = (I + A)^-1 column by column, 4 lanes per column split the sums: lane p keeps xs[m] = T[4m + p][col].
    float xs[C/4];
    const int col = tid / 4, p = tid % 4;
#pragma unroll
    for (int m = 0; m < C/4; ++m) {
        xs[m] = 4*m + p == col ? 1.0f : 0.0f;
    }
#pragma unroll
    for (int t = 1; t < C; ++t) {
        float part = 0.0f;
#pragma unroll
        for (int m = 0; m < (t + 3)/4; ++m) {
            if (4*m + p < t) {
                part += sA[t*(C + 4) + 4*m + p] * xs[m];
            }
        }
        part += __shfl_xor_sync(0xFFFFFFFF, part, 1, WARP_SIZE);
        part += __shfl_xor_sync(0xFFFFFFFF, part, 2, WARP_SIZE);
        if (t > col && (t % 4) == p) {
            xs[t/4] = -part;
        }
    }
    // K^T into region 1 (Q is no longer needed), with the Kt^T store to global.
    half * KT = (half *) (out + off_KT);
    const float gC = sgam[C - 1];
#pragma unroll 8
    for (int e = tid; e < C*D; e += 256) {
        const int i = e / C, t = e % C;
        const half kv = sK[t*LK + i];
        sKT[i*LT + t] = kv;
        KT[i*C + t] = __float2half(expf(sb[C - 1] - sb[t]) * __half2float(kv));
    }
    __syncthreads();
    // V^T into region 0 (K is no longer needed); T diag(beta gamma) and T diag(beta) into region 2 (A is no longer needed).
#pragma unroll 8
    for (int e = tid; e < C*D; e += 256) {
        const int t = e / D, j = e % D;
        const int64_t tt = t0 + t;
        const int64_t tc = tt < n_tokens ? tt : t_last;
        const float m = tt < n_tokens ? 1.0f : 0.0f;
        sVT[j*LT + t] = __float2half(m*v[seq*sv3 + tc*sv2 + h*sv1 + j]);
    }
#pragma unroll
    for (int m = 0; m < C/4; ++m) {
        const int t = 4*m + p;
        sT1[t*LT + col] = __float2half(xs[m] * sbeta[col] * sgam[col]);
        sT2[t*LT + col] = __float2half(xs[m] * sbeta[col]);
    }
    __syncthreads();

    // W' = -T diag(beta gamma) K and U0 = T diag(beta) V: 32 output tiles each, 8 per warp.
    half  * W   = (half  *) (out + off_W);
    float * U0T = (float *) (out + off_U0);
    for (int tile_idx = warp; tile_idx < 64; tile_idx += 8) {
        const bool is_v = tile_idx >= 32;
        const int tb = (tile_idx % 32) / 8, nb = (tile_idx % 32) % 8;
        const half * TA = is_v ? sT2 : sT1;
        const half * BT = is_v ? sVT : sKT;
        tile_D acc;
#pragma unroll
        for (int kb = 0; kb <= tb; ++kb) { // T is lower triangular
            tile_AB a, bt;
            const half2 * pa = (const half2 *) (TA + (16*tb + lane % 16)*LT + 16*kb + 8*(lane / 16));
            const half2 * pb = (const half2 *) (BT + (16*nb + lane % 16)*LT + 16*kb + 8*(lane / 16));
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                a.x[l]  = pa[l];
                bt.x[l] = pb[l];
            }
            mma(acc, a, bt);
        }
        const int n = 16*nb + tile_D::get_j(0);
        if (is_v) {
            // 8 consecutive t per lane.
            float4 * dst4 = (float4 *) (U0T + n*C + 16*tb + tile_D::get_i(0));
            dst4[0] = make_float4(acc.x[0], acc.x[1], acc.x[2], acc.x[3]);
            dst4[1] = make_float4(acc.x[4], acc.x[5], acc.x[6], acc.x[7]);
        } else {
#pragma unroll
            for (int l = 0; l < tile_D::ne; ++l) {
                W[(16*tb + tile_D::get_i(l))*D + n] = __float2half(-acc.x[l]);
            }
        }
    }
    if (tid == 0) {
        *(float *) (out + off_G) = gC;
    }
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, scratch, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, n_chunks,
        growth_flag);
    NO_DEVICE_CODE;
#endif
}

// One wave per (head, group of 16 state columns); the state stays in registers as f32 tiles.
static __global__ void __launch_bounds__(32)
gated_delta_net_chunk_scan(const char * scratch, const float * curr_state, float * dst, float * state,
        int64_t H, int64_t n_tokens, int n_chunks, float scale, const int * growth_flag,
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    using namespace ggml_cuda_mma;
    using namespace gdn_chunk;
    typedef tile<16,  8, half2, DATA_LAYOUT_I_MAJOR> tile_AB;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_D;

    const int h = blockIdx.x, J0 = 16*blockIdx.y, seq = blockIdx.z;
    const int lane = threadIdx.x;
    const int j = lane % 16, ihalf = 8*(lane / 16);

    // S[b] holds rows i = 16b + ihalf + l of column J0 + j (state stored transposed: M[col][i] = S[i][col]).
    tile_D S[D/16];
    const float * s_in = curr_state + (int64_t(seq)*H + h)*D*D + int64_t(J0 + j)*D;
#pragma unroll
    for (int b = 0; b < D/16; ++b) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            S[b].x[l] = s_in[16*b + ihalf + l];
        }
    }

    if (growth_flag[seq*H + h]) {
        // Recurrent fallback: lanes j and j+16 hold 64 rows each of column J0 + j.
        const uint32_t iq1 = fastmodulo(h, neqk1_magic);
        const uint32_t iq3 = fastdiv(seq, rq3_magic);
        for (int64_t t = 0; t < n_tokens; ++t) {
            const float * k_t = k + iq3*sq3 + t*sq2 + iq1*sq1;
            const float * q_t = q + iq3*sq3 + t*sq2 + iq1*sq1;
            const int64_t gb = seq*sb3 + t*sb2 + h*sb1;
            const float g_val = expf(g[gb]);
            const float beta_val = beta[gb];
            float kv = 0.0f;
#pragma unroll
            for (int b = 0; b < D/16; ++b) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    kv += S[b].x[l] * k_t[16*b + ihalf + l];
                }
            }
            kv += __shfl_xor_sync(0xFFFFFFFF, kv, 16, WARP_SIZE);
            const float delta = (v[seq*sv3 + t*sv2 + h*sv1 + J0 + j] - g_val*kv) * beta_val;
            float o = 0.0f;
#pragma unroll
            for (int b = 0; b < D/16; ++b) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    S[b].x[l] = g_val*S[b].x[l] + k_t[16*b + ihalf + l]*delta;
                    o += S[b].x[l] * q_t[16*b + ihalf + l];
                }
            }
            o += __shfl_xor_sync(0xFFFFFFFF, o, 16, WARP_SIZE);
            if (lane < 16) {
                dst[((int64_t(seq)*n_tokens + t)*H + h)*D + J0 + j] = o * scale;
            }
        }
        float * s_out = state + (int64_t(seq)*H + h)*D*D + int64_t(J0 + j)*D;
#pragma unroll
        for (int b = 0; b < D/16; ++b) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                s_out[16*b + ihalf + l] = S[b].x[l];
            }
        }
        return;
    }

    const auto to_B = [](const tile_D & t) {
        tile_AB r;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            r.x[l] = make_half2(t.x[2*l], t.x[2*l + 1]);
        }
        return r;
    };
    const auto load_A = [&](const half * M, const int ld, const int r0, const int k0) {
        tile_AB a;
        const half2 * p = (const half2 *) (M + (r0 + lane % 16)*ld + k0 + ihalf);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            a.x[l] = p[l];
        }
        return a;
    };

    for (int c = 0; c < n_chunks; ++c) {
        const char * in = scratch + (int64_t(seq)*H + h)*n_chunks*stride + int64_t(c)*stride;
        const half  * W   = (const half  *) (in + off_W);
        const half  * Qg  = (const half  *) (in + off_Q);
        const half  * KT  = (const half  *) (in + off_KT);
        const half  * P   = (const half  *) (in + off_P);
        const float * U0T = (const float *) (in + off_U0) + (J0 + j)*C;
        const float gC = *(const float *) (in + off_G);

        tile_AB SB[D/16];
#pragma unroll
        for (int b = 0; b < D/16; ++b) {
            SB[b] = to_B(S[b]);
        }

        // U = U0 - W S
        tile_D U[C/16];
#pragma unroll
        for (int tb = 0; tb < C/16; ++tb) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                U[tb].x[l] = U0T[16*tb + ihalf + l];
            }
#pragma unroll
            for (int b = 0; b < D/16; ++b) {
                mma(U[tb], load_A(W, D, 16*tb, 16*b), SB[b]);
            }
        }
        tile_AB UB[C/16];
#pragma unroll
        for (int sb = 0; sb < C/16; ++sb) {
            UB[sb] = to_B(U[sb]);
        }

        // O = gamma Q S + P U, causal: P tiles above the diagonal are zero.
#pragma unroll
        for (int tb = 0; tb < C/16; ++tb) {
            tile_D O;
#pragma unroll
            for (int b = 0; b < D/16; ++b) {
                mma(O, load_A(Qg, D, 16*tb, 16*b), SB[b]);
            }
#pragma unroll
            for (int sb = 0; sb <= tb; ++sb) {
                mma(O, load_A(P, C, 16*tb, 16*sb), UB[sb]);
            }
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int64_t t = int64_t(c)*C + 16*tb + ihalf + l;
                if (t < n_tokens) {
                    dst[((int64_t(seq)*n_tokens + t)*H + h)*D + J0 + j] = O.x[l] * scale;
                }
            }
        }

        // S' = gamma_C S + Kt^T U
#pragma unroll
        for (int b = 0; b < D/16; ++b) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                S[b].x[l] *= gC;
            }
#pragma unroll
            for (int sb = 0; sb < C/16; ++sb) {
                mma(S[b], load_A(KT, C, 16*b, 16*sb), UB[sb]);
            }
        }
    }

    float * s_out = state + (int64_t(seq)*H + h)*D*D + int64_t(J0 + j)*D;
#pragma unroll
    for (int b = 0; b < D/16; ++b) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            s_out[16*b + ihalf + l] = S[b].x[l];
        }
    }
#else
    GGML_UNUSED_VARS(scratch, curr_state, dst, state, H, n_tokens, n_chunks, scale, growth_flag, q, k, v, g, beta,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic);
    NO_DEVICE_CODE;
#endif
}
#endif

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream,
        const int32_t * state_indices, int64_t state_row_stride) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // Use smaller workgroups for state updates during decode on RDNA4.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
#if defined(GGML_USE_HIP)
    static const bool prefill_clustered = []() {
        const char * env = getenv("GGML_HIP_GDN_PREFILL_GEOMETRY");
        return env && std::atoi(env) == 3;
    }();
    if constexpr (!KDA) {
        if (prefill_clustered && GGML_CUDA_CC_IS_RDNA4(cc) && S_v == 128 && n_tokens >= 32 && !state_indices) {
            const ggml_cuda_kernel_launch_params params(dim3(H,n_seqs,16),dim3(8,8),0,stream);
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128,false,keep_rs_t,8,8>,params,
                q_d,k_d,v_d,g_d,b_d,s_d,dst_d,state_d,H,
                n_tokens,n_seqs,sq1,sq2,sq3,sv1,sv2,sv3,
                sb1,sb2,sb3,neqk1_magic,rq3_magic,scale,state_slot_stride,K,state_indices,state_row_stride);
            return;
        }
    }
#endif
    if (GGML_CUDA_CC_IS_RDNA4(cc) && S_v == 128 && n_tokens <= 8) {
#if defined(GGML_USE_HIP)
        static const bool clustered = []() {
            const char * env = getenv("GGML_HIP_GDN_CLUSTER_EXACT");
            return env && std::atoi(env) != 0;
        }();
        if constexpr (!KDA) {
            if ((clustered || state_indices) && n_tokens >= 1 && n_tokens <= 4) {
                const ggml_cuda_kernel_launch_params params(dim3(H, n_seqs, S_v / 8), dim3(8, 8, 1), 0, stream);
                if (state_indices) {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, keep_rs_t, 8, 8, true>, params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_indices, state_row_stride);
                } else {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, keep_rs_t, 8, 8>, params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_indices, state_row_stride);
                }
                return;
            }
        }
#endif
        const ggml_cuda_kernel_launch_params decode_params(dim3(H, n_seqs, S_v / 2), dim3(32, 2, 1), 0, stream);
        ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, 2>, decode_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_indices, state_row_stride);
        return;
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_indices, state_row_stride);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_indices, state_row_stride);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_indices, state_row_stride);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_indices, state_row_stride);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_tensor * gathered = nullptr) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d = static_cast<const float *>(gathered ? gathered->src[0]->data : src_state->data);
    const int32_t * state_indices = gathered ? static_cast<const int32_t *>(gathered->src[1]->data) : nullptr;
    const int64_t state_row_stride = gathered ? gathered->src[0]->nb[1] / sizeof(float) : 0;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if defined(GGML_USE_HIP)
    // Chunked prefill: final state only, one state per sequence, scalar decay.
    static const bool chunked = [] {
        const char * env = getenv("GGML_HIP_GDN_CHUNKED");
        return !env || std::atoi(env) != 0;
    }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // With snapshots (speculative decoding keeps the last K states) the chunks stop at a chunk boundary before the last
    // K tokens and the recurrent kernel finishes the tail from the intermediate state.
    const int64_t T1 = keep_rs ? ((n_tokens - K) / gdn_chunk::C) * gdn_chunk::C : n_tokens;
    if (chunked && !kda && (!keep_rs || (n_seqs == 1 && T1 >= gdn_chunk::C)) && !state_indices && S_v == gdn_chunk::D &&
            n_tokens >= 256 && GGML_CUDA_CC_IS_RDNA4(cc)) {
        const int n_chunks = int((T1 + gdn_chunk::C - 1) / gdn_chunk::C);
        ggml_cuda_pool_alloc<char> scratch(ctx.pool(), size_t(n_seqs)*H*n_chunks*gdn_chunk::stride);
        ggml_cuda_pool_alloc<int>  growth_flag(ctx.pool(), n_seqs*H);
        CUDA_CHECK(cudaMemsetAsync(growth_flag.get(), 0, n_seqs*H*sizeof(int), stream));
        const uint3 neqk1_magic = init_fastdiv_values(neqk1);
        const uint3 rq3_magic   = init_fastdiv_values(rq3);
        ggml_cuda_pool_alloc<float> mid_state(ctx.pool(), keep_rs ? size_t(S_v)*S_v*H : 0);
        gated_delta_net_chunk_prepare<<<dim3(n_chunks, H, n_seqs), dim3(32, 8), 0, stream>>>(q_d, k_d, v_d, g_d, b_d, scratch.get(),
            H, T1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, n_chunks, growth_flag.get());
        CUDA_CHECK(cudaGetLastError());
        gated_delta_net_chunk_scan<<<dim3(H, gdn_chunk::D/16, n_seqs), 32, 0, stream>>>(scratch.get(), s_d, dst_d,
            keep_rs ? mid_state.get() : state_d, H, T1, n_chunks, scale, growth_flag.get(), q_d, k_d, v_d, g_d, b_d,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic);
        CUDA_CHECK(cudaGetLastError());
        if (keep_rs) {
            // One sequence: token offsets only.
            launch_gated_delta_net<false, true>(q_d + T1*sq2, k_d + T1*sq2, v_d + T1*sv2, g_d + T1*sb2, b_d + T1*sb2,
                mid_state.get(), dst_d + T1*S_v*H, state_d, S_v, H, n_tokens - T1, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, nullptr, 0);
        }
        return;
    }
#endif

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_indices, state_row_stride);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_indices, state_row_stride);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_indices, state_row_stride);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, state_indices, state_row_stride);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

#if defined(GGML_USE_HIP)
void ggml_cuda_op_gdn_indexed(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gathered, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, cache, gathered);
}
#endif
