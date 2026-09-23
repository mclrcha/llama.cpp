#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"
#include "fattn-rdna4.cuh"

#include <cstdlib>
#include <type_traits>

// FlashAttention for prefill on RDNA4 with DKQ == DV == 256.
//
// A block handles 64 consecutive tokens of one Q head with 8 waves: 4 row groups of 16 tokens, each with a pair of waves.
// Per step of 32 KV rows, wave c of a pair computes S^T = K Q^T for KV rows [16c, 16c + 16) over the full head size,
// the pair exchanges S through SRAM, both apply the same online softmax to all 32 rows and wave c accumulates
// O^T = V^T P^T for DV rows [128c, 128c + 128) in f32. Computing the transposed products lets the WMMA C layout of S^T
// serve directly as the B operand of the P V product and keeps the softmax statistics of a token in the lane that also
// holds its output columns. Q (scaled by scale*log2(e)) stays in registers as the B operand of K Q^T.
// K is stored row-major in SRAM (A operand of K Q^T), V transposed (A operand of V^T P^T).

namespace fattn_r4 {
constexpr int D         = 256;
constexpr int BC        = 32;         // KV rows per step
constexpr int K_STRIDE  = D + 8;      // halfs per K row in SRAM (528 bytes, conflict-free 16 byte row reads)
constexpr int VT_STRIDE = BC + 8;     // halfs per transposed V row in SRAM (80 bytes)
constexpr int K_BYTES   = BC*K_STRIDE*2;
constexpr int VT_BYTES  = D*VT_STRIDE*2;
// nwarps waves per block, nwarps/2 row groups of 16 tokens.
constexpr int nq(const int nwarps)   { return 8*nwarps; }
constexpr int smem(const int nwarps) { return K_BYTES + VT_BYTES + nwarps*16*16*4; }
constexpr int KV_MAX_STEP = 256;
}

#if defined(GGML_USE_HIP) && defined(RDNA4)
#define FATTN_RDNA4_AVAILABLE
#endif

typedef _Float16 fattn_r4_half8  __attribute__((ext_vector_type(8)));
typedef float    fattn_r4_float8 __attribute__((ext_vector_type(8)));

#ifdef FATTN_RDNA4_AVAILABLE
// Barrier for SRAM only: __syncthreads also invalidates the vector L0 cache (acquire at workgroup scope in WGP mode).
static __device__ __forceinline__ void fattn_r4_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Value of lane l ^ 16.
static __device__ __forceinline__ float fattn_r4_xor16(const float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
}
#endif // FATTN_RDNA4_AVAILABLE

// Per block of tokens and chunk of 256 KV rows: KV_bounds[.].x = number of chunks up to the last one with a value that is
// not -inf (end of the KV loop), KV_bounds[.].y = first chunk with a value that is not 0 (below it the mask is skipped).
static __global__ void fattn_rdna4_kv_bounds(
        const char * __restrict__ mask, int2 * __restrict__ KV_bounds, const int ne01, const int ne11,
        const int nb31, const int64_t nb33, const int nq) {
    const int tile  = blockIdx.x;
    const int chunk = blockIdx.y;
    const int seq   = blockIdx.z;
    const int t0    = tile*nq;
    const int t1    = min(t0 + nq, ne01);
    const int col   = chunk*fattn_r4::KV_MAX_STEP + threadIdx.x;
    const char * mask_s = mask + seq*nb33;

    int any_ok = 0;
    int any_nz = 0;
    if (col < ne11) {
        for (int t = t0; t < t1; ++t) {
            const float v = __half2float(((const half *) (mask_s + int64_t(t)*nb31))[col]);
            any_ok |= !isinf(v);
            any_nz |= v != 0.0f;
        }
    }
    any_ok = __syncthreads_or(any_ok);
    any_nz = __syncthreads_or(any_nz);
    if (threadIdx.x == 0) {
        int2 * b = KV_bounds + seq*gridDim.x + tile;
        if (any_ok) {
            atomicMax(&b->x, chunk + 1);
        }
        if (any_nz) {
            atomicMin(&b->y, chunk);
        }
    }
}

static __global__ void fattn_rdna4_kv_bounds_init(int2 * __restrict__ KV_bounds, const int n, const int nchunks) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        KV_bounds[i] = make_int2(0, nchunks);
    }
}

template <int NW, bool use_mask>
__launch_bounds__(NW*WARP_SIZE, 1)
static __global__ void fattn_rdna4_d256(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, const int2 * __restrict__ KV_bounds, float * __restrict__ dst,
        const float scale_log2, const int ne01, const int ne02, const int ne11, const int gqa_ratio,
        const int nb01, const int nb02, const int64_t nb03,
        const int nb11, const int nb12, const int64_t nb13,
        const int nb21, const int nb22, const int64_t nb23,
        const int nb31, const int64_t nb33) {
#ifdef FATTN_RDNA4_AVAILABLE
    using namespace fattn_r4;
    typedef fattn_r4_half8  half8;
    typedef fattn_r4_float8 float8;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int rg   = w >> 1;      // row group: tokens [16 rg, 16 rg + 16) of the block
    const int c    = w & 1;       // KV half of S and DV half of O
    const int h    = lane >> 4;
    const int lq   = lane & 15;

    const int tile = blockIdx.x;
    const int head = blockIdx.y;
    const int seq  = blockIdx.z;
    const int kvh  = head / gqa_ratio;
    const int tq   = tile*nq(NW) + rg*16 + lq;   // token of the Q/S/O column held by this lane
    const bool tq_ok = tq < ne01;

    extern __shared__ __align__(16) char fattn_r4_smem[];
    half  * sK  = (half  *)  fattn_r4_smem;
    half  * sVT = (half  *) (fattn_r4_smem + K_BYTES);
    float * sS  = (float *) (fattn_r4_smem + K_BYTES + VT_BYTES);

    // Q^T as B operand: lane holds token tq, head dims [16 k + 8 h, 16 k + 8 h + 8).
    half8 qb[D/16];
    {
        const float * Qp = (const float *) (Q + seq*nb03 + int64_t(head)*nb02 + int64_t(tq_ok ? tq : 0)*nb01);
#pragma unroll
        for (int k = 0; k < D/16; ++k) {
            const float4 a = *(const float4 *) (Qp + 16*k + 8*h);
            const float4 b = *(const float4 *) (Qp + 16*k + 8*h + 4);
            const float s = tq_ok ? scale_log2 : 0.0f;
            qb[k][0] = (_Float16) (a.x*s); qb[k][1] = (_Float16) (a.y*s); qb[k][2] = (_Float16) (a.z*s); qb[k][3] = (_Float16) (a.w*s);
            qb[k][4] = (_Float16) (b.x*s); qb[k][5] = (_Float16) (b.y*s); qb[k][6] = (_Float16) (b.z*s); qb[k][7] = (_Float16) (b.w*s);
        }
    }

    const char * Kh = K + seq*nb13 + int64_t(kvh)*nb12;
    const char * Vh = V + seq*nb23 + int64_t(kvh)*nb22;
    const char * mrow = use_mask ? mask + seq*nb33 + int64_t(tq_ok ? tq : 0)*nb31 : nullptr;

    // KV rows [0, kv_end) are processed, the mask is added for rows >= kv_nm.
    int kv_end = ne11;
    int kv_nm  = 0;
    if (use_mask) {
        const int2 b = KV_bounds[seq*gridDim.x + tile];
        kv_end = min(b.x*KV_MAX_STEP, ne11);
        kv_nm  = min(b.y*KV_MAX_STEP, kv_end);
    }

    // Register staging of the next K/V tile.
    // K: 32/NW chunks of 16 bytes per thread, rows tid/32 + NW i, head dims [8 (tid%32), 8 (tid%32) + 8).
    // V: rows 2 lq and 2 lq + 1, head dim chunks (32/NW) w + h (+ 2 for NW == 8): half waves 8 dims apart, no bank conflicts.
    // The mask is loaded first: waits on the load counter are in order.
    static_assert(NW == 8 || NW == 16, "bad NW");
    const int tid = w*WARP_SIZE + lane;
    const int vcA = (32/NW)*w + h;
    const int vcB = vcA + 2;
    const int krow = tid >> 5;
    const int kcol = tid & 31;
    struct stage {
        int4 m;
        int4 k0, k1, k2, k3;
        int4 v0, v1, v2, v3;
    };
    auto load_mask = [&](stage & st, const int kv0) {
        st.m = *(const int4 *) (mrow + int64_t(kv0 + 16*c + 8*h)*sizeof(half));
    };
    auto load_tile = [&](stage & st, const int kv0, auto with_mask) {
        if constexpr (decltype(with_mask)::value) {
            load_mask(st, kv0);
        }
        const char * k0 = Kh + int64_t(kv0 + krow)*nb11 + kcol*16;
        st.k0 = *(const int4 *) (k0);
        st.k1 = *(const int4 *) (k0 + NW*int64_t(nb11));
        if constexpr (NW == 8) {
            st.k2 = *(const int4 *) (k0 + 16*int64_t(nb11));
            st.k3 = *(const int4 *) (k0 + 24*int64_t(nb11));
        }
        const char * v0 = Vh + int64_t(kv0 + 2*lq)*nb21;
        st.v0 = *(const int4 *) (v0        + vcA*16);
        st.v1 = *(const int4 *) (v0 + nb21 + vcA*16);
        if constexpr (NW == 8) {
            st.v2 = *(const int4 *) (v0        + vcB*16);
            st.v3 = *(const int4 *) (v0 + nb21 + vcB*16);
        }
    };

    float8 O[8];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            O[dt][i] = 0.0f;
        }
    }
    float m = -1e30f;
    float l = 0.0f;

    auto step = [&](stage & st, const int kv0, auto with_mask) {
        constexpr bool M = decltype(with_mask)::value;
        fattn_r4_lds_barrier(); // previous step is done with K, V^T and S in SRAM

        {
            half * k0 = sK + krow*K_STRIDE + kcol*8;
            *(int4 *) (k0)               = st.k0;
            *(int4 *) (k0 + NW*K_STRIDE) = st.k1;
            if constexpr (NW == 8) {
                *(int4 *) (k0 + 16*K_STRIDE) = st.k2;
                *(int4 *) (k0 + 24*K_STRIDE) = st.k3;
            }
        }
        {
            // Head dim i of rows 2 lq and 2 lq + 1 packed as one 32 bit word.
            const uint32_t a0[4] = {uint32_t(st.v0.x), uint32_t(st.v0.y), uint32_t(st.v0.z), uint32_t(st.v0.w)};
            const uint32_t a1[4] = {uint32_t(st.v1.x), uint32_t(st.v1.y), uint32_t(st.v1.z), uint32_t(st.v1.w)};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                *(uint32_t *) (sVT + (8*vcA + 2*j    )*VT_STRIDE + 2*lq) = (a0[j] & 0xFFFFu) | (a1[j] << 16);
                *(uint32_t *) (sVT + (8*vcA + 2*j + 1)*VT_STRIDE + 2*lq) = (a0[j] >> 16)     | (a1[j] & 0xFFFF0000u);
            }
            if constexpr (NW == 8) {
                const uint32_t b0[4] = {uint32_t(st.v2.x), uint32_t(st.v2.y), uint32_t(st.v2.z), uint32_t(st.v2.w)};
                const uint32_t b1[4] = {uint32_t(st.v3.x), uint32_t(st.v3.y), uint32_t(st.v3.z), uint32_t(st.v3.w)};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    *(uint32_t *) (sVT + (8*vcB + 2*j    )*VT_STRIDE + 2*lq) = (b0[j] & 0xFFFFu) | (b1[j] << 16);
                    *(uint32_t *) (sVT + (8*vcB + 2*j + 1)*VT_STRIDE + 2*lq) = (b0[j] >> 16)     | (b1[j] & 0xFFFF0000u);
                }
            }
        }
        fattn_r4_lds_barrier();

        const int4 mcur = st.m;
        if (kv0 + BC < kv_end) {
            load_tile(st, kv0 + BC, with_mask);
        }

        // S^T for KV rows [16 c, 16 c + 16): lane holds token tq, KV rows 16 c + 8 h + i.
        // Two accumulators (even/odd head dim blocks) halve the chain of dependent WMMAs.
        float8 s;
        float8 s1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s[i]  = 0.0f;
            s1[i] = 0.0f;
        }
        {
            // SRAM reads one pair of head dim blocks ahead of the WMMAs.
            const half * kr = sK + (16*c + lq)*K_STRIDE + 8*h;
            half8 a0n = *(const half8 *) (kr);
            half8 a1n = *(const half8 *) (kr + 16);
#pragma unroll
            for (int k = 0; k < D/16; k += 2) {
                const half8 a0 = a0n;
                const half8 a1 = a1n;
                if (k + 2 < D/16) {
                    a0n = *(const half8 *) (kr + 16*k + 32);
                    a1n = *(const half8 *) (kr + 16*k + 48);
                }
                s  = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, qb[k],     s);
                s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, qb[k + 1], s1);
            }
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 0); // DS read
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // WMMA
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // WMMA
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s[i] += s1[i];
        }
        if constexpr (M) {
            const half8 mk = *(const half8 *) &mcur;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[i] += 1.4426950408889634f*float(mk[i]);
            }
        }

        float * sSw = sS + w*256;
        *(float4 *) (sSw +      4*lane) = make_float4(s[0], s[1], s[2], s[3]);
        *(float4 *) (sSw + 128 + 4*lane) = make_float4(s[4], s[5], s[6], s[7]);
        fattn_r4_lds_barrier();
        float8 sp;
        {
            const float * sSp = sS + (w ^ 1)*256;
            const float4 a = *(const float4 *) (sSp +       4*lane);
            const float4 b = *(const float4 *) (sSp + 128 + 4*lane);
            sp[0] = a.x; sp[1] = a.y; sp[2] = a.z; sp[3] = a.w;
            sp[4] = b.x; sp[5] = b.y; sp[6] = b.z; sp[7] = b.w;
        }

        float mx = s[0];
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            mx = fmaxf(mx, s[i]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            mx = fmaxf(mx, sp[i]);
        }
        mx = fmaxf(mx, fattn_r4_xor16(mx));
        // Lazy rescaling: the reference maximum only moves when the row maximum exceeds it by more than 2^8,
        // P <= 256 stays exact enough in f16 and the result does not depend on the reference.
        const bool  grow  = mx > m + 8.0f;
        const float m_new = grow ? mx : m;
        const float alpha = __builtin_amdgcn_exp2f(m - m_new);
        m = m_new;
        const bool rescale = __builtin_amdgcn_ballot_w32(grow) != 0;

        half8 p_own;
        half8 p_oth;
        float ls = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float po = __builtin_amdgcn_exp2f(s[i]  - m_new);
            const float pp = __builtin_amdgcn_exp2f(sp[i] - m_new);
            ls += po + pp;
            p_own[i] = (_Float16) po;
            p_oth[i] = (_Float16) pp;
        }
        ls += fattn_r4_xor16(ls);
        l = l*alpha + ls;

        const half8 b0 = c == 0 ? p_own : p_oth; // KV rows [0, 16)
        const half8 b1 = c == 0 ? p_oth : p_own; // KV rows [16, 32)

        if (rescale) {
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    O[dt][i] *= alpha;
                }
            }
        }
        {
            // SRAM reads of V^T one head dim tile ahead of the WMMAs.
            const half * vt = sVT + (128*c + lq)*VT_STRIDE + 8*h;
            half8 a0n = *(const half8 *) (vt);
            half8 a1n = *(const half8 *) (vt + 16);
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                const half8 a0 = a0n;
                const half8 a1 = a1n;
                if (dt + 1 < 8) {
                    a0n = *(const half8 *) (vt + 16*(dt + 1)*VT_STRIDE);
                    a1n = *(const half8 *) (vt + 16*(dt + 1)*VT_STRIDE + 16);
                }
                O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, O[dt]);
                O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, O[dt]);
            }
            // Reads of tile dt + 1 are issued before the WMMAs of tile dt.
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 1); // DS read
#pragma unroll
            for (int dt = 0; dt < 6; ++dt) {
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 1); // WMMA
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 1); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 1); // WMMA
        }
    };

    stage sa;
    sa.m = make_int4(0, 0, 0, 0);
    // KV rows below kv_nm: the mask is 0 for all tokens of the block and is neither loaded nor added.
    const std::bool_constant<false> no_mask;
    const std::bool_constant<use_mask> yes_mask;
    int kv0 = 0;
    if (kv_end > 0) {
        if (kv_nm > 0) {
            load_tile(sa, 0, no_mask);
        } else {
            load_tile(sa, 0, yes_mask);
        }
    }
    for (; kv0 + BC < kv_nm; kv0 += BC) {
        step(sa, kv0, no_mask);
    }
    if constexpr (use_mask) {
        // Last step without the mask prefetches the first tile with it.
        if (kv0 < kv_nm) {
            if (kv0 + BC < kv_end) {
                load_mask(sa, kv0 + BC);
            }
            step(sa, kv0, no_mask);
            kv0 += BC;
        }
    }
    for (; kv0 < kv_end; kv0 += BC) {
        step(sa, kv0, yes_mask);
    }

    if (!tq_ok) {
        return;
    }
    const float inv_l = l > 0.0f ? 1.0f/l : 0.0f;
    float * out = dst + (int64_t(seq)*ne01 + tq)*ne02*D + int64_t(head)*D + 128*c + 8*h;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        *(float4 *) (out + 16*dt)     = make_float4(O[dt][0]*inv_l, O[dt][1]*inv_l, O[dt][2]*inv_l, O[dt][3]*inv_l);
        *(float4 *) (out + 16*dt + 4) = make_float4(O[dt][4]*inv_l, O[dt][5]*inv_l, O[dt][6]*inv_l, O[dt][7]*inv_l);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, KV_bounds, dst, scale_log2, ne01, ne02, ne11, gqa_ratio, nb01, nb02, nb03,
        nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33);
    NO_DEVICE_CODE;
#endif // FATTN_RDNA4_AVAILABLE
}

static int fattn_rdna4_min_tokens() {
    static const int v = [] {
        const char * env = getenv("GGML_HIP_FA_D256");
        return env ? std::atoi(env) : 256; // 0 disables, otherwise minimum number of tokens
    }();
    return v;
}

bool ggml_cuda_flash_attn_ext_rdna4_supported(int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    const int min_tokens = fattn_rdna4_min_tokens();
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (min_tokens <= 0 || Q->ne[1] < min_tokens) {
        return false;
    }
    if (Q->ne[0] != fattn_r4::D || K->ne[0] != fattn_r4::D || V->ne[0] != fattn_r4::D) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || sinks) {
        return false;
    }
    if ((K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0) || (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (K->ne[1] % fattn_r4::BC != 0 || K->ne[1] != V->ne[1] || Q->ne[2] % K->ne[2] != 0 || K->ne[2] != V->ne[2]) {
        return false;
    }
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 || Q->nb[2] % 16 || Q->nb[3] % 16 || (uintptr_t) Q->data % 16) {
        return false;
    }
    if (mask) {
        if (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3]) ||
                mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->nb[1] % 16 || (uintptr_t) mask->data % 16) {
            return false;
        }
    }
    // f16 K/V: rows of 16 byte chunks.
    if (K->type == GGML_TYPE_F16 && (K->nb[1] % 16 || K->nb[2] % 16 || K->nb[3] % 16 || (uintptr_t) K->data % 16)) {
        return false;
    }
    if (V->type == GGML_TYPE_F16 && (V->nb[1] % 16 || V->nb[2] % 16 || V->nb[3] % 16 || (uintptr_t) V->data % 16)) {
        return false;
    }
    return true;
#else
    GGML_UNUSED_VARS(device, dst);
    return false;
#endif // defined(GGML_USE_HIP)
}

// Converts a quantized K or V into the f16 buffer reserved after dst (same as launch_fattn).
static const char * fattn_rdna4_to_f16(const ggml_tensor * T, half * T_f16, size_t & nb1, size_t & nb2, size_t & nb3, cudaStream_t stream) {
    const size_t bs = ggml_blck_size(T->type);
    const size_t ts = ggml_type_size(T->type);
    if (ggml_is_contiguously_allocated(T)) {
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(T->type);
        to_fp16(T->data, T_f16, ggml_nelements(T), stream);
        nb1 = nb1*bs*sizeof(half)/ts;
        nb2 = nb2*bs*sizeof(half)/ts;
        nb3 = nb3*bs*sizeof(half)/ts;
    } else {
        GGML_ASSERT(T->nb[0] == ts);
        to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(T->type);
        to_fp16(T->data, T_f16, T->ne[0], T->ne[1], T->ne[2], T->ne[3], nb1/ts, nb2/ts, nb3/ts, stream);
        nb1 = T->ne[0]*sizeof(half);
        nb2 = T->ne[1]*nb1;
        nb3 = T->ne[2]*nb2;
    }
    return (const char *) T_f16;
}

void ggml_cuda_flash_attn_ext_rdna4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    cudaStream_t stream = ctx.stream();

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1], nb12 = K->nb[2], nb13 = K->nb[3];
    if (K->type != GGML_TYPE_F16) {
        GGML_ASSERT(f16_extra.K != 0);
        K_data = fattn_rdna4_to_f16(K, (half *) f16_extra.K, nb11, nb12, nb13, stream);
    }
    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1], nb22 = V->nb[2], nb23 = V->nb[3];
    if (V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21 = nb11; nb22 = nb12; nb23 = nb13;
        } else {
            GGML_ASSERT(f16_extra.V != 0);
            V_data = fattn_rdna4_to_f16(V, (half *) f16_extra.V, nb21, nb22, nb23, stream);
        }
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int ne01   = Q->ne[1];
    const int ne02   = Q->ne[2];
    const int ne03   = Q->ne[3];
    const int ne11   = K->ne[1];
    // 16 waves (128 tokens) per block halve the K/V traffic per FLOP; GGML_HIP_FA_R4_NW=8 selects 64 tokens.
    static const int nw = [] { const char * e = getenv("GGML_HIP_FA_R4_NW"); return e && std::atoi(e) == 8 ? 8 : 16; }();
    const int nq     = fattn_r4::nq(nw);
    const int ntiles = (ne01 + nq - 1)/nq;

    const int     nb31 = mask ? mask->nb[1] : 0;
    const int64_t nb33 = mask && mask->ne[3] > 1 ? mask->nb[3] : 0;

    ggml_cuda_pool_alloc<int2> KV_bounds(ctx.pool());
    if (mask) {
        const int nchunks = (ne11 + fattn_r4::KV_MAX_STEP - 1)/fattn_r4::KV_MAX_STEP;
        const int n = ntiles*ne03;
        KV_bounds.alloc(n);
        fattn_rdna4_kv_bounds_init<<<(n + 255)/256, 256, 0, stream>>>(KV_bounds.get(), n, nchunks);
        fattn_rdna4_kv_bounds<<<dim3(ntiles, nchunks, ne03), fattn_r4::KV_MAX_STEP, 0, stream>>>(
            (const char *) mask->data, KV_bounds.get(), ne01, ne11, nb31, nb33, nq);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 grid(ntiles, ne02, ne03);
    const dim3 block(WARP_SIZE, nw);
    const float scale_log2 = scale*1.4426950408889634f;
    const int gqa_ratio = ne02 / K->ne[2];
#define FATTN_R4_LAUNCH(NW, M) fattn_rdna4_d256<NW, M><<<grid, block, fattn_r4::smem(NW), stream>>>( \
            (const char *) Q->data, K_data, V_data, mask ? (const char *) mask->data : nullptr, mask ? KV_bounds.get() : nullptr, (float *) dst->data, \
            scale_log2, ne01, ne02, ne11, gqa_ratio, Q->nb[1], Q->nb[2], Q->nb[3], \
            nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33)
    if (nw == 16) {
        if (mask) {
            FATTN_R4_LAUNCH(16, true);
        } else {
            FATTN_R4_LAUNCH(16, false);
        }
    } else {
        if (mask) {
            FATTN_R4_LAUNCH(8, true);
        } else {
            FATTN_R4_LAUNCH(8, false);
        }
    }
#undef FATTN_R4_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
