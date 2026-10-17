#include "common.cuh"
#include "convert.cuh"
#include "mmdq-rdna4.cuh"

#include <cstdlib>
#include <type_traits>

// Dense matrix multiplication on RDNA4 for K-quant weights and large batches (prefill).
//
// The weights are dequantized to f16 with the expressions of ggml_get_to_fp16_cuda and the activations are converted to
// f16 (saturating), then WMMA multiplies them with f32 accumulation: the precision of the dequantize + hipBLAS path without
// materializing the f16 weights, and deterministic (fixed summation order, no atomics).
// A block of WM x WN waves computes a (64 WM) x (64 WN) tile of dst^T (weight rows x tokens); wave (wm, wn) owns rows
// [64 wm, 64 wm + 64) and tokens [64 wn, 64 wn + 64) as 4 x 4 WMMA tiles. For every step of 32 values of K the threads
// dequantize the weight tile into SRAM (double buffered, 64/WN values per thread) while each wave loads the f16
// activations of its tokens directly from global memory in the WMMA B layout, one step ahead.

namespace mmdq_r4 {
constexpr int BK = 32;
constexpr int SA = BK + 8; // halfs per weight row in SRAM (80 bytes: conflict-free 16 byte row reads)
}

#if defined(GGML_USE_HIP) && defined(RDNA4)
#define MMDQ_RDNA4_AVAILABLE
#endif

typedef _Float16 mmdq_half8  __attribute__((ext_vector_type(8)));
typedef float    mmdq_float8 __attribute__((ext_vector_type(8)));

// Raw data of VPT consecutive values of a 32-value sub-block window.
template <ggml_type type, int VPT> struct mmdq_raw;

// Quant bytes of the pair of sub-blocks (low nibbles: even sub-block, high nibbles: odd sub-block).
template <int VPT> struct mmdq_raw<GGML_TYPE_Q4_K, VPT> { int qs[VPT/4]; uint32_t dm; uint32_t sc[3]; };
template <int VPT> struct mmdq_raw<GGML_TYPE_Q5_K, VPT> { int qs[VPT/4]; int qh[VPT/4]; uint32_t dm; uint32_t sc[3]; };
// Blocks of 210 bytes: the vector loads are only 2 byte aligned (unaligned global loads are supported on RDNA4).
template <int VPT> struct mmdq_raw<GGML_TYPE_Q6_K, VPT> { int ql[VPT/4]; int qh[VPT/4]; int sc; half d; };

template <ggml_type type> constexpr int mmdq_block_bytes();
template <> constexpr int mmdq_block_bytes<GGML_TYPE_Q4_K>() { return sizeof(block_q4_K); }
template <> constexpr int mmdq_block_bytes<GGML_TYPE_Q5_K>() { return sizeof(block_q5_K); }
template <> constexpr int mmdq_block_bytes<GGML_TYPE_Q6_K>() { return sizeof(block_q6_K); }

#ifdef MMDQ_RDNA4_AVAILABLE
// Barrier for SRAM only: __syncthreads also invalidates the vector L0 cache (acquire at workgroup scope in WGP mode).
static __device__ __forceinline__ void mmdq_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int N>
static __device__ __forceinline__ void mmdq_load_words(int * dst, const uint8_t * src) {
    if constexpr (N == 4) {
        const int4 v = *(const int4 *) src;
        dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
    } else {
        static_assert(N == 2, "bad word count");
        const int2 v = *(const int2 *) src;
        dst[0] = v.x; dst[1] = v.y;
    }
}

// Raw data of values [32 s + VPT p, 32 s + VPT p + VPT) of the super-block at blk.
template <ggml_type type, int VPT>
static __device__ __forceinline__ void mmdq_load(mmdq_raw<type, VPT> & r, const char * blk, const int s, const int p) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        typedef std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K> block_t;
        const block_t * b = (const block_t *) blk;
        mmdq_load_words<VPT/4>(r.qs, b->qs + 32*(s >> 1) + VPT*p);
        if constexpr (type == GGML_TYPE_Q5_K) {
            mmdq_load_words<VPT/4>(r.qh, b->qh + VPT*p);
        }
        r.dm = *(const uint32_t *) &b->dm;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            r.sc[i] = ((const uint32_t *) b->scales)[i];
        }
    } else {
        static_assert(type == GGML_TYPE_Q6_K, "bad type");
        const block_q6_K * b = (const block_q6_K *) blk;
        const int nh = s >> 2;
        const int qd = s & 3;
        mmdq_load_words<VPT/4>(r.ql, b->ql + 64*nh + 32*(qd & 1) + VPT*p);
        mmdq_load_words<VPT/4>(r.qh, b->qh + 32*nh + VPT*p);
        r.sc = b->scales[8*nh + 2*qd + VPT*p/16];
        r.d  = b->d;
    }
}

// The VPT f16 values of mmdq_load, computed as dequantize_q4_K / q5_K / q6_K do.
template <ggml_type type, int VPT>
static __device__ __forceinline__ void mmdq_dequant(const mmdq_raw<type, VPT> & r, const int s, half2 * y) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        // get_scale_min_k4(s, scales) on the 3 words of the scales, without an indexed local array (scratch)
        const int sh = 8*(s & 3);
        const uint32_t b0 = (r.sc[0] >> sh) & 0xFF;
        const uint32_t b1 = (r.sc[1] >> sh) & 0xFF;
        const uint32_t b2 = (r.sc[2] >> sh) & 0xFF;
        const int sc = s < 4 ? b0 & 63 : (b2 & 0xF) | ((b0 >> 6) << 4);
        const int m  = s < 4 ? b1 & 63 : (b2 >>  4) | ((b1 >> 6) << 4);
        const half2 dm = *(const half2 *) &r.dm;
        const float dall = __low2half(dm);
        const float dmin = __high2half(dm);
        const float d1 = dall * sc; const float m1 = dmin * m;
        const int shift = 4*(s & 1);
#pragma unroll
        for (int i = 0; i < VPT/4; ++i) {
            // 4 quants per word: nibble | high bit << 4 (= nibble + 16 for a set bit), converted byte by byte
            int q = (r.qs[i] >> shift) & 0x0F0F0F0F;
            if constexpr (type == GGML_TYPE_Q5_K) {
                q |= ((r.qh[i] >> s) & 0x01010101) << 4;
            }
            float v[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                v[b] = d1 * ((q >> (8*b)) & 0xFF) - m1;
            }
            y[2*i + 0] = make_half2(ggml_cuda_cast<half>(v[0]), ggml_cuda_cast<half>(v[1]));
            y[2*i + 1] = make_half2(ggml_cuda_cast<half>(v[2]), ggml_cuda_cast<half>(v[3]));
        }
    } else {
        const float d = r.d;
        const int qd = s & 3;
        const int lshift = 4*(qd >> 1);
        const int hshift = 2*qd;
#pragma unroll
        for (int i = 0; i < VPT/4; ++i) {
            const int q = ((r.ql[i] >> lshift) & 0x0F0F0F0F) | (((r.qh[i] >> hshift) & 0x03030303) << 4);
            float v[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                v[b] = d * r.sc * ((int8_t) ((q >> (8*b)) & 0xFF) - 32);
            }
            y[2*i + 0] = make_half2(ggml_cuda_cast<half>(v[0]), ggml_cuda_cast<half>(v[1]));
            y[2*i + 1] = make_half2(ggml_cuda_cast<half>(v[2]), ggml_cuda_cast<half>(v[3]));
        }
    }
}

template <int VPT>
static __device__ __forceinline__ void mmdq_store(half * dst, const half2 * y) {
#pragma unroll
    for (int i = 0; i < VPT/8; ++i) {
        ((int4 *) dst)[i] = *(const int4 *) (y + 4*i);
    }
}
#endif // MMDQ_RDNA4_AVAILABLE

// dst[n, m] = sum_k W[m, k] X[n, k]; W: M rows of K quantized values (nbw bytes per row), X: f16 [N, K], dst row stride ldy.
template <ggml_type type, int WM, int WN>
__launch_bounds__(WM*WN*WARP_SIZE, 1)
static __global__ void mmdq_rdna4(
        const char * __restrict__ W, const half * __restrict__ X, float * __restrict__ dst,
        const int M, const int N, const int K, const int64_t nbw, const int64_t ldy) {
#ifdef MMDQ_RDNA4_AVAILABLE
    using namespace mmdq_r4;
    typedef mmdq_half8  half8;
    typedef mmdq_float8 float8;
    constexpr int BM  = 64*WM;
    constexpr int BN  = 64*WN;
    constexpr int TPR = WM*WN*WARP_SIZE/BM; // threads per weight row
    constexpr int VPT = BK/TPR;             // values per thread and step
    static_assert(VPT == 8 || VPT == 16, "bad tile");

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int wm   = w % WM;
    const int wn   = w / WM;
    const int l16  = lane & 15;
    const int h    = lane >> 4;
    const int m0   = blockIdx.x*BM;
    const int n0   = blockIdx.y*BN;

    __shared__ __align__(16) half sA[2][BM*SA];

    // Dequantization: thread -> weight row ar, values [VPT ap, VPT ap + VPT) of every step.
    const int tid = w*WARP_SIZE + lane;
    const int ar  = tid / TPR;
    const int ap  = tid % TPR;
    const char * wrow = W + int64_t(min(m0 + ar, M - 1))*nbw;
    half * sa_st[2] = {sA[0] + ar*SA + VPT*ap, sA[1] + ar*SA + VPT*ap};

    // Activations: B operand of token tile j holds values [16 s + 8 h, 16 s + 8 h + 8) of each step half s.
    const half * xb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xb[j] = X + int64_t(min(n0 + 64*wn + 16*j + l16, N - 1))*K + 8*h;
    }

    float8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    const int nsteps = K/BK; // multiple of 8
    mmdq_raw<type, VPT> raw;
    half8 b0[4][2];
    half8 b1[4][2];

    const auto load_b = [&](half8 (&b)[4][2], const int t) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
                b[j][s2] = *(const half8 *) (xb[j] + t*BK + 16*s2);
            }
        }
    };

    // Step t: WMMA on the weights in sA[t & 1] and the activations in bc, while the activations of step t + 1 load into bn
    // and the weights of step t + 1 are dequantized into the other buffer. The two register sets alternate (no copies),
    // so every load has a whole step of WMMA to arrive.
    const auto step = [&](const int t, const half8 (&bc)[4][2], half8 (&bn)[4][2]) {
        const bool more = t + 1 < nsteps;
        if (more) {
            mmdq_load<type, VPT>(raw, wrow + int64_t((t + 1) >> 3)*mmdq_block_bytes<type>(), (t + 1) & 7, ap);
            load_b(bn, t + 1);
        }

        const half * sa = sA[t & 1] + (64*wm + l16)*SA + 8*h;
#pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
            half8 af[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                af[i] = *(const half8 *) (sa + 16*i*SA + 16*s2);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[i], bc[j][s2], acc[i][j]);
                }
            }
        }

        if (more) {
            half2 y[VPT/2];
            mmdq_dequant<type, VPT>(raw, (t + 1) & 7, y);
            mmdq_store<VPT>(sa_st[(t + 1) & 1], y);
        }
        mmdq_lds_barrier();
    };

    mmdq_load<type, VPT>(raw, wrow, 0, ap);
    load_b(b0, 0);
    {
        half2 y[VPT/2];
        mmdq_dequant<type, VPT>(raw, 0, y);
        mmdq_store<VPT>(sa_st[0], y);
    }
    mmdq_lds_barrier();

    for (int t = 0; t < nsteps; t += 2) {
        step(t,     b0, b1);
        step(t + 1, b1, b0);
    }

    // C layout: lane holds token l16 of the tile and weight rows [8 h, 8 h + 8).
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + 64*wm + 16*i + 8*h;
        if (m >= M) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + 64*wn + 16*j + l16;
            if (n >= N) {
                continue;
            }
            float4 * yp = (float4 *) (dst + int64_t(n)*ldy + m);
            yp[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
            yp[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
        }
    }
#else
    GGML_UNUSED_VARS(W, X, dst, M, N, K, nbw, ldy);
    NO_DEVICE_CODE;
#endif // MMDQ_RDNA4_AVAILABLE
}

bool ggml_cuda_mmdq_rdna4_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = [] { const char * e = getenv("GGML_HIP_MMDQ"); return !e || std::atoi(e) != 0; }();
    if (!enabled || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K) {
        return false;
    }
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[0] % QK_K == 0 && src0->ne[1] % 16 == 0 && src1->ne[0] == src0->ne[0] &&
        dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] &&
        (uintptr_t) src0->data % 16 == 0 && (uintptr_t) dst->data % 16 == 0 &&
        src0->ne[1] <= INT_MAX && src1->ne[1] <= 65535*64;
#else
    GGML_UNUSED_VARS(src0, src1, dst);
    return false;
#endif // defined(GGML_USE_HIP)
}

template <ggml_type type, int WM, int WN>
static void mmdq_rdna4_launch(const char * W, const half * X, float * dst, const int M, const int N, const int K,
        const int64_t nbw, const int64_t ldy, cudaStream_t stream) {
    const dim3 grid((M + 64*WM - 1)/(64*WM), (N + 64*WN - 1)/(64*WN));
    const dim3 block(WARP_SIZE, WM*WN);
    mmdq_rdna4<type, WM, WN><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy);
}

template <ggml_type type>
static void mmdq_rdna4_switch_cfg(const char * W, const half * X, float * dst, const int M, const int N, const int K,
        const int64_t nbw, const int64_t ldy, cudaStream_t stream) {
    // Tile of 64 WM x 64 WN: 2 x 4 for q4_K, 2 x 8 (half the dequantization per FLOP) for the costlier q5_K/q6_K unpacking.
    // GGML_HIP_MMDQ_CFG 1: 2 x 4, 2: 2 x 8, 3: 4 x 4 for every type.
    static const int cfg_env = [] { const char * e = getenv("GGML_HIP_MMDQ_CFG"); return e ? std::atoi(e) : 0; }();
    const int cfg = cfg_env ? cfg_env : (type == GGML_TYPE_Q4_K ? 1 : 2);
    switch (cfg) {
        case 2:  mmdq_rdna4_launch<type, 2, 8>(W, X, dst, M, N, K, nbw, ldy, stream); break;
        case 3:  mmdq_rdna4_launch<type, 4, 4>(W, X, dst, M, N, K, nbw, ldy, stream); break;
        default: mmdq_rdna4_launch<type, 2, 4>(W, X, dst, M, N, K, nbw, ldy, stream); break;
    }
}

void ggml_cuda_mmdq_rdna4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int M = src0->ne[1];
    const int K = src0->ne[0];
    const int N = src1->ne[1];

    ggml_cuda_pool_alloc<half> x16(ctx.pool(), int64_t(N)*K);
    ggml_cuda_convert_f32_f16_sat((const float *) src1->data, x16.get(), int64_t(N)*K, stream);

    const char * W = (const char *) src0->data;
    float * y = (float *) dst->data;
    const int64_t ldy = dst->nb[1]/sizeof(float);
    switch (src0->type) {
        case GGML_TYPE_Q4_K: mmdq_rdna4_switch_cfg<GGML_TYPE_Q4_K>(W, x16.get(), y, M, N, K, src0->nb[1], ldy, stream); break;
        case GGML_TYPE_Q5_K: mmdq_rdna4_switch_cfg<GGML_TYPE_Q5_K>(W, x16.get(), y, M, N, K, src0->nb[1], ldy, stream); break;
        case GGML_TYPE_Q6_K: mmdq_rdna4_switch_cfg<GGML_TYPE_Q6_K>(W, x16.get(), y, M, N, K, src0->nb[1], ldy, stream); break;
        default: GGML_ABORT("unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
}
