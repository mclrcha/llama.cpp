#include "common.cuh"
#include "convert.cuh"
#include "mmdq-rdna4.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <cstddef>
#include <type_traits>

// Dense matrix multiplication on RDNA4 for K-quant / IQ4_XS weights and large batches (prefill).
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

int ggml_cuda_mmdq_layout(const ggml_tensor * src) {
    static const bool packed = [] { const char * e = getenv("GGML_HIP_MMDQ_X_LAYOUT"); return e && std::atoi(e) != 0; }();
    return packed && src->ne[0] % 16 == 0 && src->ne[1] % 16 == 0 ? 1 : 0;
}

static __global__ void mmdq_convert_x(const float * src, half * dst, int64_t K, int64_t N, int layout) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < K*N) {
        const float value = src[i];
        dst[ggml_cuda_mmdq_x_index(i/K, i%K, K, layout)] =
            ggml_cuda_cast<half>(value > 65504.0f ? 65504.0f : (value < -65504.0f ? -65504.0f : value));
    }
}

// A byte-exact permutation of 16 rows; the source remains in its original GGUF layout.
static __global__ void mmdq_pack_q5(const uint32_t * src, uint32_t * dst, int64_t words, int blocks_k) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < words) {
        constexpr int QWORDS = sizeof(block_q5_K)/sizeof(uint32_t);
        const int64_t row = i/(int64_t(blocks_k)*QWORDS);
        const int kb = (i/QWORDS)%blocks_k, word = i%QWORDS;
        const int64_t out = (((row/16)*blocks_k + kb)*QWORDS + word)*16 + row%16;
        dst[out] = src[i];
    }
}

// Raw data of VPT consecutive values of a 32-value sub-block window.
template <ggml_type type, int VPT> struct mmdq_raw;

// Quant bytes of the pair of sub-blocks (low nibbles: even sub-block, high nibbles: odd sub-block).
template <int VPT> struct mmdq_raw<GGML_TYPE_Q4_K, VPT> { int qs[VPT/4]; uint32_t dm; uint32_t sc[3]; };
template <int VPT> struct mmdq_raw<GGML_TYPE_Q5_K, VPT> { int qs[VPT/4]; int qh[VPT/4]; uint32_t dm; uint32_t sc[3]; };
// Blocks of 210 bytes: the vector loads are only 2 byte aligned (unaligned global loads are supported on RDNA4).
template <int VPT> struct mmdq_raw<GGML_TYPE_Q6_K, VPT> { int ql[VPT/4]; int qh[VPT/4]; int sc; half d; };
template <int VPT> struct mmdq_raw<GGML_TYPE_IQ4_XS, VPT> { int qs[VPT/4]; uint32_t ds; uint32_t sc; int shift; };

template <ggml_type type> constexpr int mmdq_block_bytes();
template <> constexpr int mmdq_block_bytes<GGML_TYPE_Q4_K>() { return sizeof(block_q4_K); }
template <> constexpr int mmdq_block_bytes<GGML_TYPE_Q5_K>() { return sizeof(block_q5_K); }
template <> constexpr int mmdq_block_bytes<GGML_TYPE_Q6_K>() { return sizeof(block_q6_K); }
template <> constexpr int mmdq_block_bytes<GGML_TYPE_IQ4_XS>() { return sizeof(block_iq4_xs); }

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

// REUSE retains bytes shared by consecutive sub-blocks; calls must start at s=0 and proceed in order.
template <ggml_type type, int VPT, bool REUSE>
static __device__ __forceinline__ void mmdq_load(mmdq_raw<type, VPT> & r, const char * blk, const int s, const int p) {
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        typedef std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K> block_t;
        const block_t * b = (const block_t *) blk;
        if (!REUSE || (s & 1) == 0) {
            mmdq_load_words<VPT/4>(r.qs, b->qs + 32*(s >> 1) + VPT*p);
        }
        if constexpr (type == GGML_TYPE_Q5_K) {
            if (!REUSE || s == 0) {
                mmdq_load_words<VPT/4>(r.qh, b->qh + VPT*p);
            }
        }
        if (!REUSE || s == 0) {
            r.dm = *(const uint32_t *) &b->dm;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                r.sc[i] = ((const uint32_t *) b->scales)[i];
            }
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const block_iq4_xs * b = (const block_iq4_xs *) blk;
        mmdq_load_words<VPT/4>(r.qs, b->qs + 16*s + (VPT*p % 16));
        r.shift = 4*(VPT*p/16);
        if (!REUSE || s == 0) {
            r.ds = *(const uint32_t *) &b->d;
            r.sc = *(const uint32_t *) b->scales_l;
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

template <ggml_type type, int VPT, bool REUSE>
static __device__ __forceinline__ void mmdq_load_weight(mmdq_raw<type, VPT> & r, const char * W,
        int row, int K, int64_t nbw, int t, int p, bool packed) {
    if constexpr (type == GGML_TYPE_Q5_K) {
        if (packed) {
            constexpr int QWORDS = sizeof(block_q5_K)/sizeof(uint32_t);
            const int s = t & 7;
            const uint32_t * base = (const uint32_t *) W +
                (((int64_t(row/16)*(K/256) + t/8)*QWORDS)*16 + row%16);
#pragma unroll
            for (int j = 0; j < VPT/4; ++j) {
                if (!REUSE || (s & 1) == 0) {
                    r.qs[j] = base[(offsetof(block_q5_K, qs)/4 + 8*(s/2) + VPT*p/4 + j)*16];
                }
                if (!REUSE || s == 0) {
                    r.qh[j] = base[(offsetof(block_q5_K, qh)/4 + VPT*p/4 + j)*16];
                }
            }
            if (!REUSE || s == 0) {
                r.dm = base[0];
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    r.sc[j] = base[(offsetof(block_q5_K, scales)/4 + j)*16];
                }
            }
            return;
        }
    }
    mmdq_load<type, VPT, REUSE>(r, W + int64_t(row)*nbw + int64_t(t/8)*mmdq_block_bytes<type>(), t & 7, p);
}

static __device__ __forceinline__ void mmdq_tile(int & m, int & n, int group) {
    m = blockIdx.x;
    n = blockIdx.y;
    if (group) {
        const int linear = blockIdx.y*gridDim.x + blockIdx.x;
        const int first = linear/(group*gridDim.y)*group;
        const int count = min(group, int(gridDim.x) - first);
        const int local = linear - first*gridDim.y;
        m = first + local%count;
        n = local/count;
    }
}

// The VPT f16 values of mmdq_load, computed with the same expressions as the dequantize kernels.
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
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const int ls = ((r.sc >> (4*s)) & 0x0F) | (((r.ds >> (16 + 2*s)) & 3) << 4);
        const half d0 = *(const half *) &r.ds;
        const float d = float(d0) * (ls - 32);
#pragma unroll
        for (int i = 0; i < VPT/4; ++i) {
            const int q = get_int_from_table_16(r.qs[i] >> r.shift, kvalues_iq4nl).x;
            float v[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                v[b] = d * (int8_t) ((q >> (8*b)) & 0xFF);
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
// Validated R84 kernel kept verbatim: experimental indexing must not perturb its register allocation/schedule.
template <ggml_type type, int WM, int WN, bool REUSE, bool EARLY_DEQUANT>
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
            mmdq_load<type, VPT, REUSE>(raw, wrow + int64_t((t + 1) >> 3)*mmdq_block_bytes<type>(), (t + 1) & 7, ap);
            load_b(bn, t + 1);
        }

        const auto store_next = [&] {
            if (more) {
                half2 y[VPT/2];
                mmdq_dequant<type, VPT>(raw, (t + 1) & 7, y);
                mmdq_store<VPT>(sa_st[(t + 1) & 1], y);
            }
        };

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
            if constexpr (EARLY_DEQUANT) {
                if (s2 == 0) {
                    // Leave one half of the current WMMA work after the next tile's dequantization and LDS stores.
                    store_next();
                }
            }
        }

        if constexpr (!EARLY_DEQUANT) {
            store_next();
        }
        mmdq_lds_barrier();
    };

    mmdq_load<type, VPT, REUSE>(raw, wrow, 0, ap);
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


template <ggml_type type, int WM, int WN, bool REUSE, bool EARLY_DEQUANT>
__launch_bounds__(WM*WN*WARP_SIZE, 1)
static __global__ void mmdq_rdna4_experimental(
        const char * __restrict__ W, const half * __restrict__ X, float * __restrict__ dst,
        const int M, const int N, const int K, const int64_t nbw, const int64_t ldy,
        const int layout, const int group, const bool packed_w) {
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
    int mt, nt;
    mmdq_tile(mt, nt, group);
    const int m0   = mt*BM;
    const int n0   = nt*BN;

    __shared__ __align__(16) half sA[2][BM*SA];

    // Dequantization: thread -> weight row ar, values [VPT ap, VPT ap + VPT) of every step.
    const int tid = w*WARP_SIZE + lane;
    const int ar  = tid / TPR;
    const int ap  = tid % TPR;
    const int wrow = min(m0 + ar, M - 1);
    half * sa_st[2] = {sA[0] + ar*SA + VPT*ap, sA[1] + ar*SA + VPT*ap};

    // Activations: B operand of token tile j holds values [16 s + 8 h, 16 s + 8 h + 8) of each step half s.
    const half * xb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xb[j] = X + ggml_cuda_mmdq_x_index(min(n0 + 64*wn + 16*j + l16, N - 1), 8*h, K, layout);
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
                b[j][s2] = *(const half8 *) (xb[j] + (int64_t(t)*BK + 16*s2)*(layout ? 16 : 1));
            }
        }
    };

    // Step t: WMMA on the weights in sA[t & 1] and the activations in bc, while the activations of step t + 1 load into bn
    // and the weights of step t + 1 are dequantized into the other buffer. The two register sets alternate (no copies),
    // so every load has a whole step of WMMA to arrive.
    const auto step = [&](const int t, const half8 (&bc)[4][2], half8 (&bn)[4][2]) {
        const bool more = t + 1 < nsteps;
        if (more) {
            mmdq_load_weight<type, VPT, REUSE>(raw, W, wrow, K, nbw, t + 1, ap, packed_w);
            load_b(bn, t + 1);
        }

        const auto store_next = [&] {
            if (more) {
                half2 y[VPT/2];
                mmdq_dequant<type, VPT>(raw, (t + 1) & 7, y);
                mmdq_store<VPT>(sa_st[(t + 1) & 1], y);
            }
        };

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
            if constexpr (EARLY_DEQUANT) {
                if (s2 == 0) {
                    // Leave one half of the current WMMA work after the next tile's dequantization and LDS stores.
                    store_next();
                }
            }
        }

        if constexpr (!EARLY_DEQUANT) {
            store_next();
        }
        mmdq_lds_barrier();
    };

    mmdq_load_weight<type, VPT, REUSE>(raw, W, wrow, K, nbw, 0, ap, packed_w);
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
    GGML_UNUSED_VARS(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w);
    NO_DEVICE_CODE;
#endif // MMDQ_RDNA4_AVAILABLE
}

template <ggml_type type, int WN, bool INTERLEAVE>
__launch_bounds__(2*WN*WARP_SIZE, 1)
static __global__ void mmdq_rdna4_k64(const char * W, const half * X, float * dst, int M, int N, int K,
        int64_t nbw, int64_t ldy, int layout, int group, bool packed_w) {
#ifdef MMDQ_RDNA4_AVAILABLE
    constexpr int BM = 128, BN = 64*WN, SA = 72;
    constexpr int TPR = 2*WN*WARP_SIZE/BM, VPT = 32/TPR;
    const int lane = threadIdx.x, wave = threadIdx.y;
    const int wm = wave%2, wn = wave/2, h = lane/16, l16 = lane%16;
    int mt, nt;
    mmdq_tile(mt, nt, group);
    const int m0 = mt*BM, n0 = nt*BN;
    const int tid = wave*WARP_SIZE + lane, ar = tid/TPR, ap = tid%TPR;
    const int row = min(m0 + ar, M - 1);
    __shared__ __align__(16) half shared[2][BM*SA];
    mmdq_float8 acc[4][4] = {};
    mmdq_half8 b0[4], b1[4];
    mmdq_raw<type, VPT> carry, raw[2];
    const auto load_weights = [&](int stage) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            mmdq_load_weight<type, VPT, type != GGML_TYPE_Q6_K>(carry, W, row, K, nbw, 2*stage + s, ap, packed_w);
            raw[s] = carry;
        }
    };
    const auto store_weights = [&](int stage) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            half2 values[VPT/2];
            mmdq_dequant<type, VPT>(raw[s], (2*stage + s)&7, values);
            mmdq_store<VPT>(shared[stage&1] + ar*SA + 32*s + VPT*ap, values);
        }
    };
    const auto load_b = [&](int j, int k) {
        const int token = min(n0 + 64*wn + 16*j + l16, N - 1);
        return *(const mmdq_half8 *) (X + ggml_cuda_mmdq_x_index(token, k + 8*h, K, layout));
    };
    load_weights(0);
    store_weights(0);
#pragma unroll
    for (int j = 0; j < 4; ++j) { b0[j] = load_b(j, 0); }
    mmdq_lds_barrier();
    for (int stage = 0; stage < K/64; ++stage) {
        if (stage + 1 < K/64) { load_weights(stage + 1); }
        const auto half_step = [&](int s, const mmdq_half8 (&bc)[4], mmdq_half8 (&bn)[4]) {
            const int next_k = stage*64 + (s + 1)*16;
            if constexpr (!INTERLEAVE) {
                if (next_k < K) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { bn[j] = load_b(j, next_k); }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const mmdq_half8 a = *(const mmdq_half8 *)
                    (shared[stage&1] + (64*wm + 16*i + l16)*SA + 16*s + 8*h);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr (INTERLEAVE) {
                        if (i == 0 && next_k < K) { bn[j] = load_b(j, next_k); }
                    }
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bc[j], acc[i][j]);
                }
            }
        };
        half_step(0, b0, b1);
        half_step(1, b1, b0);
        half_step(2, b0, b1);
        half_step(3, b1, b0);
        if (stage + 1 < K/64) { store_weights(stage + 1); }
        mmdq_lds_barrier();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + 64*wm + 16*i + 8*h;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + 64*wn + 16*j + l16;
            if (m < M && n < N) {
                float4 * out = (float4 *) (dst + int64_t(n)*ldy + m);
                out[0] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                out[1] = make_float4(acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w);
    NO_DEVICE_CODE;
#endif
}

bool ggml_cuda_mmdq_rdna4_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = [] { const char * e = getenv("GGML_HIP_MMDQ"); return !e || std::atoi(e) != 0; }();
    if (!enabled || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_IQ4_XS) {
        return false;
    }
    // Match the early BLAS fallback in ggml_cuda_mul_mat. A fused producer must not omit its F32 output if the
    // consumer will bypass MMDQ (compute-buffer views cannot safely clear quantization padding).
    if (src0->view_src && src0->buffer &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0)) {
        return false;
    }
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        dst->ne[2] == 1 && dst->ne[3] == 1 && src0->ne[0] > 0 && src0->ne[0] <= INT_MAX &&
        src0->ne[1] > 0 && src1->ne[1] > 0 &&
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

template <ggml_type type, int WM, int WN, bool EXPERIMENTAL>
static void mmdq_rdna4_launch_k32(const char * W, const half * X, float * dst, const int M, const int N, const int K,
        const int64_t nbw, const int64_t ldy, int layout, int group, bool packed_w, cudaStream_t stream) {
    const dim3 grid((M + 64*WM - 1)/(64*WM), (N + 64*WN - 1)/(64*WN));
    const dim3 block(WARP_SIZE, WM*WN);
    static const bool reuse = [] { const char * e = getenv("GGML_HIP_MMDQ_REUSE"); return !e || std::atoi(e) != 0; }();
    if constexpr (type != GGML_TYPE_Q6_K) {
        if (reuse) {
            if constexpr (type == GGML_TYPE_Q4_K) {
                static const bool early_dequant = [] { const char * e = getenv("GGML_HIP_MMDQ_EARLY_DEQUANT"); return !e || std::atoi(e) != 0; }();
                if (early_dequant) {
                    if constexpr (EXPERIMENTAL) {
                        mmdq_rdna4_experimental<type, WM, WN, true, true><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w);
                    } else {
                        mmdq_rdna4<type, WM, WN, true, true><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy);
                    }
                    return;
                }
            }
            if constexpr (EXPERIMENTAL) {
                mmdq_rdna4_experimental<type, WM, WN, true, false><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w);
            } else {
                mmdq_rdna4<type, WM, WN, true, false><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy);
            }
            return;
        }
    }
    if constexpr (EXPERIMENTAL) {
        mmdq_rdna4_experimental<type, WM, WN, false, false><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w);
    } else {
        mmdq_rdna4<type, WM, WN, false, false><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy);
    }
}

template <ggml_type type, int WM, int WN>
static void mmdq_rdna4_launch(const char * W, const half * X, float * dst, const int M, const int N, const int K,
        const int64_t nbw, const int64_t ldy, int layout, bool packed_w, cudaStream_t stream) {
    const dim3 grid((M + 64*WM - 1)/(64*WM), (N + 64*WN - 1)/(64*WN));
    const dim3 block(WARP_SIZE, WM*WN);
    static const int group = [] { const char * e = getenv("GGML_HIP_MMDQ_GRID"); const int v = e ? std::atoi(e) : 0; return v == 4 || v == 8 ? v : 0; }();
    static const int pipeline = [] { const char * e = getenv("GGML_HIP_MMDQ_PIPELINE"); return e ? std::atoi(e) : 0; }();
    if constexpr (WM == 2) {
        if (pipeline == 1 || pipeline == 2) {
            if (pipeline == 2) {
                mmdq_rdna4_k64<type, WN, true><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w);
            } else {
                mmdq_rdna4_k64<type, WN, false><<<grid, block, 0, stream>>>(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w);
            }
            return;
        }
    }
    if (layout || group || packed_w) {
        mmdq_rdna4_launch_k32<type, WM, WN, true>(W, X, dst, M, N, K, nbw, ldy, layout, group, packed_w, stream);
    } else {
        mmdq_rdna4_launch_k32<type, WM, WN, false>(W, X, dst, M, N, K, nbw, ldy, 0, 0, false, stream);
    }
}

template <ggml_type type>
static void mmdq_rdna4_switch_cfg(const char * W, const half * X, float * dst, const int M, const int N, const int K,
        const int64_t nbw, const int64_t ldy, int layout, bool packed_w, cudaStream_t stream) {
    // Tile of 64 WM x 64 WN: 2 x 4 for q4_K, 2 x 8 (half the dequantization per FLOP) for q5_K/q6_K/iq4_xs.
    // GGML_HIP_MMDQ_CFG 1: 2 x 4, 2: 2 x 8, 3: 4 x 4 for every type.
    static const int cfg_env = [] { const char * e = getenv("GGML_HIP_MMDQ_CFG"); return e ? std::atoi(e) : 0; }();
    const int cfg = cfg_env ? cfg_env : (type == GGML_TYPE_Q4_K ? 1 : 2);
    switch (cfg) {
        case 2:  mmdq_rdna4_launch<type, 2, 8>(W, X, dst, M, N, K, nbw, ldy, layout, packed_w, stream); break;
        case 3:  mmdq_rdna4_launch<type, 4, 4>(W, X, dst, M, N, K, nbw, ldy, layout, packed_w, stream); break;
        default: mmdq_rdna4_launch<type, 2, 4>(W, X, dst, M, N, K, nbw, ldy, layout, packed_w, stream); break;
    }
}

void ggml_cuda_mmdq_rdna4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int M = src0->ne[1];
    const int K = src0->ne[0];
    const int N = src1->ne[1];
    const int layout = ggml_cuda_mmdq_layout(src1);

    ggml_cuda_pool_alloc<half> x16_tmp(ctx.pool());
    half * x16 = nullptr;
    bool reuse = false;
#if defined(GGML_USE_HIP)
    static const bool cache_enabled = [] {
        const char * value = getenv("GGML_HIP_MMDQ_F16_CACHE");
        return !value || std::atoi(value) != 0;
    }();
    if (cache_enabled && ctx.mmvq_cache.enabled && ctx.curr_stream_no == 0) {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        // Captured graphs must not retain a slot that can be reallocated by a later graph.
        if (status == hipStreamCaptureStatusNone) {
            x16 = (half *) ctx.mmdq_cache.acquire(src1, layout, ggml_nbytes(src1)/2, stream, reuse);
        }
    }
#endif
    if (!x16) {
        x16 = x16_tmp.alloc(int64_t(N)*K);
    }
    if (!reuse) {
        if (layout) {
            mmdq_convert_x<<<(int64_t(N)*K + 255)/256, 256, 0, stream>>>((const float *) src1->data, x16, K, N, layout);
        } else {
            ggml_cuda_convert_f32_f16_sat((const float *) src1->data, x16, int64_t(N)*K, stream);
        }
    }

    const char * W = (const char *) src0->data;
    static const bool repack = [] { const char * e = getenv("GGML_HIP_MMDQ_REPACK_Q5"); return e && std::atoi(e) != 0; }();
    ggml_cuda_pool_alloc<uint32_t> wpack(ctx.pool());
    const bool packed_w = repack && src0->type == GGML_TYPE_Q5_K;
    if (packed_w) {
        const size_t words = ggml_nbytes(src0)/sizeof(uint32_t);
        uint32_t * buffer = wpack.alloc(words);
        mmdq_pack_q5<<<(words + 255)/256, 256, 0, stream>>>((const uint32_t *) W, buffer, words, K/256);
        W = (const char *) buffer;
    }
    float * y = (float *) dst->data;
    const int64_t ldy = dst->nb[1]/sizeof(float);
    switch (src0->type) {
        case GGML_TYPE_Q4_K: mmdq_rdna4_switch_cfg<GGML_TYPE_Q4_K>(W, x16, y, M, N, K, src0->nb[1], ldy, layout, packed_w, stream); break;
        case GGML_TYPE_Q5_K: mmdq_rdna4_switch_cfg<GGML_TYPE_Q5_K>(W, x16, y, M, N, K, src0->nb[1], ldy, layout, packed_w, stream); break;
        case GGML_TYPE_Q6_K: mmdq_rdna4_switch_cfg<GGML_TYPE_Q6_K>(W, x16, y, M, N, K, src0->nb[1], ldy, layout, packed_w, stream); break;
        case GGML_TYPE_IQ4_XS: mmdq_rdna4_switch_cfg<GGML_TYPE_IQ4_XS>(W, x16, y, M, N, K, src0->nb[1], ldy, layout, packed_w, stream); break;
        default: GGML_ABORT("unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type GATE, ggml_type UP>
__launch_bounds__(512, 1)
static __global__ void mmdq_rdna4_gate_up(const char * wg, const char * wu, const half * x, half * dst,
        int M, int N, int K, int64_t nbg, int64_t nbu, int input_layout, int output_layout) {
#ifdef MMDQ_RDNA4_AVAILABLE
    constexpr int BM = 64, BN = 512, SA = 40, VPT = 8;
    const int lane = threadIdx.x, wave = threadIdx.y, wm = wave%2, wn = wave/2;
    const int l16 = lane%16, h = lane/16, tid = wave*32 + lane, ar = tid/4, ap = tid%4;
    const int m0 = blockIdx.x*BM, n0 = blockIdx.y*BN;
    const bool gate_row = ar < BM;
    const int row = min(m0 + ar%BM, M - 1);
    __shared__ __align__(16) half shared[2][2*BM*SA];
    mmdq_float8 ag[2][4] = {}, au[2][4] = {};
    mmdq_half8 b0[4][2], b1[4][2];
    mmdq_raw<GATE, VPT> rg;
    mmdq_raw<UP, VPT> ru;
    const auto load_weights = [&](int t) {
        if constexpr (GATE == UP) {
            mmdq_load_weight<GATE, VPT, GATE != GGML_TYPE_Q6_K>(rg, gate_row ? wg : wu, row, K,
                gate_row ? nbg : nbu, t, ap, false);
        } else if (gate_row) {
            mmdq_load_weight<GATE, VPT, GATE != GGML_TYPE_Q6_K>(rg, wg, row, K, nbg, t, ap, false);
        } else {
            mmdq_load_weight<UP, VPT, UP != GGML_TYPE_Q6_K>(ru, wu, row, K, nbu, t, ap, false);
        }
    };
    const auto store_weights = [&](int t) {
        half2 values[VPT/2];
        if constexpr (GATE == UP) {
            mmdq_dequant<GATE, VPT>(rg, t&7, values);
        } else if (gate_row) {
            mmdq_dequant<GATE, VPT>(rg, t&7, values);
        } else {
            mmdq_dequant<UP, VPT>(ru, t&7, values);
        }
        mmdq_store<VPT>(shared[t&1] + ar*SA + VPT*ap, values);
    };
    const auto load_b = [&](mmdq_half8 (&b)[4][2], int t) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = min(n0 + 64*wn + 16*j + l16, N - 1);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                b[j][s] = *(const mmdq_half8 *) (x + ggml_cuda_mmdq_x_index(n, t*32 + 16*s + 8*h, K, input_layout));
            }
        }
    };
    load_weights(0);
    store_weights(0);
    load_b(b0, 0);
    mmdq_lds_barrier();
    const auto step = [&](int t, const mmdq_half8 (&bc)[4][2], mmdq_half8 (&bn)[4][2]) {
        const bool more = t + 1 < K/32;
        if (more) {
            load_weights(t + 1);
            load_b(bn, t + 1);
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int offset = (32*wm + 16*i + l16)*SA + 16*s + 8*h;
                const mmdq_half8 g = *(const mmdq_half8 *) (shared[t&1] + offset);
                const mmdq_half8 u = *(const mmdq_half8 *) (shared[t&1] + BM*SA + offset);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    ag[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(g, bc[j][s], ag[i][j]);
                    au[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(u, bc[j][s], au[i][j]);
                }
            }
        }
        if (more) { store_weights(t + 1); }
        mmdq_lds_barrier();
    };
    for (int t = 0; t < K/32; t += 2) {
        step(t, b0, b1);
        step(t + 1, b1, b0);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int m = m0 + 32*wm + 16*i + 8*h;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + 64*wn + 16*j + l16;
            if (m < M && n < N) {
                mmdq_half8 out;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float value = ggml_cuda_op_silu_single(ag[i][j][r])*au[i][j][r];
                    out[r] = ggml_cuda_cast<half>(value > 65504.0f ? 65504.0f : (value < -65504.0f ? -65504.0f : value));
                }
                *(mmdq_half8 *) (dst + ggml_cuda_mmdq_x_index(n, m, M, output_layout)) = out;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wg, wu, x, dst, M, N, K, nbg, nbu, input_layout, output_layout);
    NO_DEVICE_CODE;
#endif
}

bool ggml_cuda_mmdq_gate_up_supported(const ggml_tensor * gate, const ggml_tensor * up) {
    return gate->op == GGML_OP_MUL_MAT && up->op == GGML_OP_MUL_MAT && gate->src[1] == up->src[1] &&
        ggml_are_same_shape(gate, up) &&
        ggml_cuda_mmdq_rdna4_supported(gate->src[0], gate->src[1], gate) &&
        ggml_cuda_mmdq_rdna4_supported(up->src[0], up->src[1], up);
}

template <ggml_type GATE>
static void mmdq_gate_up_dispatch(ggml_type up_type, const char * wg, const char * wu, const half * x, half * dst,
        int M, int N, int K, int64_t nbg, int64_t nbu, int input_layout, int output_layout, cudaStream_t stream) {
    const dim3 grid((M + 63)/64, (N + 511)/512), block(32, 16);
#define MMDQ_GATE_UP_CASE(TYPE) \
    case TYPE: mmdq_rdna4_gate_up<GATE, TYPE><<<grid, block, 0, stream>>>( \
        wg, wu, x, dst, M, N, K, nbg, nbu, input_layout, output_layout); break
    switch (up_type) {
        MMDQ_GATE_UP_CASE(GGML_TYPE_Q4_K);
        MMDQ_GATE_UP_CASE(GGML_TYPE_Q5_K);
        MMDQ_GATE_UP_CASE(GGML_TYPE_Q6_K);
        MMDQ_GATE_UP_CASE(GGML_TYPE_IQ4_XS);
        default: GGML_ABORT("unsupported paired MMDQ type");
    }
#undef MMDQ_GATE_UP_CASE
}

void ggml_cuda_mmdq_gate_up(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
        half * dst, int output_layout) {
    const ggml_tensor * src = gate->src[1];
    const int M = gate->ne[0], N = gate->ne[1], K = src->ne[0];
    const int layout = ggml_cuda_mmdq_layout(src);
    half * x = nullptr;
#if defined(GGML_USE_HIP)
    // The output slot is already reserved. Do not evict it while preparing the input.
    for (int slot = 0; slot < ggml_hip_mmq_cache::nslots; ++slot) {
        if (ctx.mmdq_cache.layout[slot] == layout &&
                ggml_hip_mmvq_cache::same_data(ctx.mmdq_cache.src[slot], src) &&
                ctx.mmdq_cache.size[slot] >= ggml_nbytes(src)/2) {
            x = (half *) ctx.mmdq_cache.data[slot];
            break;
        }
    }
#endif
    ggml_cuda_pool_alloc<half> temporary(ctx.pool());
    if (!x) {
        x = temporary.alloc(int64_t(N)*K);
        if (layout) {
            mmdq_convert_x<<<(int64_t(N)*K + 255)/256, 256, 0, ctx.stream()>>>((const float *) src->data, x, K, N, layout);
        } else {
            ggml_cuda_convert_f32_f16_sat((const float *) src->data, x, int64_t(N)*K, ctx.stream());
        }
    }
    const auto * wg = gate->src[0];
    const auto * wu = up->src[0];
#define MMDQ_GATE_CASE(TYPE) \
    case TYPE: mmdq_gate_up_dispatch<TYPE>(wu->type, (const char *) wg->data, (const char *) wu->data, x, dst, \
        M, N, K, wg->nb[1], wu->nb[1], layout, output_layout, ctx.stream()); break
    switch (wg->type) {
        MMDQ_GATE_CASE(GGML_TYPE_Q4_K);
        MMDQ_GATE_CASE(GGML_TYPE_Q5_K);
        MMDQ_GATE_CASE(GGML_TYPE_Q6_K);
        MMDQ_GATE_CASE(GGML_TYPE_IQ4_XS);
        default: GGML_ABORT("unsupported paired MMDQ type");
    }
#undef MMDQ_GATE_CASE
    CUDA_CHECK(cudaGetLastError());
}
