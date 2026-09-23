#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

#if defined(GGML_USE_HIP)
struct conv_qk_args { float *q,*k;int heads;float eps,scale,bias; };
// Several tokens (speculative verification): same arithmetic as ssm_conv + silu for each token.
// The saved states are windows starting after `keep` tokens (speculative decoding keeps one per accepted length).
template<bool indexed,int n_tokens>
static __global__ void conv_prepare_silu_batch_f32(const float * old_state,const float * input,
        const float * weight,float * output,const int32_t * index,int64_t stride,int channels,ggml_cuda_conv_states states) {
    if constexpr(indexed) { old_state+=int64_t(*index)*stride; }
    const int c=blockIdx.x*128+threadIdx.x;
    float x[3+n_tokens];
    x[0]=old_state[3*c];x[1]=old_state[3*c+1];x[2]=old_state[3*c+2];
#pragma unroll
    for(int t=0;t<n_tokens;++t) { x[3+t]=input[int64_t(t)*channels+c]; }
    const float w0=weight[4*c],w1=weight[4*c+1],w2=weight[4*c+2],w3=weight[4*c+3];
#pragma unroll
    for(int t=0;t<n_tokens;++t) {
        float sum=0.0f;
        sum=fmaf(x[t],w0,sum);
        sum=fmaf(x[t+1],w1,sum);
        sum=fmaf(x[t+2],w2,sum);
        sum=fmaf(x[t+3],w3,sum);
        sum+=0.0f;
        output[int64_t(t)*channels+c]=ggml_cuda_op_silu_single(sum);
    }
    for(int i=0;i<states.n;++i) {
        const int keep=states.keep[i];
        float * next_state=states.dst[i];
#pragma unroll
        for(int k=0;k<=n_tokens;++k) {
            if(k==keep) { next_state[3*c]=x[k];next_state[3*c+1]=x[k+1];next_state[3*c+2]=x[k+2]; }
        }
    }
}

template<bool indexed,bool normalize>
static __global__ void conv_prepare_silu_f32(const float * old_state,const float * input,
        const float * weight,float * next_state,float * output,const int32_t * index,int64_t stride,conv_qk_args qk) {
    if constexpr(indexed) { old_state+=int64_t(*index)*stride; }
    const int tid=threadIdx.x,c=blockIdx.x*128+tid;
    float value=0.0f;
    if(tid<128) {
    const float x0=old_state[3*c],x1=old_state[3*c+1],x2=old_state[3*c+2],x3=input[c];
    float sum=0.0f;
    sum=fmaf(x0,weight[4*c],sum);
    sum=fmaf(x1,weight[4*c+1],sum);
    sum=fmaf(x2,weight[4*c+2],sum);
    sum=fmaf(x3,weight[4*c+3],sum);
    sum+=0.0f;
    next_state[3*c]=x1;next_state[3*c+1]=x2;next_state[3*c+2]=x3;
    value=ggml_cuda_op_silu_single(sum);
    output[c]=value;
    }
    if constexpr(normalize) {
        if(blockIdx.x<2*qk.heads) {
            float squared=0.0f;
            if(tid<128) { squared+=value*value; }
            extern __shared__ float partial[];
            squared=block_reduce<block_reduce_method::SUM,256>(squared,partial);
            const float scale=rsqrtf(squared/128+qk.eps);
            if(tid<128) {
                float *dst=blockIdx.x<qk.heads ? qk.q : qk.k;
                const int row=blockIdx.x<qk.heads ? blockIdx.x : blockIdx.x-qk.heads;
                const float normalized=scale*value;
                dst[row*128+tid]=qk.scale*normalized+qk.bias;
            }
        }
    }
}

void ggml_cuda_op_conv_prepare(ggml_backend_cuda_context & ctx,const ggml_tensor * concat,
        const ggml_tensor * weight,const ggml_tensor * state,ggml_tensor * output,const ggml_tensor * gathered,
        const ggml_tensor *norm,ggml_tensor *q,ggml_tensor *k,const ggml_cuda_conv_states * states) {
    const int n_tokens=int(concat->ne[0])-3;
    if(n_tokens>1) {
        GGML_ASSERT(!norm && states && states->n>=1 && states->n<=8);
        const ggml_cuda_kernel_launch_params params(dim3(concat->ne[1]/128),dim3(128),0,ctx.stream());
        const auto launch=[&](auto flag,auto tokens) {
            ggml_cuda_kernel_launch(conv_prepare_silu_batch_f32<decltype(flag)::value,decltype(tokens)::value>,params,
                static_cast<const float *>(gathered ? gathered->src[0]->data : concat->src[0]->data),
                static_cast<const float *>(concat->src[1]->data),static_cast<const float *>(weight->data),
                static_cast<float *>(output->data),
                gathered ? static_cast<const int32_t *>(gathered->src[1]->data) : nullptr,
                gathered ? int64_t(gathered->src[0]->nb[1]/sizeof(float)) : 0,int(concat->ne[1]),*states);
        };
        const auto dispatch=[&](auto flag) {
            switch(n_tokens) {
                case 2: launch(flag,std::integral_constant<int,2>{}); break;
                case 3: launch(flag,std::integral_constant<int,3>{}); break;
                case 4: launch(flag,std::integral_constant<int,4>{}); break;
                default: GGML_ABORT("unsupported conv batch");
            }
        };
        if(gathered) { dispatch(std::true_type{}); } else { dispatch(std::false_type{}); }
        return;
    }
    conv_qk_args qk{};
    if(norm) { qk={static_cast<float *>(q->data),static_cast<float *>(k->data),int(norm->ne[1]),
        ggml_get_op_params_f32(norm,0),ggml_get_op_params_f32(q,0),ggml_get_op_params_f32(q,1)}; }
    const ggml_cuda_kernel_launch_params params(dim3(concat->ne[1]/128),dim3(norm ? 256 : 128),norm ? 32*sizeof(float) : 0,ctx.stream());
    const auto launch=[&](auto flag,auto normalize) {
        ggml_cuda_kernel_launch(conv_prepare_silu_f32<decltype(flag)::value,decltype(normalize)::value>,params,
            static_cast<const float *>(gathered ? gathered->src[0]->data : concat->src[0]->data),
            static_cast<const float *>(concat->src[1]->data),static_cast<const float *>(weight->data),
            static_cast<float *>(state->data),static_cast<float *>(output->data),
            gathered ? static_cast<const int32_t *>(gathered->src[1]->data) : nullptr,
            gathered ? int64_t(gathered->src[0]->nb[1]/sizeof(float)) : 0,qk);
    };
    if(norm) {
        if(gathered) { launch(std::true_type{},std::true_type{}); } else { launch(std::false_type{},std::true_type{}); }
    } else {
        if(gathered) { launch(std::true_type{},std::false_type{}); } else { launch(std::false_type{},std::false_type{}); }
    }
}
#endif

#if defined(GGML_USE_HIP)
static __global__ void conv_prefill_halo(const float * old, const float * input, float * halo, int h) {
    const int c=blockIdx.x*128+threadIdx.x;
    const int begin=blockIdx.y*32;
    if(c>=h) { return; }
#pragma unroll
    for(int j=0;j<3;++j) {
        halo[(size_t(blockIdx.y)*3+j)*h+c]=begin==0 ? old[3*c+j] : input[size_t(begin+j-3)*h+c];
    }
}
// shifted: the output starts 3 tokens before the input, so the next chunk overwrites the last 3 inputs of this chunk;
// they are read from the next chunk's halo instead.
template <bool halo_input = false, bool shifted = false>
static __global__ void conv_prefill_direct(const float * old,const float * input,const float * weights,
        float * state,float * output,int h,int n) {
    const int c=blockIdx.x*128+threadIdx.x;
    const int begin=blockIdx.y*32;
    if (c>=h) { return; }
    float w[4],x[4];
#pragma unroll
    for(int j=0;j<4;++j) { w[j]=weights[4*c+j]; }
#pragma unroll
    for(int j=0;j<3;++j) {
        x[j]=halo_input ? old[(size_t(blockIdx.y)*3+j)*h+c] : (begin+j<3 ? old[3*c+begin+j] : input[(size_t)(begin+j-3)*h+c]);
    }
    const int end=min(begin+32,n);
    for(int t=begin;t<end;++t) {
        if(shifted && end<n && t>=end-3) {
            x[3]=old[(size_t(blockIdx.y+1)*3+(t-(end-3)))*h+c];
        } else {
            x[3]=input[(size_t)t*h+c];
        }
        float sum=0.0f;
#pragma unroll
        for(int j=0;j<4;++j) { sum+=x[j]*w[j]; }
        sum+=0.0f;
        output[(size_t)t*h+c]=ggml_cuda_op_silu_single(sum);
#pragma unroll
        for(int j=0;j<3;++j) { x[j]=x[j+1]; }
    }
    if(end==n) {
#pragma unroll
        for(int j=0;j<3;++j) { state[3*c+j]=x[j]; }
    }
}
void ggml_cuda_conv_prefill(ggml_backend_cuda_context & ctx,const ggml_tensor * cat,
        const ggml_tensor * weight,const ggml_tensor * state,ggml_tensor * output,bool scratch) {
    const int h=cat->ne[1],n=cat->src[1]->ne[0];
    static const bool halo_enabled=[] { const char * v=getenv("GGML_HIP_PREFILL_CONV_HALO"); return !v || atoi(v)!=0; }();
    const auto disjoint=[](const ggml_tensor * a,const ggml_tensor * b) {
        const uintptr_t x=(uintptr_t)a->data,y=(uintptr_t)b->data;
        return x+ggml_nbytes(a)<=y || y+ggml_nbytes(b)<=x;
    };
    if(halo_enabled && scratch && output->data==cat->src[1]->data &&
            disjoint(output,cat->src[0]) && disjoint(output,weight)) {
        const dim3 grid((h+127)/128,(n+31)/32);
        conv_prefill_halo<<<grid,128,0,ctx.stream()>>>((const float *)cat->src[0]->data,
            (const float *)cat->src[1]->data,(float *)cat->data,h);
        CUDA_CHECK(cudaGetLastError());
        conv_prefill_direct<true><<<grid,128,0,ctx.stream()>>>((const float *)cat->data,
            (const float *)cat->src[1]->data,(const float *)weight->data,(float *)state->data,(float *)output->data,h,n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Output starts at the old state, 3 tokens before the input: each channel writes token t over input t-3, which the
    // same thread has already read. The 3 tokens before each 32 token chunk are saved first: the chunk reads them as its
    // start and the previous chunk as its last inputs, because this chunk overwrites them.
    static const bool shifted_enabled=[] { const char * v=getenv("GGML_HIP_PREFILL_CONV_SHIFTED"); return !v || atoi(v)!=0; }();
    const char * old_d=(const char *)cat->src[0]->data;
    if(shifted_enabled && scratch && output->data==old_d && (const char *)cat->src[1]->data==old_d+ggml_nbytes(cat->src[0]) &&
            ggml_nbytes(output)==ggml_nbytes(cat->src[1]) && disjoint(cat,output) && disjoint(cat,cat->src[1]) &&
            disjoint(cat,weight) && disjoint(output,weight) && size_t((n+31)/32)*3*h*sizeof(float)<=ggml_nbytes(cat)) {
        const dim3 grid((h+127)/128,(n+31)/32);
        conv_prefill_halo<<<grid,128,0,ctx.stream()>>>((const float *)old_d,(const float *)cat->src[1]->data,(float *)cat->data,h);
        CUDA_CHECK(cudaGetLastError());
        conv_prefill_direct<true,true><<<grid,128,0,ctx.stream()>>>((const float *)cat->data,
            (const float *)cat->src[1]->data,(const float *)weight->data,(float *)state->data,(float *)output->data,h,n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    conv_prefill_direct<false><<<dim3((h+127)/128,(n+31)/32),128,0,ctx.stream()>>>(
        (const float *)cat->src[0]->data,(const float *)cat->src[1]->data,(const float *)weight->data,
        (float *)state->data,(float *)(scratch ? cat->data : output->data),h,n);
    CUDA_CHECK(cudaGetLastError());
    if(scratch) { CUDA_CHECK(cudaMemcpyAsync(output->data,cat->data,ggml_nbytes(output),cudaMemcpyDeviceToDevice,ctx.stream())); }
}
#endif
