// Recurrent Gated DeltaNet from ExLlamaV3 (MIT), including the fixed-order
// cross-slice reduction. The installed source this was taken from hashes to
// 969eaec94a66feac8f5bb992666e02615dbd7efd381136bcddee16d3a1848541.
#include <cuda_bf16.h>
#include <algorithm>

using bfloat16 = __nv_bfloat16;
#define SUBK 4

template <int MAX_HEAD_DIM, bool save_history, int V_SPLIT, bool MAMBA2 = false>
__global__ __launch_bounds__(MAX_HEAD_DIM * SUBK)
void cuda_recurrent_gated_delta_rule_kernel
(
                                                // k_dim = num_k_heads * k_head_dim
                                                // v_dim = num_v_heads * v_head_dim
    const bfloat16* __restrict__ mixed_qkv,     // [bsz, seqlen, (k_dim + k_dim + v_dim)]
    const float* __restrict__ g,                // [bsz, seqlen, (group * num_k_heads)]
    const bfloat16* __restrict__ beta,          // [bsz, seqlen, (group * num_k_heads)]
    float* __restrict__ recurrent_state,        // [num_slots, max_history + 1, (group * num_k_heads), k_head_dim, v_head_dim]
    bfloat16* __restrict__ core_attn_out,       // [bsz, seqlen, num_v_heads, v_head_dim]
    const int bsz,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const int k_head_dim,
    const int v_head_dim,
    const float scale,
    const int* __restrict__ slots,              // [bsz]
    const int history_stride,                   // max_history + 1
    const float* __restrict__ D                 // [num_v_heads], MAMBA2 only, else nullptr
)
{
    int group = num_v_heads / num_k_heads;
    const size_t state_size = group * num_k_heads * k_head_dim * v_head_dim;
    const size_t slot_size = (size_t) history_stride * state_size;

    // Advance to batch item
    int bi = blockIdx.x;
    mixed_qkv +=        bi * seqlen * (2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads);
    g +=                bi * seqlen * (group * num_k_heads);
    beta +=             bi * seqlen * (group * num_k_heads);
    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;
    float* final_state = slot_state;
    core_attn_out +=    bi * seqlen * num_v_heads * v_head_dim;

    // Indexing
    int t = threadIdx.x;
    int bt = threadIdx.y;
    int bts = k_head_dim / SUBK;
    int lane = t % 32;
    int warp = t / 32;
    int head = blockIdx.y;
    int k_head = head / group;
    int v_chunk = blockIdx.z;
    int v_chunk_dim = v_head_dim / V_SPLIT;
    int v_start = v_chunk * v_chunk_dim;

    // Shared buffers
    __shared__ float sh_red[2][MAX_HEAD_DIM / 32];
    __shared__ float sh_k[MAX_HEAD_DIM];
    __shared__ float sh_q[MAX_HEAD_DIM];
    __shared__ float sh_dot1[MAX_HEAD_DIM];
    __shared__ float sh_dot2[MAX_HEAD_DIM];
    // Deterministic cross-slice partials: one slot per (bt, t), reduced below in a
    // fixed j order. The original shared-memory atomicAdd made the accumulation
    // order (and thus the low mantissa bits) depend on warp scheduling.
    __shared__ float sh_p1[SUBK][MAX_HEAD_DIM];
    __shared__ float sh_p2[SUBK][MAX_HEAD_DIM];

    // Iterate over sequence dim
    for (int s = 0; s < seqlen; ++s)
    {
        // Advance to q/k head
        const bfloat16* gl_q;
        const bfloat16* gl_k;
        const bfloat16* gl_v;
        if constexpr (MAMBA2)
        {
            gl_v = mixed_qkv + head * v_head_dim + v_start;
            gl_k = mixed_qkv + num_v_heads * v_head_dim + k_head * k_head_dim;
            gl_q = mixed_qkv + num_v_heads * v_head_dim + num_k_heads * k_head_dim + k_head * k_head_dim;
        }
        else
        {
            gl_q = mixed_qkv + k_head * k_head_dim;
            gl_k = mixed_qkv + (num_k_heads + k_head) * k_head_dim;
            gl_v = mixed_qkv + (2 * num_k_heads * k_head_dim) + head * v_head_dim + v_start;
        }
        bfloat16* out = core_attn_out + head * v_head_dim + v_start;

        float* gl_rs_r;
        float* gl_rs_w;
        if constexpr (save_history)
        {
            bool first = (s == 0);
            bool last = (s == seqlen - 1);
            float* history_r = first ? nullptr : slot_state + (size_t) s * state_size;
            float* history_w = last  ? final_state : slot_state + (size_t) (s + 1) * state_size;
            gl_rs_r = first ? final_state + head * (k_head_dim * v_head_dim)
                            : history_r   + head * (k_head_dim * v_head_dim);
            gl_rs_w = history_w           + head * (k_head_dim * v_head_dim);
        }
        else
        {
            gl_rs_r = final_state + head * (k_head_dim * v_head_dim);
            gl_rs_w = gl_rs_r;
        }

        // Read q/k heads and apply L2 norm
        float q, k;
        if (t < k_head_dim && bt == 0)
        {
            q = __bfloat162float(gl_q[t]);
            k = __bfloat162float(gl_k[t]);

            if constexpr (MAMBA2)
            {
                sh_k[t] = k;
                sh_q[t] = q;
            }
            else
            {
                float sumq = q * q;
                float sumk = k * k;
                #pragma unroll
                for(int offset = 16; offset > 0; offset /= 2)
                {
                    sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
                    sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
                }
                if (lane == 0)
                {
                    sh_red[0][warp] = sumq;
                    sh_red[1][warp] = sumk;
                }
            }
        }

        if constexpr (!MAMBA2)
        {
            __syncthreads();

            if (t < k_head_dim && bt == 0)
            {
                float sumq = lane < k_head_dim / 32 ? sh_red[0][lane] : 0.0f;
                float sumk = lane < k_head_dim / 32 ? sh_red[1][lane] : 0.0f;
                #pragma unroll
                for(int offset = 16; offset > 0; offset /= 2)
                {
                    sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
                    sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
                }

                q = q * rsqrtf(sumq + 1e-6f);
                k = k * rsqrtf(sumk + 1e-6f);

                // Write q, k to shmem
                sh_k[t] = k;
                sh_q[t] = q;
            }
        }

        if (t < v_chunk_dim && bt == 0)
        {
            if constexpr (!MAMBA2)
                sh_dot1[t] = 0.0f;
            sh_dot2[t] = 0.0f;
        }
        __syncthreads();

        if constexpr (!MAMBA2)
        {
            if (t < v_chunk_dim)
            {
                // Dot products with last state
                float sum = 0.0f;
                float* sh_k_rd = sh_k + bt * bts;
                float* rs_rd = gl_rs_r + v_start + t + bt * bts * v_head_dim;

                for (int i = 0; i < k_head_dim / 8 / SUBK; ++i)
                {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j, rs_rd += v_head_dim, sh_k_rd++)
                        sum = sum + *sh_k_rd * *rs_rd;
                }
                sh_p1[bt][t] = sum;
            }
            __syncthreads();

            // Fixed-order cross-slice reduction (was shared-memory atomicAdd)
            if (t < v_chunk_dim && bt == 0)
            {
                float acc = sh_p1[0][t];
                #pragma unroll
                for (int j = 1; j < SUBK; ++j)
                    acc += sh_p1[j][t];
                sh_dot1[t] = acc;
            }
            __syncthreads();
        }

        if (t < v_chunk_dim)
        {
            float g_h = __expf(g[head]);
            float beta_h = __bfloat162float(beta[head]);

            // Read v head; delta rule subtracts the decayed state readback, Mamba2 injects raw v
            float v = __bfloat162float(gl_v[t]);
            if constexpr (!MAMBA2)
                v -= sh_dot1[t] * g_h;

            // Update step
            float v_out = 0.0f;
            float* sh_k_rd = sh_k + bt * bts;
            float* sh_q_rd = sh_q + bt * bts;
            float* rs_r = gl_rs_r + v_start + t + bt * bts * v_head_dim;
            float* rs_w = gl_rs_w + v_start + t + bt * bts * v_head_dim;

            for (int i = 0; i < k_head_dim / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_r += v_head_dim, rs_w += v_head_dim, sh_k_rd++, sh_q_rd++)
                {
                    // State update step, k x v
                    float state = *rs_r;
                    state = state * g_h + *sh_k_rd * v * beta_h;
                    *rs_w = state;

                    // Accumulate attn output
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            sh_p2[bt][t] = v_out;
        }
        __syncthreads();

        // Fixed-order cross-slice reduction (was shared-memory atomicAdd)
        if (t < v_chunk_dim && bt == 0)
        {
            float acc = sh_p2[0][t];
            #pragma unroll
            for (int j = 1; j < SUBK; ++j)
                acc += sh_p2[j][t];
            sh_dot2[t] = acc;
        }
        __syncthreads();

        if (t < v_chunk_dim && bt == 0)
        {
            float v_out = sh_dot2[t];

            // Store attn output
            if constexpr (MAMBA2)
                out[t] = __float2bfloat16_rz(v_out + D[head] * __bfloat162float(gl_v[t]));
            else
                out[t] = __float2bfloat16_rz(v_out * scale);
        }

        // Next seq index
        mixed_qkv +=        2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads;
        g +=                num_v_heads;
        beta +=             num_v_heads;
        core_attn_out +=    num_v_heads * v_head_dim;
    }
}

template <bool save_history, int V_SPLIT>
__global__ __launch_bounds__(128 * SUBK)
void cuda_recurrent_gated_delta_rule_kernel_128
(
                                                // k_head_dim = v_head_dim = 128
    const bfloat16* __restrict__ mixed_qkv,     // [bsz, seqlen, (k_dim + k_dim + v_dim)]
    const float* __restrict__ g,                // [bsz, seqlen, (group * num_k_heads)]
    const bfloat16* __restrict__ beta,          // [bsz, seqlen, (group * num_k_heads)]
    float* __restrict__ recurrent_state,        // [num_slots, max_history + 1, (group * num_k_heads), 128, 128]
    bfloat16* __restrict__ core_attn_out,       // [bsz, seqlen, num_v_heads, 128]
    const int bsz,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const int k_head_dim,
    const int v_head_dim,
    const float scale,
    const int* __restrict__ slots,              // [bsz]
    const int history_stride,                   // max_history + 1
    const float* __restrict__ D                 // unused, matches the generic kernel signature
)
{
    constexpr int HEAD_DIM = 128;
    constexpr int V_CHUNK_DIM = HEAD_DIM / V_SPLIT;
    constexpr int BTS = HEAD_DIM / SUBK;

    int group = num_v_heads / num_k_heads;
    constexpr size_t HEAD_STATE_SIZE = HEAD_DIM * HEAD_DIM;
    const size_t state_size = group * num_k_heads * HEAD_STATE_SIZE;
    const size_t slot_size = (size_t) history_stride * state_size;

    int bi = blockIdx.x;
    mixed_qkv +=        bi * seqlen * (3 * HEAD_DIM * num_k_heads + HEAD_DIM * (num_v_heads - num_k_heads));
    g +=                bi * seqlen * (group * num_k_heads);
    beta +=             bi * seqlen * (group * num_k_heads);
    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;
    float* final_state = slot_state;
    core_attn_out +=    bi * seqlen * num_v_heads * HEAD_DIM;

    int t = threadIdx.x;
    int bt = threadIdx.y;
    int lane = t % 32;
    int warp = t / 32;
    int head = blockIdx.y;
    int k_head = head / group;
    int v_chunk = blockIdx.z;
    int v_start = v_chunk * V_CHUNK_DIM;

    __shared__ float sh_red[2][HEAD_DIM / 32];
    __shared__ float sh_k[HEAD_DIM];
    __shared__ float sh_q[HEAD_DIM];
    __shared__ float sh_dot1[HEAD_DIM];
    __shared__ float sh_dot2[HEAD_DIM];
    // Deterministic cross-slice partials (see the generic kernel above)
    __shared__ float sh_p1[SUBK][V_CHUNK_DIM];
    __shared__ float sh_p2[SUBK][V_CHUNK_DIM];

    for (int s = 0; s < seqlen; ++s)
    {
        const bfloat16* gl_q = mixed_qkv + k_head * HEAD_DIM;
        const bfloat16* gl_k = mixed_qkv + (num_k_heads + k_head) * HEAD_DIM;
        const bfloat16* gl_v = mixed_qkv + (2 * num_k_heads * HEAD_DIM) + head * HEAD_DIM + v_start;
        bfloat16* out = core_attn_out + head * HEAD_DIM + v_start;

        float* gl_rs_r;
        float* gl_rs_w;
        if constexpr (save_history)
        {
            bool first = (s == 0);
            bool last = (s == seqlen - 1);
            float* history_r = first ? nullptr : slot_state + (size_t) s * state_size;
            float* history_w = last  ? final_state : slot_state + (size_t) (s + 1) * state_size;
            gl_rs_r = first ? final_state + head * HEAD_STATE_SIZE
                            : history_r   + head * HEAD_STATE_SIZE;
            gl_rs_w = history_w           + head * HEAD_STATE_SIZE;
        }
        else
        {
            gl_rs_r = final_state + head * HEAD_STATE_SIZE;
            gl_rs_w = gl_rs_r;
        }

        float q = __bfloat162float(gl_q[t]);
        float k = __bfloat162float(gl_k[t]);

        float sumq = q * q;
        float sumk = k * k;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }
        if (lane == 0)
        {
            sh_red[0][warp] = sumq;
            sh_red[1][warp] = sumk;
        }
        __syncthreads();

        sumq = lane < HEAD_DIM / 32 ? sh_red[0][lane] : 0.0f;
        sumk = lane < HEAD_DIM / 32 ? sh_red[1][lane] : 0.0f;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }

        q = q * rsqrtf(sumq + 1e-6f);
        k = k * rsqrtf(sumk + 1e-6f);
        sh_k[t] = k;
        sh_q[t] = q;

        if (t < V_CHUNK_DIM && bt == 0)
        {
            sh_dot1[t] = 0.0f;
            sh_dot2[t] = 0.0f;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float sum = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* rs_rd = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_rd += HEAD_DIM, sh_k_rd++)
                    sum = sum + *sh_k_rd * *rs_rd;
            }
            sh_p1[bt][t] = sum;
        }
        __syncthreads();

        // Fixed-order cross-slice reduction (was shared-memory atomicAdd)
        if (t < V_CHUNK_DIM && bt == 0)
        {
            float acc = sh_p1[0][t];
            #pragma unroll
            for (int j = 1; j < SUBK; ++j)
                acc += sh_p1[j][t];
            sh_dot1[t] = acc;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float g_h = __expf(g[head]);
            float beta_h = __bfloat162float(beta[head]);
            float v = __bfloat162float(gl_v[t]) - sh_dot1[t] * g_h;
            float v_out = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* sh_q_rd = sh_q + bt * BTS;
            float* rs_r = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;
            float* rs_w = gl_rs_w + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_r += HEAD_DIM, rs_w += HEAD_DIM, sh_k_rd++, sh_q_rd++)
                {
                    float state = *rs_r;
                    state = state * g_h + *sh_k_rd * v * beta_h;
                    *rs_w = state;
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            sh_p2[bt][t] = v_out;
        }
        __syncthreads();

        // Fixed-order cross-slice reduction (was shared-memory atomicAdd)
        if (t < V_CHUNK_DIM && bt == 0)
        {
            float acc = sh_p2[0][t];
            #pragma unroll
            for (int j = 1; j < SUBK; ++j)
                acc += sh_p2[j][t];
            sh_dot2[t] = acc;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM && bt == 0)
            out[t] = __float2bfloat16_rz(sh_dot2[t] * scale);

        mixed_qkv +=        2 * HEAD_DIM * num_k_heads + HEAD_DIM * num_v_heads;
        g +=                num_v_heads;
        beta +=             num_v_heads;
        core_attn_out +=    num_v_heads * HEAD_DIM;
    }
}

extern "C" void q38_gdn_recurrent(
    const void* mixed_qkv,
    const float* g,
    const void* beta,
    float* recurrent_state,
    void* core_attn_out,
    int bsz,
    int seqlen,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const int* slots,
    int history_stride,
    int history)
{
    int v_split = (bsz == 1 && k_head_dim <= 128 && v_head_dim == 128 && num_v_heads <= 64) ? 4 : 1;
    dim3 blocks(bsz, num_v_heads, v_split);
    int thread_x = std::max(k_head_dim, v_head_dim / v_split);
    dim3 threads(thread_x, SUBK);
    float scale = 1.0f / sqrtf(static_cast<float>(k_head_dim));
    auto qkv = reinterpret_cast<const bfloat16*>(mixed_qkv);
    auto beta_bf = reinterpret_cast<const bfloat16*>(beta);
    auto out = reinterpret_cast<bfloat16*>(core_attn_out);

    auto launch = [&](auto kernel) {
        kernel<<<blocks, threads>>>(
            qkv, g, beta_bf, recurrent_state, out, bsz, seqlen, num_k_heads, num_v_heads,
            k_head_dim, v_head_dim, scale, slots, history_stride, nullptr);
    };

    if (!history) {
        if (k_head_dim == 128 && v_head_dim == 128) {
            if (v_split == 4) launch(cuda_recurrent_gated_delta_rule_kernel_128<false, 4>);
            else              launch(cuda_recurrent_gated_delta_rule_kernel_128<false, 1>);
        } else if (thread_x <= 128) {
            if (v_split == 4) launch(cuda_recurrent_gated_delta_rule_kernel<128, false, 4>);
            else              launch(cuda_recurrent_gated_delta_rule_kernel<128, false, 1>);
        } else if (thread_x <= 256) {
            launch(cuda_recurrent_gated_delta_rule_kernel<256, false, 1>);
        }
    } else {
        if (k_head_dim == 128 && v_head_dim == 128) {
            if (v_split == 4) launch(cuda_recurrent_gated_delta_rule_kernel_128<true, 4>);
            else              launch(cuda_recurrent_gated_delta_rule_kernel_128<true, 1>);
        } else if (thread_x <= 128) {
            if (v_split == 4) launch(cuda_recurrent_gated_delta_rule_kernel<128, true, 4>);
            else              launch(cuda_recurrent_gated_delta_rule_kernel<128, true, 1>);
        } else if (thread_x <= 256) {
            launch(cuda_recurrent_gated_delta_rule_kernel<256, true, 1>);
        }
    }
}

__device__ __forceinline__ float _sigmoid_fast_exp(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ bfloat16 trunc_bf16(float x)
{
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float as_float(bfloat16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float as_float(float x)
{
    return x;
}

__device__ __forceinline__ float softplus(float x)
{
    if (x > 20.0f) return x;
    return log1pf(__expf(x));
}

#define FUSED_OP_2_THREADS 512

template <typename a_log_T>
__global__ void gated_delta_net_fused_op_2_kernel
(
    const float* __restrict__ in_b,
    const float* __restrict__ in_a,
    const bfloat16* __restrict__ in_dt_bias,
    const a_log_T* __restrict__ in_a_log,
    bfloat16* __restrict__ out_beta,
    float* __restrict__ out_g,
    int B,
    int S,
    int H,
    int rows_per_block,
    const float beta_scale
)
{
    int t = threadIdx.x % H;
    int row = blockIdx.x * rows_per_block + threadIdx.x / H;
    if (row >= B * S) return;

    in_b += row * H + t;
    in_a += row * H + t;
    in_dt_bias += t;
    in_a_log += t;
    out_beta += row * H + t;
    out_g += row * H + t;

    float beta = _sigmoid_fast_exp(*in_b) * beta_scale;
    float dt_bias = as_float(*in_dt_bias);
    float g = -softplus(*in_a + dt_bias) * __expf(as_float(*in_a_log));

    *out_beta = trunc_bf16(beta);
    *out_g = g;
}

extern "C" void q38_gdn_fused_op_2(
    const float* b,
    const float* a,
    const void* dt_bias,
    const float* a_log,
    void* beta,
    float* g,
    int bsz,
    int seqlen,
    int heads,
    float beta_scale)
{
    int rows_per_block = FUSED_OP_2_THREADS / heads;
    int threads = rows_per_block * heads;
    int blocks = (bsz * seqlen + rows_per_block - 1) / rows_per_block;
    gated_delta_net_fused_op_2_kernel<float><<<blocks, threads>>>(
        b, a, reinterpret_cast<const bfloat16*>(dt_bias), a_log,
        reinterpret_cast<bfloat16*>(beta), g,
        bsz, seqlen, heads, rows_per_block, beta_scale);
}

#define CONV1D_MAX_K 16
#define CONV1D_NUM_THREADS 256

template <bool ACT, bool HISTORY>
__global__ __launch_bounds__(CONV1D_NUM_THREADS)
void conv1d_update_kernel
(
    const bfloat16* __restrict__ x,
    bfloat16* __restrict__ conv_state,
    const int* __restrict__ slots,
    const bfloat16* __restrict__ weight,
    const bfloat16* __restrict__ bias,
    bfloat16* __restrict__ out,
    const int dim,
    const int seqlen,
    const int state_size,
    const int K
)
{
    int d = blockIdx.x * CONV1D_NUM_THREADS + threadIdx.x;
    if (d >= dim) return;
    int b = blockIdx.y;
    int slot = slots ? slots[b] : b;

    const bfloat16* x_d = x + ((size_t) b * dim + d) * seqlen;
    bfloat16* state_d = conv_state + ((size_t) slot * dim + d) * state_size;

    float w[CONV1D_MAX_K];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < K) w[k] = __bfloat162float(weight[(size_t) d * K + k]);

    float bias_d = bias ? __bfloat162float(bias[d]) : 0.0f;

    float old_state[CONV1D_MAX_K];
    float win[CONV1D_MAX_K];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < K) old_state[k] = __bfloat162float(state_d[k]);
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K - 1; ++k)
        if (k < K - 1) win[k] = old_state[k + 1];

    for (int s = 0; s < seqlen; ++s)
    {
        win[K - 1] = __bfloat162float(x_d[s]);

        float acc = bias_d;
        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K; ++k)
            if (k < K) acc = fmaf(w[k], win[k], acc);

        if constexpr (ACT)
            acc *= _sigmoid_fast_exp(acc);

        out[((size_t) b * seqlen + s) * dim + d] = __float2bfloat16_rn(acc);

        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K - 1; ++k)
            if (k < K - 1) win[k] = win[k + 1];
    }

    if constexpr (!HISTORY)
    {
        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K; ++k)
        {
            if (k < K)
            {
                int src_t = seqlen + k;
                float v = (src_t < K) ? old_state[src_t] : __bfloat162float(x_d[src_t - K]);
                state_d[k] = __float2bfloat16_rn(v);
            }
        }
    }
    else
    {
        int total = K + seqlen;
        int write_size = state_size < total ? state_size : total;
        int dst_start = state_size - write_size;
        int src_start = total - write_size;
        for (int j = 0; j < write_size; ++j)
        {
            int src_t = src_start + j;
            float v = (src_t < K) ? old_state[src_t] : __bfloat162float(x_d[src_t - K]);
            state_d[dst_start + j] = __float2bfloat16_rn(v);
        }
    }
}

extern "C" void q38_gdn_conv1d(
    const void* x,
    void* conv_state,
    const void* weight,
    const void* bias,
    void* out,
    int bsz,
    int dim,
    int seqlen,
    int state_size,
    int kernel,
    int activation)
{
    dim3 blocks((dim + CONV1D_NUM_THREADS - 1) / CONV1D_NUM_THREADS, bsz);
    auto launch = [&](auto kernel_fn) {
        kernel_fn<<<blocks, CONV1D_NUM_THREADS>>>(
            reinterpret_cast<const bfloat16*>(x),
            reinterpret_cast<bfloat16*>(conv_state),
            nullptr,
            reinterpret_cast<const bfloat16*>(weight),
            reinterpret_cast<const bfloat16*>(bias),
            reinterpret_cast<bfloat16*>(out),
            dim, seqlen, state_size, kernel);
    };
    if (activation) launch(conv1d_update_kernel<true, false>);
    else launch(conv1d_update_kernel<false, false>);
}
