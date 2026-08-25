// delta_rule.cu -- CUDA compute kernels for DeltaNet delta_rule (per-token decode and prefill).
//
// Port of Metal delta_rule kernels for DeltaNet recurrent state update.
// Math and layout are identical to the Metal reference -- this is the portable equivalent
// fused GPU path for Windows/Linux (NVIDIA), eliminating GPU<->CPU sync on
// the recurrent step of DeltaNet.
//
// Qwen3.5-4B DeltaNet: 24 recurrent layers, each containing:
// 1. QMatMul projections (Candle CUDA, on GPU)
// 2. Prep: conv1d + sigmoid + softplus + L2 norm + head expansion
// 3. Delta rule: decay + sk + delta + rank-1 update + output
// 4. Norm + gate: Group RMS norm + SiLU(z) gating
//
// Dimensions (Qwen3.5-4B):
//   n_k_heads = 16, n_v_heads = 32
//   head_k_dim = 128, head_v_dim = 128
//   key_dim = 2048, value_dim = 4096
//   channels = 8192 (key_dim*2 + value_dim)
//   conv_kernel = 4
//
// Precision note: per-token decode kernels operate in F32. Long-prompt
// tests on Metal showed that F16 in decode accumulates cumulative drift, causing
// the model to emit EOS on the first decode token after a large prefill. F16 is used
// only in prefill/batch (a separate fused kernel).

#include "cuda_utils.cuh"
#include <stdint.h>

// ===============================================================
// Parameters (layout MUST match struct DeltaParams in Rust:
//   #[repr(C)] 11 fields of u32/f32, 44 bytes). Passed by value.
// ===============================================================

struct DeltaParams {
    unsigned int n_k_heads;   // 16
    unsigned int n_v_heads;   // 32
    unsigned int head_k_dim;  // 128
    unsigned int head_v_dim;  // 128
    unsigned int key_dim;     // n_k_heads * head_k_dim = 2048
    unsigned int value_dim;   // n_v_heads * head_v_dim = 4096
    unsigned int channels;    // key_dim * 2 + value_dim = 8192
    unsigned int conv_kernel; // 4
    float q_scale;            // 1 / sqrt(head_k_dim)
    float rms_norm_eps;       // 1e-6
    unsigned int heads_per_kv;// n_v_heads / n_k_heads = 2
};

// ===============================================================
// Helper functions (F32, single-precision math)
// ===============================================================

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float softplus_f(float x) {
    // log(1 + exp(x)), numerically stable version (matches Metal).
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + __expf(x));
}

// ===============================================================
// Kernel 1: delta_conv1d_prep
//
// Conv1d step + SiLU + sigmoid(beta) + softplus(alpha)*A
//
// Launch: grid=(ceil(channels/256),1,1), block=(256,1,1)
// Global tid; guard `tid < channels` (like Metal dispatch_threads).
//
// conv_state layout = [(conv_k-1), channels] row-major
// conv_weights layout = [channels, conv_k] row-major
// ===============================================================
extern "C" __global__ void delta_conv1d_prep(
    const float* __restrict__ qkv_raw,      // [channels] -- from QMatMul
    const float* __restrict__ beta_raw,     // [n_v_heads]
    const float* __restrict__ alpha_raw,    // [n_v_heads]
    const float* __restrict__ conv_weights, // [channels * conv_k]
    const float* __restrict__ dt_bias,      // [n_v_heads]
    const float* __restrict__ ssm_a,        // [n_v_heads]
    float* __restrict__ conv_state,         // [(conv_k-1) * channels] -- persistent
    float* __restrict__ qkv_conv_out,       // [channels] -- output
    float* __restrict__ beta_out,           // [n_v_heads]
    float* __restrict__ gate_out,           // [n_v_heads]
    const DeltaParams params
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int channels = params.channels;
    const unsigned int conv_k = params.conv_kernel;

    // -- Part A: Conv1d step (one thread per channel) --
    if (tid < channels) {
        float sum = 0.0f;
        unsigned int weight_base = tid * conv_k;

        // Convolution with previous inputs from buffer
        for (unsigned int i = 0; i < conv_k - 1; i++) {
            sum += conv_state[i * channels + tid] * conv_weights[weight_base + i];
        }
        // Current input
        sum += qkv_raw[tid] * conv_weights[weight_base + conv_k - 1];

        // SiLU activation
        qkv_conv_out[tid] = silu_f(sum);

        // Update conv_state: shift left + write new input
        if (conv_k > 2) {
            for (unsigned int i = 0; i < conv_k - 2; i++) {
                conv_state[i * channels + tid] = conv_state[(i + 1) * channels + tid];
            }
        }
        conv_state[(conv_k - 2) * channels + tid] = qkv_raw[tid];
    }

    // -- Part B: sigmoid(beta) and softplus(alpha)*A (first n_v_heads threads) --
    if (tid < params.n_v_heads) {
        beta_out[tid] = sigmoid_f(beta_raw[tid]);
        float alpha_biased = alpha_raw[tid] + dt_bias[tid];
        gate_out[tid] = softplus_f(alpha_biased) * ssm_a[tid];
    }
}

// ===============================================================
// Kernel 2: delta_l2_norm_expand
//
// L2 norm Q and K per k_head, expand heads 16->32, Q scaling, copy V.
//
// Launch: grid=(n_v_heads,1,1), block=(head_k_dim,1,1)=(128)
//   blockIdx.x = head_v (0..31), threadIdx.x = dim (0..127)
// Shared memory tree-reduction (sum of squares) within block.
// ===============================================================
extern "C" __global__ void delta_l2_norm_expand(
    const float* __restrict__ qkv_conv, // [channels] -- conv1d output
    float* __restrict__ q_out,          // [n_v_heads * head_k_dim]
    float* __restrict__ k_out,          // [n_v_heads * head_k_dim]
    float* __restrict__ v_out,          // [n_v_heads * head_v_dim]
    const DeltaParams params
) {
    const unsigned int head_v = blockIdx.x;   // 0..31
    const unsigned int dim = threadIdx.x;     // 0..127
    const unsigned int hkd = params.head_k_dim; // 128
    const unsigned int hvd = params.head_v_dim; // 128
    const unsigned int key_dim = params.key_dim;

    // GQA: v-head -> k-head via modulo (matches Metal/production)
    const unsigned int head_k = head_v % params.n_k_heads;

    __shared__ float sq_sum_q[128];
    __shared__ float sq_sum_k[128];

    // Q starts at 0, K -- after Q (key_dim) in qkv_conv
    unsigned int q_idx = head_k * hkd + dim;
    unsigned int k_idx = key_dim + head_k * hkd + dim;

    float q_val = qkv_conv[q_idx];
    float k_val = qkv_conv[k_idx];

    sq_sum_q[dim] = q_val * q_val;
    sq_sum_k[dim] = k_val * k_val;

    __syncthreads();

    // Tree reduction (128 -> 64 -> ... -> 1)
    for (unsigned int stride = hkd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_sum_q[dim] += sq_sum_q[dim + stride];
            sq_sum_k[dim] += sq_sum_k[dim + stride];
        }
        __syncthreads();
    }

    float q_inv_norm = rsqrtf(sq_sum_q[0] + 1e-6f);
    float k_inv_norm = rsqrtf(sq_sum_k[0] + 1e-6f);

    unsigned int out_idx = head_v * hkd + dim;
    q_out[out_idx] = q_val * q_inv_norm * params.q_scale;
    k_out[out_idx] = k_val * k_inv_norm;

    // Copy V (without normalization). V is after Q and K in qkv_conv.
    unsigned int v_src_idx = key_dim * 2 + head_v * hvd + dim;
    unsigned int v_dst_idx = head_v * hvd + dim;
    v_out[v_dst_idx] = qkv_conv[v_src_idx];
}

// ===============================================================
// Kernel 3: delta_rule_kernel
//
// Recurrent DeltaNet step. Each thread owns one column of state.
//
// Launch: grid=(n_v_heads,1,1), block=(head_v_dim,1,1)=(128)
//   blockIdx.x = head (0..31), threadIdx.x = col (0..127)
//
// state layout: [n_v_heads * hd * hd], per head row-major [hd x hd]
//   state[head][row][col] = ssm_state[head*hd*hd + row*hd + col]
//
// 1. Decay:  state[row][col] *= exp(gate[head])
// 2. sk[col] = sum_row(state[row][col] * k[row])     (S^T @ k)
// 3. d[col]  = (v[col] - sk[col]) * beta[head]
// 4. state[row][col] += k[row] * d[col]               (rank-1 update)
// 5. out[col] = sum_row(state[row][col] * q[row])     (S^T @ q)
// ===============================================================
extern "C" __global__ void delta_rule_kernel(
    const float* __restrict__ q,    // [n_v_heads * head_k_dim]
    const float* __restrict__ k,    // [n_v_heads * head_k_dim]
    const float* __restrict__ v,    // [n_v_heads * head_v_dim]
    const float* __restrict__ beta, // [n_v_heads]
    const float* __restrict__ gate, // [n_v_heads]
    float* __restrict__ ssm_state,  // [n_v_heads * hd * hd] -- persistent
    float* __restrict__ output,     // [n_v_heads * head_v_dim]
    const DeltaParams params
) {
    const unsigned int head = blockIdx.x;   // 0..31
    const unsigned int col = threadIdx.x;   // 0..127
    const unsigned int hd = params.head_v_dim; // 128

    const unsigned int state_base = head * hd * hd;
    const unsigned int vec_base = head * hd;

    __shared__ float shared_sk[128];
    __shared__ float shared_d[128];

    // -- 1. Decay: each thread multiplies its column (all rows) by exp(gate) --
    float gate_exp = __expf(gate[head]);
    for (unsigned int row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] *= gate_exp;
    }
    // No barrier needed: each thread reads/writes only its column.

    // -- 2. sk[col] = sum_row(state[row][col] * k[row]) --
    float sk_val = 0.0f;
    for (unsigned int row = 0; row < hd; row++) {
        sk_val += ssm_state[state_base + row * hd + col] * k[vec_base + row];
    }
    shared_sk[col] = sk_val;

    __syncthreads();

    // -- 3. d[col] = (v[col] - sk[col]) * beta[head] --
    float beta_h = beta[head];
    float d_val = (v[vec_base + col] - shared_sk[col]) * beta_h;
    shared_d[col] = d_val;

    __syncthreads();

    // -- 4. Rank-1 update: state[row][col] += k[row] * d[col] --
    float d_col = shared_d[col];
    for (unsigned int row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] += k[vec_base + row] * d_col;
    }

    // -- 5. out[col] = sum_row(state[row][col] * q[row]) --
    float out_val = 0.0f;
    for (unsigned int row = 0; row < hd; row++) {
        out_val += ssm_state[state_base + row * hd + col] * q[vec_base + row];
    }
    output[vec_base + col] = out_val;
}

// ===============================================================
// Kernel 4: delta_norm_gate_kernel
//
// Group RMS Norm per head + SiLU(z) gating.
//
// Launch: grid=(n_v_heads,1,1), block=(head_v_dim,1,1)=(128)
//   blockIdx.x = head (0..31), threadIdx.x = dim (0..127)
//
// 1. sq_mean = mean(out[head]^2) -- shared reduction
// 2. inv_rms = 1 / sqrt(sq_mean + eps)
// 3. gated[i] = out[i] * inv_rms * norm_weight[i % hvd] * silu(z[i])
// ===============================================================
extern "C" __global__ void delta_norm_gate_kernel(
    const float* __restrict__ raw_output,  // [n_v_heads * head_v_dim]
    const float* __restrict__ z,           // [value_dim]
    const float* __restrict__ norm_weight, // [head_v_dim] -- shared across heads
    float* __restrict__ gated_output,      // [value_dim]
    const DeltaParams params
) {
    const unsigned int head = blockIdx.x;  // 0..31
    const unsigned int dim = threadIdx.x;  // 0..127
    const unsigned int hvd = params.head_v_dim;
    const float eps = params.rms_norm_eps;

    const unsigned int idx = head * hvd + dim;

    __shared__ float sq_vals[128];

    float val = raw_output[idx];
    sq_vals[dim] = val * val;

    __syncthreads();

    for (unsigned int stride = hvd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_vals[dim] += sq_vals[dim + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sq_vals[0] / (float)hvd + eps);

    float normed = val * inv_rms * norm_weight[dim];
    float z_val = z[idx];
    gated_output[idx] = normed * silu_f(z_val);
}

// ===============================================================
// PREFILL (fused, single-slot): full sequence in 4 launches
// instead of 4 x T (token-by-token). Recurrence is maintained
// inside kernel 3 loop (state in global, hot in L2).
// ===============================================================

// P1: causal depthwise conv1d + SiLU across the full sequence + beta/gate prep.
// grid = (ceil(channels/256), T), block = (256).
// Tail before t < conv_k-1 is read from persistent conv_state.
extern "C" __global__ void delta_conv1d_prefill(
    const float* __restrict__ qkv_raw,      // [T * channels]
    const float* __restrict__ beta_raw,     // [T * n_v_heads]
    const float* __restrict__ alpha_raw,    // [T * n_v_heads]
    const float* __restrict__ conv_weights, // [channels * conv_k]
    const float* __restrict__ dt_bias,      // [n_v_heads]
    const float* __restrict__ ssm_a,        // [n_v_heads]
    const float* __restrict__ conv_state,   // [(conv_k-1) * channels] persistent
    float* __restrict__ qkv_conv_out,       // [T * channels]
    float* __restrict__ beta_out,           // [T * n_v_heads]
    float* __restrict__ gate_out,           // [T * n_v_heads]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int t = blockIdx.y;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int channels = params.channels;
    const unsigned int conv_k = params.conv_kernel;
    const unsigned int n_v = params.n_v_heads;
    if (t >= T) return;

    if (ch < channels) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < conv_k; j++) {
            const int src = (int)t - (int)(conv_k - 1) + (int)j;
            float x;
            if (src >= 0) {
                x = qkv_raw[(unsigned int)src * channels + ch];
            } else {
                // tail of persistent conv_state: indices [0, conv_k-1)
                x = conv_state[(unsigned int)((int)(conv_k - 1) + src) * channels + ch];
            }
            sum += x * conv_weights[ch * conv_k + j];
        }
        qkv_conv_out[t * channels + ch] = silu_f(sum);
    }
    if (ch < n_v) {
        const unsigned int idx = t * n_v + ch;
        beta_out[idx] = sigmoid_f(beta_raw[idx]);
        gate_out[idx] = softplus_f(alpha_raw[idx] + dt_bias[ch]) * ssm_a[ch];
    }
}

// P2: L2 norm Q/K + expand + scale across the full sequence.
// grid = (n_v_heads, T), block = (head_k_dim).
extern "C" __global__ void delta_l2_norm_prefill(
    const float* __restrict__ qkv_conv, // [T * channels]
    float* __restrict__ q_out,          // [T * n_v_heads * head_k_dim]
    float* __restrict__ k_out,          // [T * n_v_heads * head_k_dim]
    float* __restrict__ v_out,          // [T * n_v_heads * head_v_dim]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int head_v = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (t >= T) return;
    const unsigned int hkd = params.head_k_dim;
    const unsigned int hvd = params.head_v_dim;
    const unsigned int key_dim = params.key_dim;
    const unsigned int n_v = params.n_v_heads;
    const unsigned int channels = params.channels;
    const unsigned int head_k = head_v % params.n_k_heads;

    const unsigned int base = t * channels;
    __shared__ float sq_q[128];
    __shared__ float sq_k[128];

    const float q_val = qkv_conv[base + head_k * hkd + dim];
    const float k_val = qkv_conv[base + key_dim + head_k * hkd + dim];
    sq_q[dim] = q_val * q_val;
    sq_k[dim] = k_val * k_val;
    __syncthreads();
    for (unsigned int stride = hkd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_q[dim] += sq_q[dim + stride];
            sq_k[dim] += sq_k[dim + stride];
        }
        __syncthreads();
    }
    const float q_inv = rsqrtf(sq_q[0] + 1e-6f);
    const float k_inv = rsqrtf(sq_k[0] + 1e-6f);

    q_out[(t * n_v + head_v) * hkd + dim] = q_val * q_inv * params.q_scale;
    k_out[(t * n_v + head_v) * hkd + dim] = k_val * k_inv;
    v_out[(t * n_v + head_v) * hvd + dim] = qkv_conv[base + key_dim * 2 + head_v * hvd + dim];
}

// P3: recurrent delta rule across the full sequence -- loop inside the kernel.
// State in registers (like llama.cpp gated_delta_net.cu pattern): warp owns
// a state column, each lane holds 4 rows (hd=128/32) in registers for the whole
// loop -- zero global state traffic between tokens. Global state is read once
// at the start and written once at the end.
// grid = (n_v_heads, hd/4), block = (32, 4): col = blockIdx.y*4 + threadIdx.y.
extern "C" __global__ void delta_rule_prefill(
    const float* __restrict__ q,     // [T * n_v * hkd]
    const float* __restrict__ k,     // [T * n_v * hkd]
    const float* __restrict__ v,     // [T * n_v * hvd]
    const float* __restrict__ beta,  // [T * n_v]
    const float* __restrict__ gate,  // [T * n_v]
    float* __restrict__ ssm_state,   // [n_v * hd * hd] persistent
    float* __restrict__ output,      // [T * n_v * hvd]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int head = blockIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int hd = params.head_v_dim;   // 128
    const unsigned int n_v = params.n_v_heads;
    const unsigned int hkd = params.head_k_dim;
    constexpr unsigned int ROWS = 4;             // hd / warp_size = 128/32

    const unsigned int state_base = head * hd * hd;

    // Initial load of the state shard into registers (col-th column, 4 rows).
    float s[ROWS];
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) {
        const unsigned int row = r * 32 + lane;
        s[r] = ssm_state[state_base + row * hd + col];
    }

    for (unsigned int t = 0; t < T; t++) {
        const unsigned int kv_base = (t * n_v + head) * hkd;
        const unsigned int out_base = (t * n_v + head) * hd;
        const float g = __expf(gate[t * n_v + head]);
        const float beta_h = beta[t * n_v + head];

        float k_reg[ROWS], q_reg[ROWS];
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            const unsigned int row = r * 32 + lane;
            k_reg[r] = k[kv_base + row];
            q_reg[r] = q[kv_base + row];
        }

        // kv_col = (S^T k)[col] = sum_row S[row][col] * k[row]
        float kv_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) kv_part += s[r] * k_reg[r];
        float kv_col = kv_part;
        for (int o = 16; o > 0; o >>= 1) kv_col += __shfl_down_sync(0xffffffff, kv_col, o);
        // broadcast in warp
        kv_col = __shfl_sync(0xffffffff, kv_col, 0);

        // delta = (v[col] - g * kv_col) * beta
        const float v_col = v[out_base + col];
        const float delta_col = (v_col - g * kv_col) * beta_h;

        // S = g * S + k * delta^T; attn = (S^T q)[col]
        float attn_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            s[r] = g * s[r] + k_reg[r] * delta_col;
            attn_part += s[r] * q_reg[r];
        }
        float attn_col = attn_part;
        for (int o = 16; o > 0; o >>= 1) attn_col += __shfl_down_sync(0xffffffff, attn_col, o);
        if (lane == 0) output[out_base + col] = attn_col;
    }

    // Final write of state.
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) {
        const unsigned int row = r * 32 + lane;
        ssm_state[state_base + row * hd + col] = s[r];
    }
}

// P3-v2: тот же delta rule, но блок 1024 потоков ведёт 32 колонки состояния,
// а k/q токена кладутся в shared ОДИН раз на блок. В v1 каждая колонка читала
// k и q заново: 128 колонок × 512 токенов × 1 КБ = ~2 ГБ глобальных чтений на
// запуск, что при 360 ГБ/с и давало измеренные 4.12 мс — ядро упиралось в
// память, а не в последовательность. Здесь трафик в 32 раза меньше.
// Математика колонки не меняется (тот же порядок FMA и warp-редукций) →
// результат бит-в-бит совпадает с v1.
// grid = (n_v_heads, hd/32), block = (32, 32): col = blockIdx.y*32 + threadIdx.y.
extern "C" __global__ void delta_rule_prefill_v2(
    const float* __restrict__ q,     // [T * n_v * hkd]
    const float* __restrict__ k,     // [T * n_v * hkd]
    const float* __restrict__ v,     // [T * n_v * hvd]
    const float* __restrict__ beta,  // [T * n_v]
    const float* __restrict__ gate,  // [T * n_v]
    float* __restrict__ ssm_state,   // [n_v * hd * hd] persistent
    float* __restrict__ output,      // [T * n_v * hvd]
    const DeltaParams params,
    const unsigned int T
) {
    extern __shared__ float smem[];          // [hkd] k + [hkd] q
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int col = blockIdx.y * blockDim.y + warp;
    const unsigned int hd = params.head_v_dim;   // 128
    const unsigned int n_v = params.n_v_heads;
    const unsigned int hkd = params.head_k_dim;
    constexpr unsigned int ROWS = 4;             // hd / warp_size = 128/32
    const unsigned int tid = warp * blockDim.x + lane;
    const unsigned int nthreads = blockDim.x * blockDim.y;

    float* sk = smem;
    float* sq = smem + hkd;

    const unsigned int state_base = head * hd * hd;
    float s[ROWS];
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) {
        const unsigned int row = r * 32 + lane;
        s[r] = ssm_state[state_base + row * hd + col];
    }

    for (unsigned int t = 0; t < T; t++) {
        const unsigned int kv_base = (t * n_v + head) * hkd;
        const unsigned int out_base = (t * n_v + head) * hd;

        __syncthreads();                          // прошлый токен дочитан
        for (unsigned int i = tid; i < hkd; i += nthreads) {
            sk[i] = k[kv_base + i];
            sq[i] = q[kv_base + i];
        }
        __syncthreads();

        const float g = __expf(gate[t * n_v + head]);
        const float beta_h = beta[t * n_v + head];

        float k_reg[ROWS], q_reg[ROWS];
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            const unsigned int row = r * 32 + lane;
            k_reg[r] = sk[row];
            q_reg[r] = sq[row];
        }

        float kv_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) kv_part += s[r] * k_reg[r];
        float kv_col = kv_part;
        for (int o = 16; o > 0; o >>= 1) kv_col += __shfl_down_sync(0xffffffff, kv_col, o);
        kv_col = __shfl_sync(0xffffffff, kv_col, 0);

        const float v_col = v[out_base + col];
        const float delta_col = (v_col - g * kv_col) * beta_h;

        float attn_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            s[r] = g * s[r] + k_reg[r] * delta_col;
            attn_part += s[r] * q_reg[r];
        }
        float attn_col = attn_part;
        for (int o = 16; o > 0; o >>= 1) attn_col += __shfl_down_sync(0xffffffff, attn_col, o);
        if (lane == 0) output[out_base + col] = attn_col;
    }

    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) {
        const unsigned int row = r * 32 + lane;
        ssm_state[state_base + row * hd + col] = s[r];
    }
}

// P3-chunked: рекуррентность блоками по C токенов.
//
// Вывод (S — состояние [hkd × hvd], c_t — лог-кумулята гейта внутри блока):
//     S_t = exp(c_t) [ S_0 + Σ_{i≤t} (k_i/exp(c_i)) δ_iᵀ ]
//     δ_t = β_t ( v_t − S_0ᵀ k̂_t − Σ_{i<t} exp(c_t−c_i)(k_i·k_t) δ_i )
//     o_t = exp(c_t) S_0ᵀ q_t + Σ_{i≤t} exp(c_t−c_i)(k_i·q_t) δ_i
//     S_C = exp(c_C) S_0 + Σ_i exp(c_C−c_i) k_i δ_iᵀ
// Все множители входят как exp(c_t−c_i) при t ≥ i, то есть ≤ 1 — отдельно
// k/exp(c_i) не считаем, иначе при затухающем гейте f32 переполняется.
//
// Раскладка: блок ведёт COLS столбцов состояния одной головы. Поток владеет
// 32 строками своего столбца в регистрах, поэтому Sᵀk и Sᵀq считаются локально
// (нужна лишь 4-сторонняя редукция по группам строк), а warp-редукций на
// каждый токен, которые съедали 41% времени в последовательном ядре, нет.
// grid = (n_v_heads, hvd/COLS), block = (COLS, ROWGRP).
#define DR_CHUNK 32
#define DR_COLS 64
#define DR_ROWGRP 4

extern "C" __global__ void delta_rule_prefill_chunked(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ gate,
    float* __restrict__ ssm_state,
    float* __restrict__ output,
    const DeltaParams params,
    const unsigned int T
) {
    extern __shared__ float smem[];
    const unsigned int hkd = params.head_k_dim;   // 128
    const unsigned int hvd = params.head_v_dim;   // 128
    const unsigned int n_v = params.n_v_heads;
    const unsigned int head = blockIdx.x;
    const unsigned int col0 = blockIdx.y * DR_COLS;

    const unsigned int col_l = threadIdx.x;              // 0..COLS-1
    const unsigned int rowgrp = threadIdx.y;             // 0..ROWGRP-1
    const unsigned int tid = rowgrp * DR_COLS + col_l;
    const unsigned int nthreads = DR_COLS * DR_ROWGRP;
    const unsigned int rows_per = hkd / DR_ROWGRP;       // 32
    const unsigned int row0 = rowgrp * rows_per;
    const unsigned int col = col0 + col_l;

    float* sk = smem;                                    // [C][hkd]
    float* sq = sk + DR_CHUNK * hkd;                     // [C][hkd]
    float* sd = sq + DR_CHUNK * hkd;                     // [C][COLS]  (δ)
    float* sa = sd + DR_CHUNK * DR_COLS;                 // [C][C]     (k_i·k_t)
    float* sbq = sa + DR_CHUNK * DR_CHUNK;               // [C][C]     (k_i·q_t)
    float* sc = sbq + DR_CHUNK * DR_CHUNK;               // [C] лог-кумулята
    float* sbeta = sc + DR_CHUNK;                        // [C]
    float* sred = sbeta + DR_CHUNK;                      // [COLS][ROWGRP]

    // Состояние: строки [row0, row0+rows_per) своего столбца — в регистрах.
    float st[32];
    #pragma unroll
    for (unsigned int r = 0; r < 32; r++) {
        st[r] = ssm_state[head * hkd * hvd + (row0 + r) * hvd + col];
    }

    for (unsigned int t0 = 0; t0 < T; t0 += DR_CHUNK) {
        const unsigned int C = min((unsigned int)DR_CHUNK, T - t0);

        // 1. K/Q блока в shared + гейт/бета.
        for (unsigned int idx = tid; idx < C * hkd; idx += nthreads) {
            const unsigned int i = idx / hkd, d = idx % hkd;
            const unsigned int base = ((t0 + i) * n_v + head) * hkd;
            sk[i * hkd + d] = k[base + d];
            sq[i * hkd + d] = q[base + d];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (unsigned int i = 0; i < C; i++) {
                acc += gate[(t0 + i) * n_v + head];
                sc[i] = acc;
                sbeta[i] = beta[(t0 + i) * n_v + head];
            }
        }
        __syncthreads();

        // 2. Матрицы попарных скалярных произведений с затуханием.
        for (unsigned int idx = tid; idx < C * C; idx += nthreads) {
            const unsigned int t = idx / C, i = idx % C;
            if (i > t) {
                sa[idx] = 0.0f;
                sbq[idx] = 0.0f;
                continue;
            }
            float dk = 0.0f, dq = 0.0f;
            for (unsigned int d = 0; d < hkd; d++) {
                const float ki = sk[i * hkd + d];
                dk += ki * sk[t * hkd + d];
                dq += ki * sq[t * hkd + d];
            }
            const float decay = __expf(sc[t] - sc[i]);
            sa[idx] = (i < t) ? decay * dk : 0.0f;   // строго нижняя
            sbq[idx] = decay * dq;                   // включая диагональ
        }
        __syncthreads();

        // 3. Проекции состояния сразу для ВСЕХ токенов блока: pk[t][col] =
        //    Σ_row S[row][col] k_t[row], аналогично pq. Раньше это считалось
        //    внутри последовательного цикла по t с двумя блочными
        //    синхронизациями на токен — теперь один проход с atomicAdd по
        //    группам строк (4-сторонняя редукция без syncthreads на каждый t).
        for (unsigned int idx = tid; idx < C * DR_COLS; idx += nthreads) {
            spk[idx] = 0.0f;
            spq[idx] = 0.0f;
        }
        __syncthreads();
        for (unsigned int t = 0; t < C; t++) {
            float pk = 0.0f, pq = 0.0f;
            #pragma unroll
            for (unsigned int r = 0; r < 32; r++) {
                const float sv = st[r];
                pk += sv * sk[t * hkd + row0 + r];
                pq += sv * sq[t * hkd + row0 + r];
            }
            atomicAdd(&spk[t * DR_COLS + col_l], pk);
            atomicAdd(&spq[t * DR_COLS + col_l], pq);
        }
        __syncthreads();

        // 4. Прямая подстановка: единственная по-настоящему последовательная
        //    часть. Сумму по i делим между группами строк, чтобы работали все
        //    потоки, и оставляем одну синхронизацию на токен.
        for (unsigned int t = 0; t < C; t++) {
            float acc = 0.0f;
            for (unsigned int i = rowgrp; i < t; i += DR_ROWGRP) {
                acc += sa[t * DR_CHUNK + i] * sd[i * DR_COLS + col_l];
            }
            sred[col_l * DR_ROWGRP + rowgrp] = acc;
            __syncthreads();
            if (rowgrp == 0) {
                float sum = 0.0f;
                #pragma unroll
                for (unsigned int g = 0; g < DR_ROWGRP; g++) {
                    sum += sred[col_l * DR_ROWGRP + g];
                }
                const unsigned int vb = ((t0 + t) * n_v + head) * hvd;
                const float w = v[vb + col] - __expf(sc[t]) * spk[t * DR_COLS + col_l];
                sd[t * DR_COLS + col_l] = sbeta[t] * (w - sum);
            }
            __syncthreads();
        }

        // 5. Выход блока — теперь полностью параллельно по (t, col).
        for (unsigned int idx = tid; idx < C * DR_COLS; idx += nthreads) {
            const unsigned int t = idx / DR_COLS, cl = idx % DR_COLS;
            float o = __expf(sc[t]) * spq[idx];
            for (unsigned int i = 0; i <= t; i++) {
                o += sbq[t * DR_CHUNK + i] * sd[i * DR_COLS + cl];
            }
            output[((t0 + t) * n_v + head) * hvd + col0 + cl] = o;
        }
        __syncthreads();

        // 6. Состояние: S ← exp(c_C) S + Σ_i exp(c_C − c_i) k_i δ_iᵀ.
        const float gc = sc[C - 1];
        #pragma unroll
        for (unsigned int r = 0; r < 32; r++) {
            st[r] *= __expf(gc);
        }
        for (unsigned int i = 0; i < C; i++) {
            const float w = __expf(gc - sc[i]) * sd[i * DR_COLS + col_l];
            #pragma unroll
            for (unsigned int r = 0; r < 32; r++) {
                st[r] += w * sk[i * hkd + row0 + r];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int r = 0; r < 32; r++) {
        ssm_state[head * hkd * hvd + (row0 + r) * hvd + col] = st[r];
    }
}

// ДИАГНОСТИКА (не для продакшена): те же обращения к памяти и та же
// арифметика, но с одной warp-редукцией на токен (probe1) и без редукций
// вовсе (probe0). Результат заведомо неверен — ядра нужны, чтобы измерить,
// какую долю времени занимают сами редукции.
extern "C" __global__ void delta_rule_prefill_probe1(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ beta,
    const float* __restrict__ gate, float* __restrict__ ssm_state,
    float* __restrict__ output, const DeltaParams params, const unsigned int T)
{
    const unsigned int head = blockIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int hd = params.head_v_dim;
    const unsigned int n_v = params.n_v_heads;
    const unsigned int hkd = params.head_k_dim;
    constexpr unsigned int ROWS = 4;
    const unsigned int state_base = head * hd * hd;
    float s[ROWS];
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) s[r] = ssm_state[state_base + (r * 32 + lane) * hd + col];
    for (unsigned int t = 0; t < T; t++) {
        const unsigned int kv_base = (t * n_v + head) * hkd;
        const unsigned int out_base = (t * n_v + head) * hd;
        const float g = __expf(gate[t * n_v + head]);
        const float beta_h = beta[t * n_v + head];
        float k_reg[ROWS], q_reg[ROWS];
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            k_reg[r] = k[kv_base + r * 32 + lane];
            q_reg[r] = q[kv_base + r * 32 + lane];
        }
        float kv_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) kv_part += s[r] * k_reg[r];
        float kv_col = kv_part;
        for (int o = 16; o > 0; o >>= 1) kv_col += __shfl_down_sync(0xffffffff, kv_col, o);
        kv_col = __shfl_sync(0xffffffff, kv_col, 0);
        const float delta_col = (v[out_base + col] - g * kv_col) * beta_h;
        float attn_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            s[r] = g * s[r] + k_reg[r] * delta_col;
            attn_part += s[r] * q_reg[r];
        }
        if (lane == 0) output[out_base + col] = attn_part;   // редукция пропущена
    }
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) ssm_state[state_base + (r * 32 + lane) * hd + col] = s[r];
}

extern "C" __global__ void delta_rule_prefill_probe0(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ beta,
    const float* __restrict__ gate, float* __restrict__ ssm_state,
    float* __restrict__ output, const DeltaParams params, const unsigned int T)
{
    const unsigned int head = blockIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int hd = params.head_v_dim;
    const unsigned int n_v = params.n_v_heads;
    const unsigned int hkd = params.head_k_dim;
    constexpr unsigned int ROWS = 4;
    const unsigned int state_base = head * hd * hd;
    float s[ROWS];
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) s[r] = ssm_state[state_base + (r * 32 + lane) * hd + col];
    for (unsigned int t = 0; t < T; t++) {
        const unsigned int kv_base = (t * n_v + head) * hkd;
        const unsigned int out_base = (t * n_v + head) * hd;
        const float g = __expf(gate[t * n_v + head]);
        const float beta_h = beta[t * n_v + head];
        float k_reg[ROWS], q_reg[ROWS];
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            k_reg[r] = k[kv_base + r * 32 + lane];
            q_reg[r] = q[kv_base + r * 32 + lane];
        }
        float kv_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) kv_part += s[r] * k_reg[r];
        const float delta_col = (v[out_base + col] - g * kv_part) * beta_h;
        float attn_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            s[r] = g * s[r] + k_reg[r] * delta_col;
            attn_part += s[r] * q_reg[r];
        }
        if (lane == 0) output[out_base + col] = attn_part;
    }
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) ssm_state[state_base + (r * 32 + lane) * hd + col] = s[r];
}

// P4: group RMS norm + SiLU(z) gate across the full sequence.
// grid = (n_v_heads, T), block = (head_v_dim).
extern "C" __global__ void delta_norm_gate_prefill(
    const float* __restrict__ raw_output,  // [T * n_v * hvd]
    const float* __restrict__ z,           // [T * value_dim]
    const float* __restrict__ norm_weight, // [hvd]
    float* __restrict__ gated_output,      // [T * value_dim]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int head = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (t >= T) return;
    const unsigned int hvd = params.head_v_dim;
    const unsigned int n_v = params.n_v_heads;
    const float eps = params.rms_norm_eps;

    const unsigned int idx = (t * n_v + head) * hvd + dim;
    const unsigned int z_idx = t * params.value_dim + head * hvd + dim;

    __shared__ float sq_vals[128];
    const float val = raw_output[idx];
    sq_vals[dim] = val * val;
    __syncthreads();
    for (unsigned int stride = hvd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) sq_vals[dim] += sq_vals[dim + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(sq_vals[0] / (float)hvd + eps);
    gated_output[z_idx] = val * inv_rms * norm_weight[dim] * silu_f(z[z_idx]);
}
