// delta_rule.metal — Metal compute kernels для DeltaNet delta_rule
//
// Qwen3.5-4B DeltaNet: 24 рекуррентных слоя, каждый содержит:
// 1. QMatMul проекции (Candle Metal, уже на GPU)
// 2. Prep: conv1d + sigmoid + softplus + L2 norm + head expansion
// 3. Delta rule: decay + sk + delta + rank-1 update + output
// 4. Norm + gate: Group RMS norm + SiLU(z) gating
//
// Все 4 этапа выполняются на GPU без GPU↔CPU синхронизации.
//
// Размерности (Qwen3.5-4B):
//   n_k_heads = 16, n_v_heads = 32
//   head_k_dim = 128, head_v_dim = 128
//   key_dim = 2048, value_dim = 4096
//   channels = 8192 (key_dim*2 + value_dim)
//   conv_kernel = 4

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════
// Параметры, передаваемые из Rust через constant buffer
// ═══════════════════════════════════════════════════════════════

struct DeltaParams {
    uint n_k_heads;      // 16
    uint n_v_heads;       // 32
    uint head_k_dim;      // 128
    uint head_v_dim;      // 128
    uint key_dim;         // n_k_heads * head_k_dim = 2048
    uint value_dim;       // n_v_heads * head_v_dim = 4096
    uint channels;        // key_dim * 2 + value_dim = 8192
    uint conv_kernel;     // 4
    float q_scale;        // 1 / sqrt(head_k_dim) = 1/11.3137...
    float rms_norm_eps;   // 1e-6
    uint heads_per_kv;    // n_v_heads / n_k_heads = 2 (для GQA head expansion)
};

// ═══════════════════════════════════════════════════════════════
// Вспомогательные функции
// ═══════════════════════════════════════════════════════════════

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float softplus(float x) {
    // log(1 + exp(x)), численно стабильная версия
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log(1.0f + exp(x));
}

// ═══════════════════════════════════════════════════════════════
// Kernel 1: delta_conv1d_prep
//
// Conv1d step с SiLU + sigmoid(beta) + softplus(alpha)*A
//
// Grid: (channels, 1, 1) = (8192, 1, 1)
// Threadgroup: (256, 1, 1)
//
// ВАЖНО: conv_state layout = [conv_k-1, channels] row-major
//        conv_weights layout = [channels, conv_k] row-major
// ═══════════════════════════════════════════════════════════════

// F32 baseline для per-token decode (precision critical для рекуррентного state).
// Long-prompt тестирование показало: F16 в decode kernels даёт cumulative drift,
// модель сразу EOS на первом decode token после большого prefill. F16 оставлен
// только в GDN fused kernels (prefill, batch buffers — main memory savings).
kernel void delta_conv1d_prep(
    device const float* qkv_raw        [[buffer(0)]],   // [channels] — from QMatMul
    device const float* beta_raw       [[buffer(1)]],   // [n_v_heads]
    device const float* alpha_raw      [[buffer(2)]],   // [n_v_heads]
    device const float* conv_weights   [[buffer(3)]],   // [channels * conv_k]
    device const float* dt_bias        [[buffer(4)]],   // [n_v_heads]
    device const float* ssm_a          [[buffer(5)]],   // [n_v_heads]
    device float*       conv_state     [[buffer(6)]],   // [(conv_k-1) * channels] — persistent
    device float*       qkv_conv_out   [[buffer(7)]],   // [channels] — output
    device float*       beta_out       [[buffer(8)]],   // [n_v_heads]
    device float*       gate_out       [[buffer(9)]],   // [n_v_heads]
    constant DeltaParams& params       [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    uint channels = params.channels;
    uint conv_k = params.conv_kernel;

    // ── Часть A: Conv1d step (8192 потоков, по одному на канал) ──
    if (tid < channels) {
        float sum = 0.0f;
        uint weight_base = tid * conv_k;

        // Свёртка с предыдущими входами из буфера
        for (uint i = 0; i < conv_k - 1; i++) {
            sum += conv_state[i * channels + tid] * conv_weights[weight_base + i];
        }
        // Текущий вход
        sum += qkv_raw[tid] * conv_weights[weight_base + conv_k - 1];

        // SiLU активация
        qkv_conv_out[tid] = silu(sum);

        // Обновление conv_state: сдвиг влево + запись нового входа
        if (conv_k > 2) {
            for (uint i = 0; i < conv_k - 2; i++) {
                conv_state[i * channels + tid] = conv_state[(i + 1) * channels + tid];
            }
        }
        conv_state[(conv_k - 2) * channels + tid] = qkv_raw[tid];
    }

    // ── Часть B: sigmoid(beta) и softplus(alpha)*A (32 потока, первые n_v_heads) ──
    if (tid < params.n_v_heads) {
        beta_out[tid] = sigmoid(beta_raw[tid]);
        float alpha_biased = alpha_raw[tid] + dt_bias[tid];
        gate_out[tid] = softplus(alpha_biased) * ssm_a[tid];
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 2: delta_l2_norm_expand
//
// L2 нормализация Q и K per k_head, расширение голов 16→32,
// Q scaling, копирование V.
//
// Grid: (n_v_heads, head_k_dim, 1) = (32, 128, 1)
// Threadgroup: (1, 128, 1) — одна голова = один threadgroup
//
// Используем threadgroup shared memory для reduction (sum of squares).
// ═══════════════════════════════════════════════════════════════

kernel void delta_l2_norm_expand(
    device const float* qkv_conv       [[buffer(0)]],   // [channels] — conv1d output
    device float*       q_out          [[buffer(1)]],   // [n_v_heads * head_k_dim]
    device float*       k_out          [[buffer(2)]],   // [n_v_heads * head_k_dim]
    device float*       v_out          [[buffer(3)]],   // [n_v_heads * head_v_dim]
    constant DeltaParams& params       [[buffer(4)]],
    uint2 gid  [[thread_position_in_grid]],             // (head_v_idx, dim_idx)
    uint2 lid  [[thread_position_in_threadgroup]],      // (0, local_dim)
    uint2 tgid [[threadgroup_position_in_grid]]         // (head_v_idx, 0)
) {
    uint head_v = gid.x;          // 0..31 (v-head index)
    uint dim = gid.y;             // 0..127
    uint hkd = params.head_k_dim; // 128
    uint hvd = params.head_v_dim; // 128
    uint key_dim = params.key_dim;
    uint heads_per_kv = params.heads_per_kv;

    // Маппинг v-head → k-head (GQA: 32 v-heads → 16 k-heads)
    // Используем modulo (совпадает с production кодом): v-heads 0-15 → k-heads 0-15, v-heads 16-31 → k-heads 0-15
    uint head_k = head_v % params.n_k_heads;

    // ── Shared memory для L2 norm reduction ──
    threadgroup float sq_sum_q[128];
    threadgroup float sq_sum_k[128];

    // Читаем Q и K raw значения для этого k-head
    uint q_idx = head_k * hkd + dim;
    uint k_idx = key_dim + head_k * hkd + dim;  // K начинается после Q в qkv_conv

    float q_val = qkv_conv[q_idx];
    float k_val = qkv_conv[k_idx];

    // Загружаем квадраты для reduction
    sq_sum_q[dim] = q_val * q_val;
    sq_sum_k[dim] = k_val * k_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Parallel reduction для суммы квадратов ──
    // Стандартная tree reduction (128 → 64 → 32 → ... → 1)
    for (uint stride = hkd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_sum_q[dim] += sq_sum_q[dim + stride];
            sq_sum_k[dim] += sq_sum_k[dim + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // sq_sum_q[0] и sq_sum_k[0] теперь содержат суммы квадратов
    float q_inv_norm = rsqrt(sq_sum_q[0] + 1e-6f);
    float k_inv_norm = rsqrt(sq_sum_k[0] + 1e-6f);

    // ── Нормализация + Q scaling + запись ──
    uint out_idx = head_v * hkd + dim;
    q_out[out_idx] = q_val * q_inv_norm * params.q_scale;
    k_out[out_idx] = k_val * k_inv_norm;

    // ── Копирование V (прямое, без нормализации) ──
    // V начинается после Q и K в qkv_conv
    uint v_src_idx = key_dim * 2 + head_v * hvd + dim;
    uint v_dst_idx = head_v * hvd + dim;
    v_out[v_dst_idx] = qkv_conv[v_src_idx];
}

// ═══════════════════════════════════════════════════════════════
// Kernel 3: delta_rule_kernel
//
// Основной рекуррентный шаг DeltaNet.
//
// Grid: (n_v_heads, head_v_dim, 1) = (32, 128, 1)
// Threadgroup: (1, 128, 1) — одна голова = один threadgroup
//
// Каждый поток обрабатывает один столбец state матрицы [hd × hd].
// state layout: [n_v_heads * hd * hd], per head = [hd (rows) × hd (cols)]
// Row-major: state[head][row][col] = ssm_state[head * hd * hd + row * hd + col]
//
// Алгоритм:
// 1. Decay: state[row][col] *= exp(gate[head]) для всех row
// 2. sk[col] = sum_row(state[row][col] * k[row])  (S^T @ k)
// 3. d[col] = (v[col] - sk[col]) * beta[head]
// 4. state[row][col] += k[row] * d[col]            (rank-1 update)
// 5. output[col] = sum_row(state[row][col] * q[row]) (S^T @ q)
// ═══════════════════════════════════════════════════════════════

kernel void delta_rule_kernel(
    device const float* q              [[buffer(0)]],   // [n_v_heads * head_k_dim]
    device const float* k              [[buffer(1)]],   // [n_v_heads * head_k_dim]
    device const float* v              [[buffer(2)]],   // [n_v_heads * head_v_dim]
    device const float* beta           [[buffer(3)]],   // [n_v_heads]
    device const float* gate           [[buffer(4)]],   // [n_v_heads]
    device float*       ssm_state      [[buffer(5)]],   // [n_v_heads * hd * hd] — persistent
    device float*       output         [[buffer(6)]],   // [n_v_heads * head_v_dim]
    constant DeltaParams& params       [[buffer(7)]],
    uint2 gid  [[thread_position_in_grid]],             // (head_idx, dim_idx)
    uint2 lid  [[thread_position_in_threadgroup]]       // (0, local_dim)
) {
    uint head = gid.x;      // 0..31
    uint col = gid.y;       // 0..127
    uint hd = params.head_v_dim;  // 128

    // Базовые смещения
    uint state_base = head * hd * hd;  // начало state этой головы
    uint vec_base = head * hd;         // начало q/k/v этой головы

    // ── Shared memory для sk и d (нужны всем потокам в threadgroup) ──
    threadgroup float shared_sk[128];
    threadgroup float shared_d[128];

    // ── 1. Decay: state[row][col] *= exp(gate[head]) ──
    float gate_exp = exp(gate[head]);
    for (uint row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] *= gate_exp;
    }

    // Барьер не нужен — каждый поток пишет в свой столбец

    // ── 2. sk[col] = sum_row(state[row][col] * k[head*hd + row]) ──
    //    Это S^T @ k: каждый поток вычисляет один элемент результата
    float sk_val = 0.0f;
    for (uint row = 0; row < hd; row++) {
        sk_val += ssm_state[state_base + row * hd + col] * k[vec_base + row];
    }
    shared_sk[col] = sk_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. d[col] = (v[col] - sk[col]) * beta[head] ──
    float beta_h = beta[head];
    float d_val = (v[vec_base + col] - shared_sk[col]) * beta_h;
    shared_d[col] = d_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 4. Rank-1 update: state[row][col] += k[row] * d[col] ──
    //    Каждый поток обновляет свой столбец
    float d_col = shared_d[col];
    for (uint row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] += k[vec_base + row] * d_col;
    }

    // ── 5. output[col] = sum_row(state[row][col] * q[row]) ──
    //    S^T @ q: каждый поток вычисляет один элемент выхода
    float out_val = 0.0f;
    for (uint row = 0; row < hd; row++) {
        out_val += ssm_state[state_base + row * hd + col] * q[vec_base + row];
    }
    output[vec_base + col] = out_val;
}

// ═══════════════════════════════════════════════════════════════
// Kernel 4: delta_norm_gate_kernel
//
// Group RMS Norm per head + SiLU(z) gating.
//
// Grid: (n_v_heads, head_v_dim, 1) = (32, 128, 1)
// Threadgroup: (1, 128, 1) — одна голова = один threadgroup
//
// Алгоритм:
// 1. sq_mean = mean(output[head]^2) — threadgroup reduction
// 2. inv_rms = 1 / sqrt(sq_mean + eps)
// 3. gated[i] = output[i] * inv_rms * norm_weight[i % hvd] * silu(z[head*hvd + i])
// ═══════════════════════════════════════════════════════════════

kernel void delta_norm_gate_kernel(
    device const float* raw_output     [[buffer(0)]],   // [n_v_heads * head_v_dim]
    device const float* z              [[buffer(1)]],   // [value_dim]
    device const float* norm_weight    [[buffer(2)]],   // [head_v_dim] — shared across heads
    device float*       gated_output   [[buffer(3)]],   // [value_dim]
    constant DeltaParams& params       [[buffer(4)]],
    uint2 gid  [[thread_position_in_grid]],             // (head_idx, dim_idx)
    uint2 lid  [[thread_position_in_threadgroup]]       // (0, local_dim)
) {
    uint head = gid.x;       // 0..31
    uint dim = gid.y;        // 0..127
    uint hvd = params.head_v_dim;
    float eps = params.rms_norm_eps;

    uint idx = head * hvd + dim;

    // ── Shared memory для reduction ──
    threadgroup float sq_vals[128];

    float val = raw_output[idx];
    sq_vals[dim] = val * val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Tree reduction для суммы квадратов ──
    for (uint stride = hvd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_vals[dim] += sq_vals[dim + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // sq_vals[0] = сумма квадратов по голове
    float inv_rms = rsqrt(sq_vals[0] / float(hvd) + eps);

    // ── Norm * weight * SiLU(z) ──
    float normed = val * inv_rms * norm_weight[dim];
    float z_val = z[idx];
    gated_output[idx] = normed * silu(z_val);
}
