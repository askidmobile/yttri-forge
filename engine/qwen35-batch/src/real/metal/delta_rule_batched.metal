// delta_rule_batched.metal — Metal compute kernels для batched DeltaNet decode
//
// Phase 1 true batched decode: 4 kernel с осью slot B. Каждый (head, slot) независим
// (decode не имеет sequential token loop, в отличие от fused prefill). Math per
// (head, slot) идентична per-head math из delta_rule.metal — добавлена только
// slot-индексация в state/temp буферах и grid.
//
// Размерности (Qwen3.5-4B):
//   n_k_heads = 16, n_v_heads = 32
//   head_k_dim = 128, head_v_dim = 128
//   key_dim = 2048, value_dim = 4096
//   channels = 8192 (key_dim*2 + value_dim)
//   conv_kernel = 4
//   B = batch_size (slot count), 1..MAX_BATCH
//
// Layout буферов (leading B):
//   ssm_state   [B * n_v_heads * hd * hd]
//   conv_state  [B * (conv_k-1) * channels]
//   qkv_conv    [B * channels]
//   q           [B * n_v_heads * head_k_dim]
//   k           [B * n_v_heads * head_k_dim]
//   v           [B * n_v_heads * head_v_dim]
//   beta        [B * n_v_heads]
//   gate        [B * n_v_heads]
//   delta_out   [B * n_v_heads * head_v_dim]
//   gated_out   [B * value_dim]
// Веса (conv_weights, dt_bias, ssm_a, norm_weight) — shared across slots, без B.
// Входы от batched QMatMul: qkv_raw [B*channels], z_raw [B*value_dim],
//   beta_raw [B*n_v_heads], alpha_raw [B*n_v_heads] (row-major, leading B).

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════
// Параметры (layout ДОЛЖЕН совпадать со DeltaParams в delta_rule_batched_metal.rs)
// ═══════════════════════════════════════════════════════════════

struct DeltaParams {
    uint n_k_heads;       // 16
    uint n_v_heads;       // 32
    uint head_k_dim;      // 128
    uint head_v_dim;      // 128
    uint key_dim;         // n_k_heads * head_k_dim = 2048
    uint value_dim;       // n_v_heads * head_v_dim = 4096
    uint channels;        // key_dim * 2 + value_dim = 8192
    uint conv_kernel;    // 4
    float q_scale;        // 1 / sqrt(head_k_dim)
    float rms_norm_eps;   // 1e-6
    uint heads_per_kv;   // n_v_heads / n_k_heads = 2 (GQA)
    uint batch_size;      // B — число одновременных слотов
};

// ═══════════════════════════════════════════════════════════════
// Вспомогательные функции (копия из delta_rule.metal — identical math)
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
// Kernel 1: delta_conv1d_prep_batched
//
// Conv1d step с SiLU + sigmoid(beta) + softplus(alpha)*A, per slot.
//
// Grid: (channels, B, 1)  — один поток на (channel, slot)
// Threadgroup: (256, 1, 1)
//
// conv_state layout = [B, conv_k-1, channels] row-major
// conv_weights layout = [channels, conv_k] row-major (shared, без B)
// ═══════════════════════════════════════════════════════════════

// F32 для precision (см. комментарий в delta_rule.metal — decode precision critical).
kernel void delta_conv1d_prep_batched(
    device const float* qkv_raw        [[buffer(0)]],   // [B * channels] — from batched QMatMul (batch_idx)
    device const float* beta_raw       [[buffer(1)]],   // [B * n_v_heads] (batch_idx)
    device const float* alpha_raw      [[buffer(2)]],   // [B * n_v_heads] (batch_idx)
    device const float* conv_weights   [[buffer(3)]],   // [channels * conv_k] — shared
    device const float* dt_bias        [[buffer(4)]],   // [n_v_heads] — shared
    device const float* ssm_a          [[buffer(5)]],   // [n_v_heads] — shared
    device float*       conv_state     [[buffer(6)]],   // [CAP * (conv_k-1) * channels] — persistent (real slot_idx)
    device float*       qkv_conv_out   [[buffer(7)]],   // [B * channels] — output (batch_idx)
    device float*       beta_out       [[buffer(8)]],   // [B * n_v_heads] (batch_idx, temp)
    device float*       gate_out       [[buffer(9)]],   // [B * n_v_heads] (batch_idx, temp)
    constant DeltaParams& params       [[buffer(10)]],
    constant const uint* slot_ids      [[buffer(11)]],  // [B] — batch_idx → real slot_idx (indirection)
    uint2 tid [[thread_position_in_grid]]              // (channel, batch_idx)
) {
    uint channels = params.channels;
    uint conv_k = params.conv_kernel;
    uint n_v = params.n_v_heads;

    uint ch = tid.x;
    uint bidx = tid.y;            // позиция в активном batch'е (0..B-1)

    if (bidx >= params.batch_size) return;

    // Persistent conv_state индексируется по реальному slot_idx (indirection);
    // входы/выходы (qkv_raw, qkv_conv_out, beta_raw, beta_out, gate_out) — по batch_idx.
    uint real_slot = slot_ids[bidx];

    // ── Часть A: Conv1d step (per (channel, batch_idx)) ──
    if (ch < channels) {
        uint slot_conv_base = real_slot * (conv_k - 1) * channels;  // persistent
        uint slot_qkv_base = bidx * channels;                         // input/temp (batch_idx)

        float sum = 0.0f;
        uint weight_base = ch * conv_k;

        // Свёртка с предыдущими входами из буфера (этого slot'а)
        for (uint i = 0; i < conv_k - 1; i++) {
            sum += conv_state[slot_conv_base + i * channels + ch] * conv_weights[weight_base + i];
        }
        // Текущий вход
        sum += qkv_raw[slot_qkv_base + ch] * conv_weights[weight_base + conv_k - 1];

        // SiLU активация
        qkv_conv_out[slot_qkv_base + ch] = silu(sum);

        // Обновление conv_state для этого slot'а: сдвиг влево + запись нового входа
        if (conv_k > 2) {
            for (uint i = 0; i < conv_k - 2; i++) {
                conv_state[slot_conv_base + i * channels + ch] =
                    conv_state[slot_conv_base + (i + 1) * channels + ch];
            }
        }
        conv_state[slot_conv_base + (conv_k - 2) * channels + ch] = qkv_raw[slot_qkv_base + ch];
    }

    // ── Часть B: sigmoid(beta) и softplus(alpha)*A (per (batch_idx, head)) ──
    // beta_out/gate_out — TEMP, индекс по batch_idx. beta_raw/alpha_raw — входы (batch_idx).
    if (ch < n_v) {
        uint slot_head = bidx * n_v + ch;  // batch_idx для temp/входов
        beta_out[slot_head] = sigmoid(beta_raw[slot_head]);
        float alpha_biased = alpha_raw[slot_head] + dt_bias[ch];
        gate_out[slot_head] = softplus(alpha_biased) * ssm_a[ch];
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 2: delta_l2_norm_expand_batched
//
// L2 нормализация Q и K per k_head, расширение голов 16→32,
// Q scaling, копирование V — per (head, slot).
//
// Grid: threadgroups (n_v_heads, B, 1), each threadgroup (1, head_k_dim, 1)
// Один threadgroup = один (v_head, slot) — shared memory reduction внутри.
// ═══════════════════════════════════════════════════════════════

kernel void delta_l2_norm_expand_batched(
    device const float* qkv_conv       [[buffer(0)]],   // [B * channels] — conv1d output
    device float*       q_out          [[buffer(1)]],   // [B * n_v_heads * head_k_dim]
    device float*       k_out          [[buffer(2)]],   // [B * n_v_heads * head_k_dim]
    device float*       v_out          [[buffer(3)]],   // [B * n_v_heads * head_v_dim]
    constant DeltaParams& params       [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],        // (head_v, slot, 0)
    uint3 lid  [[thread_position_in_threadgroup]]       // (0, dim, 0)
) {
    uint head_v = tgid.x;          // 0..31
    uint slot = tgid.y;            // 0..B-1
    uint dim = lid.y;              // 0..127
    uint hkd = params.head_k_dim;  // 128
    uint hvd = params.head_v_dim;  // 128
    uint key_dim = params.key_dim;
    uint n_v = params.n_v_heads;

    // slot offsets (leading B)
    uint slot_qkv = slot * params.channels;          // qkv_conv per slot base
    uint slot_q = slot * n_v * hkd;                  // q_out per slot base
    uint slot_k = slot * n_v * hkd;                  // k_out per slot base
    uint slot_v = slot * n_v * hvd;                  // v_out per slot base

    // Маппинг v-head → k-head (GQA: 32 v-heads → 16 k-heads) — same as original
    uint head_k = head_v % params.n_k_heads;

    // ── Shared memory для L2 norm reduction ──
    threadgroup float sq_sum_q[128];
    threadgroup float sq_sum_k[128];

    // Читаем Q и K raw значения для этого k-head (в рамках slot'а)
    uint q_idx = slot_qkv + head_k * hkd + dim;
    uint k_idx = slot_qkv + key_dim + head_k * hkd + dim;  // K после Q в qkv_conv

    float q_val = qkv_conv[q_idx];
    float k_val = qkv_conv[k_idx];

    // Загружаем квадраты для reduction
    sq_sum_q[dim] = q_val * q_val;
    sq_sum_k[dim] = k_val * k_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Parallel reduction (128 → 64 → ... → 1) ──
    for (uint stride = hkd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_sum_q[dim] += sq_sum_q[dim + stride];
            sq_sum_k[dim] += sq_sum_k[dim + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // sq_sum_q[0] и sq_sum_k[0] — суммы квадратов по k-head
    float q_inv_norm = rsqrt(sq_sum_q[0] + 1e-6f);
    float k_inv_norm = rsqrt(sq_sum_k[0] + 1e-6f);

    // ── Нормализация + Q scaling + запись (per v-head, per slot) ──
    uint out_idx = slot_q + head_v * hkd + dim;
    q_out[out_idx] = q_val * q_inv_norm * params.q_scale;
    k_out[slot_k + head_v * hkd + dim] = k_val * k_inv_norm;

    // ── Копирование V (прямое, без нормализации) — per v-head, per slot ──
    uint v_src_idx = slot_qkv + key_dim * 2 + head_v * hvd + dim;
    uint v_dst_idx = slot_v + head_v * hvd + dim;
    v_out[v_dst_idx] = qkv_conv[v_src_idx];
}

// ═══════════════════════════════════════════════════════════════
// Kernel 3: delta_rule_kernel_batched
//
// Основной рекуррентный шаг DeltaNet, per (head, slot).
//
// Grid: threadgroups (n_v_heads, B, 1), each threadgroup (1, head_v_dim, 1)
// Один threadgroup = один (head, slot) — shared memory reduction внутри,
// register/state изоляция автоматична (разные threadgroups не делят shared mem).
//
// state layout: [B * n_v_heads * hd * hd], per (slot, head) = [hd (rows) × hd (cols)]
// Row-major: state[slot][head][row][col] = ssm_state[slot * n_v*hd*hd + head*hd*hd + row*hd + col]
//
// Алгоритм (идентичен delta_rule.metal — только slot offset в state/vectors):
// 1. Decay: state[row][col] *= exp(gate[slot][head])
// 2. sk[col] = sum_row(state[row][col] * k[slot][head][row])   (S^T @ k)
// 3. d[col] = (v[slot][head][col] - sk[col]) * beta[slot][head]
// 4. state[row][col] += k[slot][head][row] * d[col]            (rank-1 update)
// 5. output[slot][head][col] = sum_row(state[row][col] * q[slot][head][row]) (S^T @ q)
// ═══════════════════════════════════════════════════════════════

kernel void delta_rule_kernel_batched(
    device const float* q              [[buffer(0)]],   // [B * n_v_heads * head_k_dim] (batch_idx, temp)
    device const float* k              [[buffer(1)]],   // [B * n_v_heads * head_k_dim] (batch_idx, temp)
    device const float* v              [[buffer(2)]],   // [B * n_v_heads * head_v_dim] (batch_idx, temp)
    device const float* beta           [[buffer(3)]],   // [B * n_v_heads] (batch_idx, temp)
    device const float* gate           [[buffer(4)]],   // [B * n_v_heads] (batch_idx, temp)
    device float*       ssm_state      [[buffer(5)]],   // [CAP * n_v_heads * hd * hd] — persistent (real slot_idx)
    device float*       output         [[buffer(6)]],   // [B * n_v_heads * head_v_dim] (batch_idx, temp)
    constant DeltaParams& params       [[buffer(7)]],
    constant const uint* slot_ids      [[buffer(8)]],   // [B] — batch_idx → real slot_idx (indirection)
    uint3 tgid [[threadgroup_position_in_grid]],        // (head, batch_idx, 0)
    uint3 lid  [[thread_position_in_threadgroup]]       // (0, col, 0)
) {
    uint head = tgid.x;       // 0..31
    uint bidx = tgid.y;       // 0..B-1 (позиция в активном batch'е)
    uint col = lid.y;         // 0..127
    uint hd = params.head_v_dim;  // 128
    uint n_v = params.n_v_heads;

    // Persistent ssm_state — по реальному slot_idx (indirection);
    // temp q/k/v/beta/gate/output — по batch_idx.
    uint real_slot = slot_ids[bidx];
    uint slot_head = bidx * n_v + head;                     // temp beta/gate (batch_idx)
    uint state_base = real_slot * n_v * hd * hd + head * hd * hd;  // persistent state
    uint vec_base = bidx * n_v * hd + head * hd;              // temp q/k/v/output (batch_idx)

    // ── Shared memory для sk и d (один threadgroup = (head, slot), изоляция автоматична) ──
    threadgroup float shared_sk[128];
    threadgroup float shared_d[128];

    // ── 1. Decay: state[row][col] *= exp(gate[slot][head]) ──
    float gate_exp = exp(gate[slot_head]);
    for (uint row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] *= gate_exp;
    }

    // Барьер не нужен — каждый поток пишет в свой столбец в этом threadgroup

    // ── 2. sk[col] = sum_row(state[row][col] * k[slot][head][row]) ──
    float sk_val = 0.0f;
    for (uint row = 0; row < hd; row++) {
        sk_val += ssm_state[state_base + row * hd + col] * k[vec_base + row];
    }
    shared_sk[col] = sk_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. d[col] = (v[slot][head][col] - sk[col]) * beta[slot][head] ──
    float beta_h = beta[slot_head];
    float d_val = (v[vec_base + col] - shared_sk[col]) * beta_h;
    shared_d[col] = d_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 4. Rank-1 update: state[row][col] += k[slot][head][row] * d[col] ──
    float d_col = shared_d[col];
    for (uint row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] += k[vec_base + row] * d_col;
    }

    // ── 5. output[slot][head][col] = sum_row(state[row][col] * q[slot][head][row]) ──
    float out_val = 0.0f;
    for (uint row = 0; row < hd; row++) {
        out_val += ssm_state[state_base + row * hd + col] * q[vec_base + row];
    }
    output[vec_base + col] = out_val;
}

// ═══════════════════════════════════════════════════════════════
// Kernel 4: delta_norm_gate_kernel_batched
//
// Group RMS Norm per (head, slot) + SiLU(z) gating, per slot.
//
// Grid: threadgroups (n_v_heads, B, 1), each threadgroup (1, head_v_dim, 1)
//
// Алгоритм (идентичен delta_rule.metal — slot offset в vectors/z/output):
// 1. sq_mean = mean(output[slot][head]^2) — threadgroup reduction
// 2. inv_rms = 1 / sqrt(sq_mean + eps)
// 3. gated[slot][head*hvd+i] = output[..] * inv_rms * norm_weight[i % hvd] * silu(z[slot*value_dim + head*hvd + i])
// ═══════════════════════════════════════════════════════════════

kernel void delta_norm_gate_kernel_batched(
    device const float* raw_output     [[buffer(0)]],   // [B * n_v_heads * head_v_dim]
    device const float* z              [[buffer(1)]],   // [B * value_dim]
    device const float* norm_weight    [[buffer(2)]],   // [head_v_dim] — shared across slots/heads
    device float*       gated_output   [[buffer(3)]],   // [B * value_dim]
    constant DeltaParams& params       [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],        // (head, slot, 0)
    uint3 lid  [[thread_position_in_threadgroup]]       // (0, dim, 0)
) {
    uint head = tgid.x;       // 0..31
    uint slot = tgid.y;       // 0..B-1
    uint dim = lid.y;         // 0..127
    uint hvd = params.head_v_dim;
    uint n_v = params.n_v_heads;
    float eps = params.rms_norm_eps;

    uint idx = slot * n_v * hvd + head * hvd + dim;     // raw_output per (slot, head)
    uint z_idx = slot * params.value_dim + head * hvd + dim;
    uint out_idx = slot * params.value_dim + head * hvd + dim;

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

    // sq_vals[0] = сумма квадратов по (slot, head)
    float inv_rms = rsqrt(sq_vals[0] / float(hvd) + eps);

    // ── Norm * weight * SiLU(z) — per (slot, head) ──
    float normed = val * inv_rms * norm_weight[dim];
    float z_val = z[z_idx];
    gated_output[out_idx] = normed * silu(z_val);
}