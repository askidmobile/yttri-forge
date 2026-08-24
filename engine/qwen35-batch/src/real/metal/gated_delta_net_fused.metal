// gated_delta_net_fused.metal — Batch-aware fused Metal kernels для DeltaNet
//
// Phase 1 порта fused Gated DeltaNet kernel из llama.cpp (T-265).
//
// Заменяет 4 раздельных per-token Metal kernel'а из delta_rule.metal на
// 4 batch-aware kernel'а с token loop ВНУТРИ. Это позволяет одним dispatch'ом
// обработать всю seq_len, радикально снижая CPU dispatch overhead и
// устраняя CPU prefill через Accelerate.
//
// Размерности (Qwen3.5-4B):
//   n_v_heads = 16, head_v_dim = head_k_dim = 128
//   key_dim = value_dim = 16 * 128 = 2048
//   channels = key_dim*2 + value_dim = 6144
//   conv_kernel = 4
//   heads_per_kv = 1 (без head expansion в этой модели)
//
// 4 kernel'а:
//   1. gdn_conv1d_batch_prep   — conv1d step + sigmoid(beta) + softplus(alpha+dt)*ssm_a
//                                token loop ВНУТРИ, conv_state sequential update
//   2. gdn_l2_norm_batch        — L2 normalize Q/K + Q scaling + V copy
//                                токены параллельно через grid
//   3. gdn_recurrent_batch      — recurrent step (по llama.cpp gated_delta_net_impl)
//                                token loop ВНУТРИ, state в регистрах через simd_sum
//   4. gdn_norm_gate_batch      — RMS norm per head + SiLU(z) gate
//                                токены параллельно через grid
//
// ПРИМЕЧАНИЕ: kernel 3+4 разделены, потому что объединить fused recurrent + RMS
// в один kernel с разной thread group geometry сложно. RMS reduction по всему
// head value_dim требует threadgroup shared memory, а recurrent держит state
// в регистрах через simd_sum. 4 batch-aware kernel'а вместо текущих 4×T —
// это всё равно радикальное сокращение dispatch overhead.

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════
// Константы (зашиты для Qwen3.5-2B/4B варианта)
// ═══════════════════════════════════════════════════════════════

// NSG = NumStateRowsperThread — каждый thread держит 4 строки state в регистрах
// 32 (simdgroup width) × 4 = 128 = head_v_dim
constexpr constant short NSG = 4;

// Размерности модели — захардкожены для Qwen3.5
constexpr constant short HEAD_V_DIM = 128;
constexpr constant short HEAD_K_DIM = 128;

// ═══════════════════════════════════════════════════════════════
// Общая структура параметров (передаётся через constant buffer)
// ═══════════════════════════════════════════════════════════════

struct GdnFusedParams {
    uint n_tokens;       // T — кол-во токенов в batch
    uint channels;       // key_dim*2 + value_dim
    uint key_dim;        // n_k_heads * head_k_dim
    uint value_dim;      // n_v_heads * head_v_dim
    float q_scale;       // 1 / sqrt(head_k_dim)
    float rms_norm_eps;  // 1e-6
    uint n_v_heads;      // 16 (Qwen3.5-2B), 32 (Qwen3.5-4B)
    uint conv_kernel;    // 4
    uint n_k_heads;      // 16 (Qwen3.5-2B и 4B); GQA expansion: head_k = head_v % n_k_heads
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
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log(1.0f + exp(x));
}

// ═══════════════════════════════════════════════════════════════
// Kernel 1: gdn_conv1d_batch_prep
//
// Per-channel conv1d step + sigmoid(beta) + softplus(alpha+dt)*ssm_a
// для всей последовательности из T токенов.
//
// Каждый thread обрабатывает ОДИН канал (или элемент beta/alpha если tid<n_v_heads).
// Token loop ВНУТРИ — conv_state обновляется sequential между токенами.
//
// Grid:        (channels) threads — каждый thread = один channel
// Threadgroup: 256 threads (или меньше если channels < 256)
//
// conv_state layout: [conv_k-1, channels] row-major, persistent между вызовами
// conv_weights layout: [channels, conv_k] row-major
// qkv_t layout: [T, channels] row-major
// qkv_conv_out layout: [T, channels] row-major
// beta_out / gate_out layout: [T, n_v_heads] row-major
// ═══════════════════════════════════════════════════════════════

// F32 baseline (как llama.cpp ggml_gated_delta_net). F16 I/O вариант пробовался
// но даёт cumulative drift на длинных prefill (>1K токенов): F16 промежуточные
// q/k/v/beta/gate буферы загружаются recurrent loop'ом и quantization error
// накапливается в ssm_state F32 через каждый токен.
kernel void gdn_conv1d_batch_prep(
    device const float* qkv_t          [[buffer(0)]],   // [T, channels]
    device const float* beta_t         [[buffer(1)]],   // [T, n_v_heads]
    device const float* alpha_t        [[buffer(2)]],   // [T, n_v_heads]
    device const float* conv_weights   [[buffer(3)]],   // [channels, conv_k]
    device const float* dt_bias        [[buffer(4)]],   // [n_v_heads]
    device const float* ssm_a          [[buffer(5)]],   // [n_v_heads]
    device float*       conv_state     [[buffer(6)]],   // [(conv_k-1) * channels] persistent
    device float*       qkv_conv_out   [[buffer(7)]],   // [T, channels]
    device float*       beta_out       [[buffer(8)]],   // [T, n_v_heads]
    device float*       gate_out       [[buffer(9)]],   // [T, n_v_heads]
    constant GdnFusedParams& params    [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    uint channels = params.channels;
    uint conv_k = params.conv_kernel;
    uint n_tokens = params.n_tokens;
    uint n_v_heads = params.n_v_heads;

    // ── Часть A: Conv1d по всем токенам для своего канала ──
    if (tid < channels) {
        uint weight_base = tid * conv_k;

        // Загружаем веса своего канала в регистры (conv_k=4 значений)
        float w0 = conv_weights[weight_base + 0];
        float w1 = conv_weights[weight_base + 1];
        float w2 = conv_weights[weight_base + 2];
        float w3 = conv_weights[weight_base + 3];

        // Загружаем conv_state в регистры (3 значения для conv_k=4)
        float s0 = conv_state[0 * channels + tid];
        float s1 = conv_state[1 * channels + tid];
        float s2 = conv_state[2 * channels + tid];

        for (uint t = 0; t < n_tokens; t++) {
            float x_cur = qkv_t[t * channels + tid];

            float sum = s0 * w0 + s1 * w1 + s2 * w2 + x_cur * w3;
            qkv_conv_out[t * channels + tid] = silu(sum);

            s0 = s1;
            s1 = s2;
            s2 = x_cur;
        }

        conv_state[0 * channels + tid] = s0;
        conv_state[1 * channels + tid] = s1;
        conv_state[2 * channels + tid] = s2;
    }

    if (tid < n_v_heads) {
        uint head = tid;
        float dt_b = dt_bias[head];
        float a = ssm_a[head];

        for (uint t = 0; t < n_tokens; t++) {
            float beta_raw = beta_t[t * n_v_heads + head];
            float alpha_raw = alpha_t[t * n_v_heads + head];

            beta_out[t * n_v_heads + head] = sigmoid(beta_raw);
            float alpha_biased = alpha_raw + dt_b;
            gate_out[t * n_v_heads + head] = softplus(alpha_biased) * a;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 2: gdn_l2_norm_batch
//
// L2 нормализация Q/K per head, Q scaling, копирование V.
// Каждый токен независим — параллелизуются через grid.
//
// Grid:        (n_v_heads, T) threadgroups
// Threadgroup: (1, head_k_dim=128) threads
//
// Threadgroup memory используется для tree reduction суммы квадратов.
// ═══════════════════════════════════════════════════════════════

kernel void gdn_l2_norm_batch(
    device const float* qkv_conv       [[buffer(0)]],   // [T, channels]
    device float*       q_out          [[buffer(1)]],   // [T, n_v_heads * head_k_dim]
    device float*       k_out          [[buffer(2)]],   // [T, n_v_heads * head_k_dim]
    device float*       v_out          [[buffer(3)]],   // [T, n_v_heads * head_v_dim]
    constant GdnFusedParams& params    [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],          // (head_idx, t)
    uint2 lid [[thread_position_in_threadgroup]]        // (0, dim)
) {
    uint head_v = tg.x;       // 0..n_v_heads-1
    uint t = tg.y;            // 0..n_tokens-1
    uint dim = lid.y;         // 0..head_k_dim-1
    uint hkd = HEAD_K_DIM;    // 128
    uint hvd = HEAD_V_DIM;    // 128
    uint key_dim = params.key_dim;
    uint channels = params.channels;

    // T-282 ROOT CAUSE FIX: GQA expansion в DeltaNet.
    // Qwen3.5-4B: n_k_heads=16, n_v_heads=32 → 2 V per K. Без modulo
    // head_k=head_v читал K-данные за пределами Q-region для head_v=16..31,
    // corrupting Q/K после L2-norm и ломая translate (T-273 root cause).
    // Decode kernel `delta_rule.metal` уже делал `head_v % n_k_heads`, fused
    // prefill kernel пропустил GQA expansion. Aligned with llama.cpp behavior.
    uint head_k = head_v % params.n_k_heads;

    threadgroup float sq_sum_q[128];
    threadgroup float sq_sum_k[128];

    uint qkv_t_base = t * channels;

    uint q_idx = qkv_t_base + head_k * hkd + dim;
    uint k_idx = qkv_t_base + key_dim + head_k * hkd + dim;

    float q_val = qkv_conv[q_idx];
    float k_val = qkv_conv[k_idx];

    sq_sum_q[dim] = q_val * q_val;
    sq_sum_k[dim] = k_val * k_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = hkd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_sum_q[dim] += sq_sum_q[dim + stride];
            sq_sum_k[dim] += sq_sum_k[dim + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float q_inv_norm = rsqrt(sq_sum_q[0] + 1e-6f);
    float k_inv_norm = rsqrt(sq_sum_k[0] + 1e-6f);

    uint q_out_idx = t * params.value_dim + head_v * hkd + dim;
    uint k_out_idx = t * params.value_dim + head_v * hkd + dim;
    uint v_out_idx = t * params.value_dim + head_v * hvd + dim;

    q_out[q_out_idx] = q_val * q_inv_norm * params.q_scale;
    k_out[k_out_idx] = k_val * k_inv_norm;

    uint v_src_idx = qkv_t_base + key_dim * 2 + head_v * hvd + dim;
    v_out[v_out_idx] = qkv_conv[v_src_idx];
}

// ═══════════════════════════════════════════════════════════════
// Kernel 3: gdn_recurrent_batch
//
// Recurrent step DeltaNet по образцу llama.cpp kernel_gated_delta_net_impl.
// Token loop ВНУТРИ. State (NSG=4 строк) держится в регистрах,
// reduction через simd_sum.
//
// Grid:        (HEAD_V_DIM/NSG=32, n_v_heads) threadgroups
// Threadgroup: (32, NSG=4) threads = 128 thread'ов = 4 simdgroup'а
//
// Каждая (i20, head) пара — это один столбец state в одном head.
// i20 = tg.x * NSG + ty — глобальный column index в head value_dim.
// tx = thread_position_in_threadgroup.x = 0..31 (simdgroup lane)
// ty = thread_position_in_threadgroup.y = 0..NSG-1 (выбор столбца внутри блока NSG)
//
// Каждый thread держит NSG=4 строк state в регистрах ls[NSG]:
//   is = tx*NSG + j (j=0..NSG-1) — индекс строки (0..127)
//   ls[j] = state[head][is][i20]
//
// State у нас row-major: state[head][row][col]
//   ssm_state[head * hd * hd + row * hd + col]
//
// Алгоритм по llama.cpp:
//   ls = load_state(head, rows, col=i20)
//   for t in 0..T:
//     g = exp(gate[t][head])
//     ls *= g
//     s_k = sum_simd(ls[j] * k[t][head][rows[j]])
//     d = (v[t][head][i20] - s_k) * beta[t][head]
//     ls += k[t][head][rows] * d
//     y = sum_simd(ls[j] * q[t][head][rows[j]])
//     if tx == 0:  raw_output[t][head][i20] = y
//   store_state(head, rows, i20, ls)
//
// ВАЖНО: в Metal simd_sum reduction идёт по simdgroup (32 lane'а tx=0..31).
// y и s_k агрегируются по строкам (rows = tx*NSG + j для j=0..NSG-1) —
// это все 128 строк головы, разбросанные между 32 lane'ами по 4 на каждом.
// ═══════════════════════════════════════════════════════════════

kernel void gdn_recurrent_batch(
    device const float* q              [[buffer(0)]],   // [T, n_v_heads * head_k_dim]
    device const float* k              [[buffer(1)]],   // [T, n_v_heads * head_k_dim]
    device const float* v              [[buffer(2)]],   // [T, n_v_heads * head_v_dim]
    device const float* beta           [[buffer(3)]],   // [T, n_v_heads]
    device const float* gate           [[buffer(4)]],   // [T, n_v_heads]
    device float*       ssm_state      [[buffer(5)]],   // [n_v_heads * hd * hd] persistent
    device float*       raw_output     [[buffer(6)]],   // [T, n_v_heads * head_v_dim]
    constant GdnFusedParams& params    [[buffer(7)]],
    uint2 tg    [[threadgroup_position_in_grid]],       // (col_block, head)
    uint2 tpitg [[thread_position_in_threadgroup]]      // (32, NSG)
) {
    const uint tx = tpitg.x;          // 0..31 — simdgroup lane
    const uint ty = tpitg.y;          // 0..NSG-1
    const uint head = tg.y;           // 0..n_v_heads-1
    const uint i20 = tg.x * NSG + ty; // 0..HEAD_V_DIM-1 — column index в head

    const uint hd = HEAD_V_DIM;       // = HEAD_K_DIM = 128
    const uint n_tokens = params.n_tokens;
    const uint n_v_heads = params.n_v_heads;
    const uint value_dim = params.value_dim;

    const uint state_head_base = head * hd * hd;

    float ls[NSG];
    for (short j = 0; j < NSG; j++) {
        const uint is = tx * NSG + j;
        ls[j] = ssm_state[state_head_base + is * hd + i20];
    }

    for (uint t = 0; t < n_tokens; t++) {
        const uint vec_t_base = t * value_dim + head * hd;
        const float beta_h = beta[t * n_v_heads + head];
        const float gate_h = gate[t * n_v_heads + head];

        const float g_exp = exp(gate_h);
        float s_k = 0.0f;
        for (short j = 0; j < NSG; j++) {
            const uint is = tx * NSG + j;
            ls[j] *= g_exp;
            s_k += ls[j] * k[vec_t_base + is];
        }
        s_k = simd_sum(s_k);

        const float d = (v[vec_t_base + i20] - s_k) * beta_h;

        float y = 0.0f;
        for (short j = 0; j < NSG; j++) {
            const uint is = tx * NSG + j;
            ls[j] += k[vec_t_base + is] * d;
            y += ls[j] * q[vec_t_base + is];
        }
        y = simd_sum(y);

        if (tx == 0) {
            raw_output[vec_t_base + i20] = y;
        }
    }

    for (short j = 0; j < NSG; j++) {
        const uint is = tx * NSG + j;
        ssm_state[state_head_base + is * hd + i20] = ls[j];
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 4: gdn_norm_gate_batch
//
// Group RMS Norm per head + SiLU(z) gating, batch-aware.
// Каждый (head, t) обрабатывается одним threadgroup.
//
// Grid:        (n_v_heads, T) threadgroups
// Threadgroup: (1, head_v_dim=128) threads
//
// Алгоритм идентичен per-token delta_norm_gate_kernel, но с batch offset'ами.
// ═══════════════════════════════════════════════════════════════

kernel void gdn_norm_gate_batch(
    device const float* raw_output     [[buffer(0)]],   // [T, n_v_heads * head_v_dim]
    device const float* z              [[buffer(1)]],   // [T, value_dim]
    device const float* norm_weight    [[buffer(2)]],   // [head_v_dim] shared across heads
    device float*       gated_output   [[buffer(3)]],   // [T, value_dim]
    constant GdnFusedParams& params    [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],          // (head, t)
    uint2 lid [[thread_position_in_threadgroup]]        // (0, dim)
) {
    uint head = tg.x;
    uint t = tg.y;
    uint dim = lid.y;
    uint hvd = HEAD_V_DIM;
    float eps = params.rms_norm_eps;
    uint value_dim = params.value_dim;

    uint idx = t * value_dim + head * hvd + dim;

    threadgroup float sq_vals[128];

    float val = raw_output[idx];
    sq_vals[dim] = val * val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = hvd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_vals[dim] += sq_vals[dim + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(sq_vals[0] / float(hvd) + eps);

    float normed = val * inv_rms * norm_weight[dim];
    float z_val = z[idx];
    gated_output[idx] = normed * silu(z_val);
}
