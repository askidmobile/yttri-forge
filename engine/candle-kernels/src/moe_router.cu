// MoE routing kernels — device-side only, no CUDA Runtime API.
// PTX-compatible via cudarc dynamic-loading stack.
//
// Pipeline:
//   1. moe_softmax_topk_kernel: per-token softmax + stable iterative argmax top-k
//   2. moe_sort_by_expert_kernel: produce sorted (expert_id, token_id, weight) triples
//   3. moe_exclusive_scan_kernel: Hillis-Steele exclusive scan → expert_offsets
//
// All workspace buffers pre-allocated by caller. No cudaMallocAsync in hot path.

#include "cuda_fp16.h"
#include <stdint.h>

#define WARP_SIZE 32

// ─── Warp reduction helpers ──────────────────────────────────────────────────

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

// ─── 1. Softmax + Stable Top-K ────────────────────────────────────────────────
//
// One block per token. Block dim = max(n_experts, 32) rounded to power of 2.
// Shared mem: probs[n_experts] + masked[n_experts] (as uint8).
//
// llama.cpp-compatible: strict `>` in argmax → tie-break by lower expert index.
// norm_topk_prob: weights /= sum(selected probs).

extern "C" __global__ void moe_softmax_topk_kernel(
    const float* __restrict__ logits,  // [n_tokens, n_experts]
    int32_t* __restrict__ expert_ids,  // [n_tokens * topk] — selected expert indices
    float* __restrict__ weights,       // [n_tokens * topk] — normalized weights
    int n_experts,
    int topk,
    int norm_topk_prob  // 0 or 1
) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;

    // Shared memory for probs and mask
    extern __shared__ char smem[];
    float* probs = reinterpret_cast<float*>(smem);
    uint8_t* masked = reinterpret_cast<uint8_t*>(smem + n_experts * sizeof(float));

    // --- Load logits and compute max ---
    float max_val = -INFINITY;
    if (tid < n_experts) {
        probs[tid] = logits[token * n_experts + tid];
        max_val = probs[tid];
    }
    // Block-level max reduction (for block_dim > 32)
    max_val = warp_reduce_max(max_val);
    // Inter-warp reduction via shared mem if block_dim > WARP_SIZE
    if (block_dim > WARP_SIZE) {
        __shared__ float warp_max[32];
        if (tid % WARP_SIZE == 0) {
            warp_max[tid / WARP_SIZE] = max_val;
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            float val = (tid < (block_dim + WARP_SIZE - 1) / WARP_SIZE)
                          ? warp_max[tid]
                          : -INFINITY;
            val = warp_reduce_max(val);
            if (tid == 0) warp_max[0] = val;
        }
        __syncthreads();
        max_val = warp_max[0];
    }

    // --- Softmax ---
    float exp_val = 0.0f;
    if (tid < n_experts) {
        float e = expf(probs[tid] - max_val);
        probs[tid] = e;
        exp_val = e;
    }
    float sum = warp_reduce_sum(exp_val);
    if (block_dim > WARP_SIZE) {
        __shared__ float warp_sum[32];
        if (tid % WARP_SIZE == 0) {
            warp_sum[tid / WARP_SIZE] = sum;
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            float val = (tid < (block_dim + WARP_SIZE - 1) / WARP_SIZE)
                          ? warp_sum[tid]
                          : 0.0f;
            val = warp_reduce_sum(val);
            if (tid == 0) warp_sum[0] = val;
        }
        __syncthreads();
        sum = warp_sum[0];
    }

    if (tid < n_experts) {
        probs[tid] = (sum > 0.0f) ? probs[tid] / sum : 0.0f;
        masked[tid] = 0;
    }
    __syncthreads();

    // --- Iterative argmax top-k ---
    // Strict `>` → first max wins → tie-break by lower expert index (matches llama.cpp).
    __shared__ float selected_weight_sum;
    if (tid == 0) selected_weight_sum = 0.0f;
    __syncthreads();
    for (int k = 0; k < topk; k++) {
        // Each thread finds local max among unmasked experts
        float best_val = -INFINITY;
        int best_idx = -1;
        for (int i = tid; i < n_experts; i += block_dim) {
            if (!masked[i] && probs[i] > best_val) {
                best_val = probs[i];
                best_idx = i;
            }
        }

        // Block-level reduction to find global max.
        // Двухступенчато: warp-reduce с tie-break по индексу, затем по warp'ам
        // через shared. Раньше: shared_val[32] только для tid<32 — части
        // потоков 32..255 терялись → выбор только среди экспертов 0..31
        // (сломанный роутинг на 256 экспертах, мусорная генерация).
        __shared__ float warp_val[32];
        __shared__ int warp_idx[32];
        const unsigned int lane = tid % WARP_SIZE;
        const unsigned int wid = tid / WARP_SIZE;
        const unsigned int nwarp = (block_dim + WARP_SIZE - 1) / WARP_SIZE;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_val, o);
            int oi = __shfl_xor_sync(0xffffffff, best_idx, o);
            if (ov > best_val || (ov == best_val && oi >= 0 && (best_idx < 0 || oi < best_idx))) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            warp_val[wid] = best_val;
            warp_idx[wid] = best_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float bv = warp_val[0];
            int bi = warp_idx[0];
            for (unsigned int i = 1; i < nwarp; i++) {
                // Strict `>` → lower index wins on tie
                if (warp_val[i] > bv || (warp_val[i] == bv && warp_idx[i] >= 0 && (bi < 0 || warp_idx[i] < bi))) {
                    bv = warp_val[i];
                    bi = warp_idx[i];
                }
            }
            int sel = bi;
            if (sel >= 0) {
                masked[sel] = 1;
                expert_ids[token * topk + k] = sel;
                float w = probs[sel];
                weights[token * topk + k] = w;
                selected_weight_sum += w;
            } else {
                expert_ids[token * topk + k] = 0;
                weights[token * topk + k] = 0.0f;
            }
        }
        __syncthreads();
    }

    // --- Normalize weights if norm_topk_prob ---
    if (norm_topk_prob && selected_weight_sum > 0.0f && tid < topk) {
        weights[token * topk + tid] /= selected_weight_sum;
    }
}

// ─── 2. Sort by Expert ────────────────────────────────────────────────────────
//
// Produces three sorted arrays from (token_id, expert_id, weight) triples:
//   sorted_token_ids[m_total]  — token index for each (token,expert) pair
//   sorted_expert_ids[m_total] — expert index for each pair
//   sorted_weights[m_total]    — weight for each pair
//
// One block per expert. Each block scans all token*topk pairs, collects those
// matching its expert_id, writes them contiguously starting at expert_offset.
//
// m_total = n_tokens * topk.

extern "C" __global__ void moe_sort_by_expert_kernel(
    const int32_t* __restrict__ expert_ids,  // [n_tokens * topk]
    const float* __restrict__ weights,       // [n_tokens * topk]
    const int32_t* __restrict__ expert_offsets,  // [n_experts + 1] — exclusive scan result
    int32_t* __restrict__ sorted_token_ids,  // [m_total]
    int32_t* __restrict__ sorted_expert_ids,// [m_total]
    float* __restrict__ sorted_weights,     // [m_total]
    int n_tokens,
    int topk,
    int n_experts
) {
    const int expert = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;

    int write_pos = expert_offsets[expert];
    const int end_pos = expert_offsets[expert + 1];

    // Each thread scans a strided subset of the token*topk pairs
    for (int i = tid; i < n_tokens * topk; i += block_dim) {
        int token_idx = i / topk;
        int k_idx = i % topk;
        if (expert_ids[i] == expert) {
            // Atomic reservation of write position
            int my_pos = atomicAdd(&write_pos, 1);
            if (my_pos < end_pos) {
                sorted_token_ids[my_pos] = token_idx;
                sorted_expert_ids[my_pos] = expert;
                sorted_weights[my_pos] = weights[i];
            }
        }
    }
}

// ─── 3. Expert Offset Count + Exclusive Scan ───────────────────────────────────
//
// Two-phase: count tokens per expert, then exclusive scan → offsets.
// Phase A (count): atomicAdd per expert.
// Phase B (scan): Hillis-Steele exclusive scan in shared mem.
//
// Combined into one kernel: count first, sync, then scan.
// Assumes n_experts ≤ 1024 (fits in one block).

extern "C" __global__ void moe_count_and_scan_kernel(
    const int32_t* __restrict__ expert_ids,  // [n_tokens * topk]
    int32_t* __restrict__ expert_counts,    // [n_experts] — output counts
    int32_t* __restrict__ expert_offsets,   // [n_experts + 1] — output exclusive scan
    int n_tokens,
    int topk,
    int n_experts
) {
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int m_total = n_tokens * topk;

    // --- Phase A: Count tokens per expert ---
    // Each thread processes strided elements, atomicAdd to counts.
    // Counts must be zero-initialized by caller (or we zero here).
    if (tid < n_experts) {
        expert_counts[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < m_total; i += block_dim) {
        int e = expert_ids[i];
        if (e >= 0 && e < n_experts) {
            atomicAdd(&expert_counts[e], 1);
        }
    }
    __syncthreads();

    // --- Phase B: Exclusive scan (Hillis-Steele) ---
    // Load counts into shared mem
    extern __shared__ int32_t scan_temp[];
    if (tid < n_experts) {
        scan_temp[tid] = expert_counts[tid];
    } else if (tid < block_dim) {
        scan_temp[tid] = 0;
    }
    __syncthreads();

    // Hillis-Steele inclusive scan
    for (int offset = 1; offset < block_dim; offset <<= 1) {
        int32_t temp_val = 0;
        if (tid >= offset) {
            temp_val = scan_temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            scan_temp[tid] += temp_val;
        }
        __syncthreads();
    }

    // Write exclusive scan result: offsets[0] = 0, offsets[i+1] = inclusive_sum[i]
    if (tid == 0) {
        expert_offsets[0] = 0;
    }
    if (tid < n_experts) {
        expert_offsets[tid + 1] = scan_temp[tid];
    }
}

// ─── 4. Combine: Weighted Scatter-Add ──────────────────────────────────────────
//
// Takes sorted expert outputs [m_total, n_out] and scatters weighted values
// into final output [n_tokens, n_out] via atomicAdd.
//
// Grid: (m_total / TILE_M, n_out / TILE_N). Each block processes TILE_M pairs
// and TILE_N output columns.

extern "C" __global__ void moe_combine_kernel(
    const float* __restrict__ expert_outputs,  // [m_total, n_out]
    const int32_t* __restrict__ sorted_token_ids,  // [m_total]
    const float* __restrict__ sorted_weights,       // [m_total]
    float* __restrict__ output,                     // [n_tokens, n_out]
    int n_out,
    int m_total
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;

    if (row >= m_total || col >= n_out) return;

    int token = sorted_token_ids[row];
    float weight = sorted_weights[row];
    float val = expert_outputs[row * n_out + col];

    // Weighted scatter-add into output
    atomicAdd(&output[token * n_out + col], weight * val);
}
