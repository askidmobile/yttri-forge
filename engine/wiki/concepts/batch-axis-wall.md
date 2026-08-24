---
concept: Batch-Axis Wall in Candle Metal Kernels
last_compiled: 2026-08-08
topics_connected: [qwen35-batch, candle-core, project-overview]
status: active
---

# Batch-Axis Wall in Candle Metal Kernels

## Pattern
Candle's Metal (and CUDA) kernels for recurrent + quantized ops are written **per-token** (batch axis absent or asserted to 1). Any attempt at true continuous batching — running B decode slots through one `forward` call — hits `debug_assert_eq!(seq_len, 1)` / `debug_assert_eq!(b_sz, 1)` or falls back to a slower non-batched path. The framework can *express* a `[B, 1]` tensor, but the kernels don't carry the batch axis through the recurrent state math, so per-slot state must be swapped serially (time-multiplexing) rather than processed in parallel.

This is the same class of limitation MLX's ecosystem resolved with `BatchedHybridCache` — the kernels must be rewritten with an explicit slot axis before batched decode yields real throughput gains.

## Instances
- **DeltaNet decode asserts single token** in `qwen35-batch` `ModelWeights::forward` (ported from Yttri `quantized_qwen35.rs:1469`, assert at `:1711`): `debug_assert_eq!(seq_len, 1)` + `debug_assert_eq!(b_sz, 1)`. Documented in [qwen35-batch](../topics/qwen35-batch.md) §Gotchas and REAL_MODEL.md:43-50.
- **`dispatch_delta_rule`** 4 Metal kernels (`conv1d_prep`, `l2_norm_expand`, `delta_rule`, `norm_gate`) have grid per v-head and fixed F32 per-token scratch (`DeltaNetTempBuffers` sized `[channels]`/`[n_v*hd]`) — no batch axis. REAL_MODEL.md:46-48.
- **`dispatch_gdn_fused`** (prefill) IS batch-aware (`n_tokens`) but the decode path never routes through it — prefill batching works, decode batching doesn't. REAL_MODEL.md:49-50.
- **Q4K fast-path** `dispatch_q4k_matmul` requires `n%64==0 && m%32==0`; for decode `xs=[1,B,K]` → `m=B`. With B<32 (e.g. B=4) it falls back to slower-but-correct V1. REAL_MODEL.md:51-52.
- **candle-core** ships `candle-kernels` (CUDA, partly from [dfdx](https://github.com/coreylowman/dfdx)) and `candle-metal-kernels` — both are 1-line-README "kernels used from candle" with no batch-axis documentation, confirming the per-token design assumption framework-wide. See [candle-core](../topics/candle-core.md).

## What This Means
The fork's central negative result: **continuous batching on Candle for Qwen3.5 is not a scheduler problem, it's a kernel problem.** The `BatchScheduler` + `BatchModel` abstraction (qwen35-batch) is correct and bit-exact-parity-proven, but the time-multiplexed fallback yields only ×1.07 aggregate tok/s (vs hypothesized ×1.15–1.35) because every "batched" decode step is really B sequential `forward` calls plus ~114 MB/slot snapshot/restore I/O. Per-request tok/s degrades as ~1/B — no parallelism, only interleaving with overhead.

Concretely, to get true ×1.15–1.35 (llama.cpp B=8 territory), four delta_rule Metal kernels + the fused GDN kernel + the Q4K fast-path must all grow a slot axis. That work is comparable to what the MLX ecosystem already shipped (`BatchedHybridCache`). The alternative path is an **MLX sidecar** (Yttri already uses `mlx-swift`) — defer batching to a runtime that already has the batched kernels, and use Candle for single-stream or prefill where its kernels are fine.

This pattern also explains why the prototype was built as a standalone fork rather than a patch to upstream candle: the limitation is in `candle-metal-kernels` (excluded from the workspace), and fixing it upstream would be a multi-kernel rewrite — out of scope for a batching prototype whose job was to *measure* the wall, not remove it.

## Sources
- [../topics/qwen35-batch](../topics/qwen35-batch)
- [../topics/candle-core](../topics/candle-core)
- [../topics/project-overview](../topics/project-overview)
