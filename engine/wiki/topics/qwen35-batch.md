# qwen35-batch (Continuous Batching Prototype)

## Purpose
[coverage: high — 4 sources]
Standalone crate (`qwen35-batch/`) — **continuous batching prototype for Qwen3.5-4B** on Candle. Implements N parallel decode slots over a shared weight copy, exposing `BatchModel` trait + `BatchScheduler`. Goal: empirically test hypothesis (lib.rs:18-20) of ×1.15–1.35 aggregate decode tok/s gain at B=4 vs sequential on bandwidth-bound Apple Silicon. Status: scheduler logic fully tested on mock (12 tests green); real-model adapter ported; **hypothesis NOT confirmed** — see Gotchas. Tests: `cargo test -p qwen35-batch` (mock) → 12 passed; real-model tests require feature flag + GGUF.

## Architecture
[coverage: high — 4 sources]
Three-module split (lib.rs:22-32):

**`model.rs` — `BatchModel` trait** (model.rs:100-116): `vocab_size()`, `prefill_chunk(&PrefillChunk) -> logits[last token]`, `decode_batch(&DecodeBatch) -> Vec<logits per slot>`, `reset_slot(idx)`. `DecodeItem { slot_idx, token, pos }` / `DecodeBatch { items: Vec<DecodeItem> }` (model.rs:28-52). `Sampler` trait (model.rs:68-78) with `sample()` + `sample_indexed(slot, generated, logits)` for per-request params/penalties (qwen36-server prep). `GreedySampler` = argmax (model.rs:82-97). `MockRecurrentModel` (model.rs:131-199) — deterministic pseudo-recurrent: `mix(state, token)` PCG-style (model.rs:155-160), `next_token` (model.rs:164-170); per-slot state `Vec<u64>`, reset to 0 on admit, evolves only from own tokens ⇒ output is pure function of prompt (parity invariant).

**`slot.rs` — `Slot` lifecycle** (slot.rs:12-22): `Idle → Prefilling → Decoding → Finished → Idle`. `Slot { idx, status, request, prefill_done, generated, index_pos }` (slot.rs:50-64). `admit(req)` resets then → Prefilling (slot.rs:88-93). `advance_prefill(n)` grows `prefill_done`/`index_pos`, transitions to Decoding when prompt exhausted (slot.rs:109-120). `current_token()` = last generated or last prompt token (slot.rs:126-134). `next_pos()` = `index_pos` for RoPE/KV (slot.rs:137-139). `push_token(tok)` (slot.rs:142-153): appends, increments `index_pos`, returns `done` (EOS or `max_new` reached) → Finished.

**`scheduler.rs` — `BatchScheduler<M: BatchModel>`** (scheduler.rs:67-74): holds `model`, `slots: Vec<Slot>`, `queue: VecDeque<SlotRequest>`, `sampler: Box<dyn Sampler>`, `stats: SchedulerStats`, `eos`. `step()` → `step_with(should_stop)` (scheduler.rs:122-134). Each step:
1. **Prefill phase** (scheduler.rs:137-170): pick ONE Prefilling slot, feed one chunk (`PREFILL_CHUNK = usize::MAX` = whole prompt, scheduler.rs:65), call `model.prefill_chunk`, advance; if prompt done → sample first token from last-prompt-token logits (`first_token_comitted`), push; `should_stop` may finish early.
2. **Decode phase** (scheduler.rs:172-199): collect ALL Decoding slots into one `DecodeBatch`, `model.decode_batch` once, sample per slot, push, check stop/max/EOS → Finished.

`SchedulerStats` (scheduler.rs:32-48): `decode_steps`, `total_decode_tokens`, `max_concurrent_decode`, `prefill_chunks`, `wall_ns`/`prefill_ns`/`decode_ns`, `decode_aggregate_tps()` (scheduler.rs:52-59). `StepOutcome`: `Idle | DidPrefill { first_token_emitted } | DidDecode(batch_size)` (scheduler.rs:77-85). `submit()` (scheduler.rs:100-112): admit if idle slot, else queue. `run_with_collection` + `sequential_reference` (baseline) for parity tests.

**`real/` adapter** (`#[cfg(feature = "real-model")]`, lib.rs:26-28): `Qwen35BatchAdapter` (adapter.rs:27-38) wraps `ModelWeights` + per-slot `slot_snaps: Vec<Option<StateSnapshot>>` + `slot_seeded: Vec<bool>`. `load(gguf, device, num_slots)` (adapter.rs:42-92): mmap GGUF, read EOS + vocab from `token_embd.weight` shape[0], `from_gguf_zero_copy` on Metal (adapter.rs:73-79) else `from_gguf`. `prefill_chunk` (adapter.rs:144-188): reset/restore single-slot state, `forward([1, seq], start_pos)`, snapshot at `start_pos + tokens.len()`, mark slot not-seeded. `decode_batch` (adapter.rs:190+): seeds batched buffers per slot then true batched `forward_decode_batch`.

## Talks To
[coverage: medium — 3 sources]
- **`candle-core`** (path `../candle-core`) — `Tensor`, `Device`, `DType`, `quantized::gguf_file`.
- **`candle-nn`** (path `../candle-nn`) — layers used by `ModelWeights`.
- **`candle-metal-kernels`** (macOS-only, path `../candle-metal-kernels`, adapter.rs / REAL_MODEL.md) — Q4K matmul V1-V4, SDPA, delta_rule metal kernels.
- **Yttri app** (external) — source of ported code (`frontend/src-tauri/src/modules/ai/local_llm/quantized_qwen35.rs` 4854 lines, `metal/delta_rule_metal.rs` 604, `metal/gated_delta_net_fused.rs` + `.metal` 399; REAL_MODEL.md:115-130). GGUF model file lives in Yttri resources (REAL_MODEL.md:129).

## API Surface
[coverage: high — 3 sources]
`pub use`: `model::{BatchModel, DecodeBatch, PrefillChunk}` (lib.rs:30), `scheduler::{BatchScheduler, SchedulerStats}` (lib.rs:31), `slot::{Slot, SlotStatus}` (lib.rs:32). `pub const DEFAULT_NUM_SLOTS: usize = 4` (lib.rs:34). `pub trait Sampler` + `GreedySampler` (model.rs:68-97). `pub fn mix(state, token) -> u64` + `pub fn next_token(token, state, vocab) -> u32` (model.rs:155-170) — mock primitives, also usable by external parity tests. Features (qwen35-batch/Cargo.toml:31-41): `real-model`, `metal`, `cuda`. `#[test]` targets: `scheduler_parity`, `real_qwen35_batch`, `real_qwen35_quality` (latter two require `real-model`).

## Data
[coverage: high — 4 sources]
- **Weights**: one shared `ModelWeights` (Q4_K_M quantized Qwen3.5-4B, ~2.7 GB GGUF, vocab=248320 padded to 256 for quant, eos=248046; REAL_MODEL.md:74-78). Zero-copy Metal mmap (~2583 MB in MTLResidencySet).
- **Per-slot state** (adapter.rs:30-38 + REAL_MODEL.md:55-62): `slot_snaps: Vec<Option<StateSnapshot>>` — each ~114 MB (24 DeltaNet layers' ssm_state + conv_state fp32 + 8 attention KV-caches). Snapshot/restore via `ModelWeights::snapshot_state`/`restore_state` (T-274 prompt-cache API).
- **Scheduler scratch** (mock): `MockRecurrentModel.states: Vec<u64>` — O(N) u64s.
- No persistence; all in-memory per run.

## Key Decisions
[coverage: high — 4 sources]
- **First token sampled from prefill logits, not re-forwarded** (scheduler.rs:9-12, REAL_MODEL.md:192-194): re-forwarding last prompt token would corrupt recurrent GDN conv/SSM state. Verified by `first_token_comes_from_prefill_logits`.
- **`PREFILL_CHUNK = usize::MAX`** (scheduler.rs:62-65): whole-prompt prefill. Chunked prefill needs recurrent-state checkpoints (complexity out of scope).
- **Time-multiplexing as fallback** (REAL_MODEL.md:54-62, adapter.rs:15-17): true batched decode impossible without Metal-kernel rewrite (see Gotchas); adapter restores→forward→snapshot per slot. Bit-exact parity (each slot isolated by own state).
- **Per-slot state over per-request** (lib.rs:5-12): stable `slot.idx` addresses state in model; reset on admit via `PrefillChunk::reset_first`.
- **fp32 recurrent state** (slot.rs:3-6): f16 drifts at 5K+ tokens per Yttri data.
- **`sample_indexed(slot, generated, logits)`** (model.rs:71-78): hook for per-request sampling params + presence/repetition penalties (qwen36-server); defaults to `sample()` for parity tests.

## Gotchas
[coverage: high — 4 sources]
**Architectural wall (§0c, REAL_MODEL.md:41-67)**: True batched decode on current Candle **impossible** without rewriting Metal kernels:
- `DeltaNetLayer::forward` (model_weights.rs:1469) asserts `seq_len==1` AND `b_sz==1` (line 1711) — decode strictly 1 token.
- `dispatch_delta_rule` (delta_rule_metal.rs): grid per v-head, F32 per-token scratch (`DeltaNetTempBuffers` fixed `[channels]`/`[n_v*hd]`); batch axis absent in 4 kernels (conv1d_prep, l2_norm_expand, delta_rule, norm_gate).
- `dispatch_gdn_fused` prefill is batch-aware (`n_tokens`) but decode path bypasses it.
- Q4K fast-path (`dispatch_q4k_matmul:193`): requires `n%64==0 && m%32==0`; B=4 → m=4 → falls back to slower-but-correct V1.

**Measured bench (2026-08-01, REAL_MODEL.md:69-111)** — hypothesis NOT confirmed:
| B | wall_ms | aggregate tok/s | per-req tok/s | decode-only tok/s | vs B=1 |
|---|---------|----------------|--------------|-------------------|--------|
| 1 | 2145.6  | 29.83          | 29.83        | 34.54             | 1.000× |
| 2 | 2012.6  | 31.80          | 15.90        | 36.22             | 1.066× |
| 4 | 2012.8  | 31.80          | 7.95         | 36.06             | 1.066× |

Aggregate flat at ×1.07 (saving from 4 prefills, not decode). Decode-only tok/s flat (34.5/36.2/36.1): B=4 = 4 separate forwards + restore/snapshot per slot, per-step 110.9 ms (×3.84 vs B=1 28.9 ms — linear in slots, no parallelism). Per-request degrades linearly (1/B). Parity bit-exact.

**Conclusion (REAL_MODEL.md:64-67, 106-111)**: continuous batching on Candle for Qwen3.5 = major kernel rewrite (4 delta_rule + fused GDN + Q4K fast-path for m=B), comparable to MLX ecosystem's `BatchedHybridCache`. Alternative = MLX sidecar (Yttri already uses mlx-swift, hybrid batching shipped).

## Sources
- [qwen35-batch/src/lib.rs](../../qwen35-batch/src/lib.rs)
- [qwen35-batch/src/model.rs](../../qwen35-batch/src/model.rs)
- [qwen35-batch/src/scheduler.rs](../../qwen35-batch/src/scheduler.rs)
- [qwen35-batch/src/slot.rs](../../qwen35-batch/src/slot.rs)
- [qwen35-batch/src/real/adapter.rs](../../qwen35-batch/src/real/adapter.rs)
- [qwen35-batch/REAL_MODEL.md](../../qwen35-batch/REAL_MODEL.md)
- [qwen35-batch/Cargo.toml](../../qwen35-batch/Cargo.toml)
