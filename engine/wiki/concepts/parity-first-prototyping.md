---
concept: Parity-First Prototyping
last_compiled: 2026-08-08
topics_connected: [qwen35-batch, project-overview, infrastructure]
status: active
---

# Parity-First Prototyping

## Pattern
Before touching real weights, every new scheduler/model abstraction in this fork is first proven on a **deterministic mock** whose output is a pure function of its prompt. The mock is designed so that batched-vs-sequential runs *must* produce identical token sequences if (and only if) per-slot state is correctly isolated. Only when parity holds on the mock does work proceed to the real GGUF adapter — and the real adapter inherits the same `BatchModel` trait, so the scheduler code is untouched.

This treats "correctness" as a property of the abstraction, independent of the backend, and makes the empirical negative result (batching doesn't speed up decode) trustworthy — you know the absence of speedup isn't a correctness bug papering over real parallelism.

## Instances
- **`MockRecurrentModel`** (qwen35-batch `model.rs:131-199`) — deterministic pseudo-recurrent: `mix(state, token)` PCG-style update, `next_token` deterministic from (token, state). Per-slot `Vec<u64>` state resets to 0 on admit and evolves only from that slot's own tokens ⇒ output is a pure function of the prompt, independent of slot index and presence of other slots. Documented in [qwen35-batch](../topics/qwen35-batch.md) §Architecture.
- **12 green tests on mock** (REAL_MODEL.md:9-32) before any real-model code: `slot_lifecycle`, `slot_eos_terminates`, `mock_output_is_pure_function_of_prompt`, `mock_concurrent_slots_isolated`, `batched_equals_sequential_parity`, `batched_more_requests_than_slots_recycles`, `stats_recorded`, `first_token_comes_from_prefill_logits`, + `scheduler_parity.rs` integration tests. `cargo test -p qwen35-batch` = 12 passed, 0 failed.
- **Bit-exact parity preserved into real model** (REAL_MODEL.md:77-80, 104): `real_qwen35_batched_equals_sequential_parity` — B=2 and B=4 vs B=1 produce identical token sequences. The time-multiplexed snapshot/restore isolates each slot's state, so the same correctness invariant carries over.
- **`sequential_reference` baseline** (scheduler.rs, [qwen35-batch](../topics/qwen35-batch.md) §Architecture) — scheduler ships its own reference path: run the same prompts one-at-a-time through single-slot scheduler, compare. Greedy determinism ⇒ same logits ⇒ same tokens ⇒ batched == sequential is a theorem, not just an observation.
- **Greedy sampler chosen for parity** (model.rs:80-97, [qwen35-batch](../topics/qwen35-batch.md) §Key Decisions) — argmax is deterministic, so any divergence between batched and sequential is a state-isolation bug, not sampling noise. The `Sampler` trait still has `sample_indexed(slot, generated, logits)` as a forward-looking hook for qwen36-server per-request params, but parity tests use greedy.

## What This Means
The discipline pays off exactly when the result is negative. The real-model bench (REAL_MODEL.md:69-111) showed aggregate tok/s flat at ×1.07 and per-request tok/s degrading as 1/B — i.e., batching gave no real parallelism. Because parity was proven bit-exact first, you can trust that the flat numbers reflect a genuine kernel limitation (the [batch-axis wall](./batch-axis-wall.md)), not a scheduler bug that happens to serialize work.

The pattern also shapes the API: `BatchModel` is a trait, not a struct, and the scheduler is generic `BatchScheduler<M: BatchModel>`. That let the real adapter (`Qwen35BatchAdapter`) drop in behind the same trait the mock implemented, with zero scheduler changes — the abstraction boundary was chosen to make "test on mock, then swap backend" free.

For future work, this implies the next experiment (e.g., an MLX sidecar, or rewritten Metal kernels) should preserve the same `BatchModel` trait and the same `sequential_reference` parity test. If a new backend can't pass bit-exact batched-vs-sequential on greedy, it isn't ready to be benchmarked for speedup — the speedup numbers would be meaningless without the correctness floor.

## Sources
- [../topics/qwen35-batch](../topics/qwen35-batch)
- [../topics/project-overview](../topics/project-overview)
- [../topics/infrastructure](../topics/infrastructure)
