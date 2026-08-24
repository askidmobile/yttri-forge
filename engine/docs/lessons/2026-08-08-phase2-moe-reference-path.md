# Phase 2: Qwen3.6 35B-A3B MoE reference path

Date: 2026-08-08
Commit: `1d443b85`
Plan: `bee2ee66` (Phase 2)
Status: complete

## What was built

### moe.rs (375 lines)

Qwen3.6 MoE block with three components:

1. **MoeRouter** — `route_topk(&xs [n_tokens, n_embd]) -> RoutePlan`. Flow: `linear.forward` → `.to_dtype(F32)` → `softmax_last_dim` → CPU `to_vec2` → manual iterative argmax (strict `>`, tie-break lower index) → `norm_topk_prob` normalization. Avoids `arg_sort_last_dim` (unstable in candle). Router weight dequantized to F32 at load (small tensor).

2. **PackedExperts** — `Arc<QTensor>` for gate/up/down. No dequant at load. GGUF shapes: gate `[n_experts, n_ff, n_embd]`, up same, down `[n_experts, n_embd, n_ff]`.

3. **SharedExpert** — `gate_inp: Linear` → `silu_div` (bit-exact sigmoid, 0→0.5) → `gate.forward × up.forward` → `silu × mul` → `down.forward` → `broadcast_mul(gate_scalar [n_tokens,1])`. `silu_div` does CPU round-trip (tensor is tiny `[n_tokens, 1]`).

4. **reference_routed_swiglu** — dequantize packed → group tokens by expert → `index_select` + `matmul` per expert → `broadcast_mul(weights)` → `index_add` scatter.

### model_weights.rs integration

- `Mlp` → `DenseMlp` (rename, `#[derive(Debug, Clone)]` preserved).
- `FeedForward` enum: `Dense(DenseMlp) | Moe(Qwen35MoeBlock)`. No `Debug` derive (Qwen35MoeBlock doesn't impl Debug).
- `HybridBlock.mlp: Mlp` → `HybridBlock.ff: FeedForward`.
- All 3 forward paths updated: `forward`, `forward_prefill`, `forward_decode_batch` call `self.ff.forward(&normed)` / `forward_prefill` / `forward_decode_batch`.
- `build_model_common` architecture-aware: `is_moe = metadata["general.architecture"] == "qwen35moe"`. Prefix = `qwen35moe` vs `qwen35`. All metadata/tensor reads use prefix. Arena sizing uses `max(routed, shared)` intermediate for MoE.
- MoE tensor loading: router from `ffn_gate_inp.weight` (dequant → F32 Linear), packed from `ffn_gate_exps/up_exps/down_exps.weight` (Arc<QTensor>), shared from `ffn_gate_inp_shexp` + `ffn_gate_shexp/up_shexp/down_shexp`.
- Historical Phase 2 state: `QWEN36_MOE_BACKEND=ptx|reference` defaulted to Reference and PTX bailed. Superseded by Phase 3: supported CUDA routed dtypes default to PTX; `reference` is explicit rollback. See `2026-08-13-iq-moe-cuda.md`.
- `debug_capture_single_matmul` updated: `match &block0.ff { Dense(m) => m, Moe(_) => bail! }`.

### tests/qwen35moe_reference.rs (667 lines, 11 tests)

Synthetic experts on CPU (device=Cpu, no CUDA/Metal needed). TOL=1e-5 (F32).

- `test_identity_routed_expert_silu_mul` — gate=up=down=I → output = silu(x)·x
- `test_shared_gate_half` — gate_inp=0 → silu_div(0)=0.5 → 0.5·silu(x)·x
- `test_shared_gate_one` — gate_inp=+100 → silu_div→1.0 → silu(x)·x
- `test_shared_gate_zero` — gate_inp=-100 → silu_div→0 → routed only
- `test_composition_routed_plus_shared` — top-2 uniform (0.5+0.5) + shared 0.5 → 1.5·silu(x)·x
- `test_router_topk_selection` — known router weights → expected experts + norm weights sum to 1
- `test_router_tie_break_lower_index` — equal logits → lower index wins
- `test_chunked_vs_unchunked_tolerance` — Prefill == DecodeBatch (TOL)
- `test_output_shape` — [batch, seq, n_embd] preserved
- `test_no_norm_topk_prob` — raw softmax prob (not normalized)
- `test_expert_slice_helper` — narrow/squeeze produces correct 2D slice

## Errors encountered and fixed

1. **E0308: f64 vs f32** — `TOL: f64` compared against `(got - exp).abs()` where `got`/`exp` are `f32`. Rust does not promote f32→f64 in comparisons. Fix: `TOL: f32 = 1e-5`.
2. **E0061: zeros_flat(1 arg)** — helper `zeros_flat(rows, cols)` called as `zeros_flat(total)`. Fix: removed helper, used `vec![0.0f32; total]` inline.
3. **Unused imports** — `Module as _`, `std::sync::Arc` not needed. Fix: removed.
4. **Unused variables** — `expert_slice(packed, e, rows, cols)` didn't use rows/cols. Fix: `_rows`, `_cols`.
5. **IndexP→IndexOp** — pre-existing compile error in model_weights.rs caught during integration. Fix: rename.

## Validation

- `cargo check --tests --features real-model,metal` (macOS): 0 errors, pre-existing warnings only.
- `cargo test -p qwen35-batch --test qwen35moe_reference --features real-model` (yttri-win): 11/11 pass, 0.01s, CPU-only.

## Plan decisions recorded

- PD-005: FeedForward enum no Debug derive (Qwen35MoeBlock constraint).
- PD-006: IndexP→IndexOp fix.
- PD-007: build_model_common architecture-aware (qwen35moe vs qwen35 prefix).
- PD-008: Router/shared gate_inp dequantized to F32 at load; packed experts stay Arc<QTensor>.
- PD-009: f32_qtensor helper uses QTensor::quantize_onto with F32 dtype.

## Deferred

- Task 5 (layer capture points) → Phase 8 (diagnostics scope).

## Next

Phase 3 — PTX MoE backend (IQ2_XXS/IQ2_M/Q2_K/IQ3_XXS/Q4_K sparse kernels, 60-90 hours).