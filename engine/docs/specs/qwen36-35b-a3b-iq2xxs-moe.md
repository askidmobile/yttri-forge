# Design: Qwen3.6 35B-A3B IQ2_XXS MoE Inference

## Context

The target GGUF is a Qwen3.6/Qwen3.5-family hybrid model: 40 trunk blocks arranged as ten repetitions of three DeltaNet blocks and one gated-attention block. Every trunk block uses MoE with 256 packed routed experts, top-8 routing, a routed intermediate width of 512, and one always-active shared expert with its own sigmoid gate. The existing runtime already implements the hybrid sequence model, chunked prefill, true batched decode, per-slot DeltaNet/KV state, and prompt-cache snapshots for dense models.

Existing `FusedMoeGGUF` provides router softmax, top-k selection, normalization, token/expert sorting, and packed routed tensors, but it assumes a conventional Qwen3 transformer, omits the shared expert computation, and calls CUDA functions linked from an excluded static MoE library. Existing GGUF MoE kernels support Q8_0 and K-quants only; `IQ2_XXS` currently has general CUDA dequantization but no sparse fused MoE matmul.

The target RTX 3060 has 12 GB VRAM. The approximately 10.8 GB GGUF leaves little headroom, so full or persistent dequantization of packed expert tensors is invalid. CPU offload may be retained as an explicit degraded/debug mode, but the supported performance path must keep quantized expert weights on CUDA and operate only on routed experts.

## Goals

- Load and run the 40-layer autoregressive trunk of `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` on CUDA.
- Preserve the proven Qwen3.5/Qwen3.6 DeltaNet, attention, prefill, batching, and state-management implementations.
- Compute routed top-8 and shared-expert contributions according to the model architecture.
- Provide a small, deterministic reference path for kernel parity and diagnosis.
- Use runtime-loaded CUDA kernels compatible with Windows dynamic CUDA loading.
- Bound transient memory so the model can run on an RTX 3060 12 GB.

## Non-Goals

- Vision encoder/projector support.
- Training, fine-tuning, expert parallelism, or multi-GPU sharding.
- Permanent F16/F32 copies of all packed experts.
- MTP/speculative decoding in the initial trunk milestone.
- General optimization of DeltaNet prefill beyond regressions required to run this model.
- Treating the older conventional `quantized_qwen3_moe` model as the execution engine for the hybrid architecture.

## Architecture Validation and Loader

The loader reads `general.architecture` first and dispatches dense `qwen35` and MoE `qwen35moe` configurations explicitly. It validates required metadata rather than silently accepting incompatible defaults. For the target profile, validation covers 40 trunk blocks, hidden size 2048, 16 attention Q heads, 2 KV heads, attention head dimension 256, partial RoPE dimension 64, RoPE base 10,000,000, DeltaNet head/config dimensions, full-attention interval 4, 256 experts, top-k 8, and routed/shared intermediate width 512. Values may be generalized where the current implementation already supports them, but unsupported layouts fail with the metadata key and observed value.

Each MoE block loads router and shared-gate weights as ordinary quantized matmuls and routed expert weights as packed `Arc<QTensor>` values:

- `blk.N.ffn_gate_inp.weight`
- `blk.N.ffn_gate_exps.weight`
- `blk.N.ffn_up_exps.weight`
- `blk.N.ffn_down_exps.weight`
- `blk.N.ffn_gate_shexp.weight`
- `blk.N.ffn_up_shexp.weight`
- `blk.N.ffn_down_shexp.weight`
- `blk.N.ffn_gate_inp_shexp.weight`

The loader accepts trunk tensors `blk.0` through `blk.39`. Additional `blk.40`/`nextn` tensors are reported as an optional MTP head and ignored by the trunk-only milestone. Missing required trunk or shared-expert tensors are fatal.

## Hybrid Block Integration

`HybridBlock` changes from a fixed dense `Mlp` field to a feed-forward enum with dense and MoE variants. Attention/DeltaNet execution and residual ordering remain unchanged:

1. Apply attention norm.
2. Run DeltaNet or gated attention and add the residual.
3. Apply FFN norm.
4. Run dense MLP or MoE and add the residual.

The same feed-forward dispatch is used by single-stream prefill/decode and true batched decode. MoE has no recurrent state, so snapshots continue to contain only DeltaNet and attention state. Snapshot identity/config guards add architecture, block layout, expert count, top-k, and shared-expert configuration to prevent restoring a state into an incompatible model.

## Routing and Shared Expert Semantics

For a normalized input flattened to `[tokens, hidden]`:

1. Compute router logits in F32.
2. Apply softmax over 256 routed experts.
3. Select the largest eight expert probabilities per token.
4. Normalize selected probabilities by their sum when required by Qwen3.6 metadata/config.
5. Flatten and sort token/expert pairs by expert for grouped execution while retaining the original pair index used for weights and output scatter.
6. Run routed gate/up/down projections only for selected experts and sum the weighted top-8 outputs per token.
7. Independently run the shared expert SwiGLU on every token.
8. Compute `sigmoid(ffn_gate_inp_shexp(x))` and multiply it with the shared expert output.
9. Add routed and gated shared contributions, then reshape to the original batch/sequence dimensions.

Routing results are deterministic for parity tests. The sorting/scatter contract is shared by reference and optimized backends to avoid semantic drift.

## Reference Execution

A correctness backend gathers one expert at a time from packed quantized storage, dequantizes only the rows required by the selected expert projection, performs the matmul, and releases/reuses bounded scratch before the next expert. It never dequantizes all 256 experts simultaneously. This path supports tiny synthetic fixtures and short real-model probes and is enabled explicitly for tests/diagnostics; it is not the production performance target.

Reference validation compares:

- router top-k IDs and normalized weights;
- each gate/up/down projection;
- routed reduction, sigmoid-gated shared output, and combined FFN output;
- complete block and trunk logits for prefill and decode.

## CUDA Execution

### Build and loading

MoE kernels are compiled into runtime-loadable PTX/module bindings by `candle-kernels/build.rs`, following the same `CudaDevice::get_or_load_func` pattern as quantized and DeltaNet kernels. The implementation removes runtime dependence on the `extern "C"` MoE symbols for the new path. `cuda_moe` remains explicit and is propagated from server/runtime crates through `qwen35-batch`, `candle-nn`, and kernel availability checks. A build without `cuda_moe` gives an actionable load-time error for `qwen35moe` rather than failing at link time or later inside inference.

### IQ2_XXS sparse GEMM

The packed layout is `[expert, output_row, input_blocks]`, with `QK_K=256` and the repository's verified `block_iq2_xxs` representation. The CUDA path adds `IQ2_XXS` datatype dispatch and a device implementation that dequantizes quant blocks into registers/shared memory while accumulating against routed activations. It must not create a full dense expert tensor.

Two execution shapes are required:

- Decode/small M: direct sparse quantized matvec/matmul over selected token/expert pairs, minimizing launch overhead.
- Prefill/batched M: launch an expert grid, scan bounded route tiles on device, and process matching routes in groups so one packed weight read feeds multiple token accumulators. No global route sort or offset workspace is required.

Gate and up projections emit `[token, top_k, 512]` intermediates. After SiLU multiplication, down projection applies corresponding top-k weight and accumulates/scatters to `[token, 2048]`. Input stays shared as `[token, 1, hidden]`; kernels reuse it across top-k routes. Grouped kernel uses bounded static shared task storage. Short-lived outputs remain stream-ordered CUDA allocations; no persistent per-layer workspace may consume RTX 3060 headroom.

Kernel launch APIs return errors for unsupported dtype/shape rather than silently producing an uninitialized output. CUDA launch errors are checked at the backend boundary and surfaced with projection, layer, dtype, and shape context.

## Prefill and Continuous Batching

Chunked prefill passes all tokens in a chunk through the MoE backend together, preserving the existing sequential DeltaNet state update and attention KV accumulation. Route grouping is local to each FFN invocation and does not persist across chunks.

True batched decode flattens `[B, 1, hidden]` into tokens, routes all active slots in one invocation per projection/layer, and scatters results back in batch order. Slot compaction after EOS affects only the batch-to-slot mapping already used by DeltaNet/KV state; MoE creates no slot state. Time-multiplexed fallback remains supported when the existing batched DeltaNet backend is unavailable.

Prompt-cache snapshots remain valid because MoE is stateless. Cache keys/identity include the model architecture and expert configuration, and cache restore is tested against an uncached run.

## Memory Strategy

- Keep packed `IQ2_XXS` routed experts quantized on CUDA.
- Do not retain full dequantized expert matrices between calls.
- Avoid global routing scratch: grouped prefill uses bounded static shared task tiles. Keep short-lived activation/output allocation stream-ordered; do not retain one large workspace per layer.
- Measure free/used VRAM before load, after load, after first prefill, and during steady decode.
- Fail model initialization with an actionable memory estimate when required weights plus reserved runtime headroom cannot fit.
- Treat managed-memory paging or CPU expert offload as a degraded mode, never as evidence that the 12 GB CUDA target passed.

The acceptance run uses conservative context and slot settings first, then records the supported matrix of context length, prefill chunk, and slot count. No universal large-context claim is made merely because GGUF advertises 262,144 tokens.

## Validation Strategy

1. Synthetic CPU/reference tests for routing, normalization, shared gating, packed shape/stride, and error handling.
2. CUDA `IQ2_XXS` projection tests across expert IDs, output rows, multiple quant blocks, odd token distributions, empty experts, top-8 weighting, and decode/prefill shapes.
3. Pinned llama.cpp-versus-CUDA MoE and full-block parity with fixed teacher-forced states and explicit absolute/relative tolerances derived from measured F32 accumulation and IQ quantization. Exact greedy is diagnostic for low-margin 2-bit decisions; high-margin argmax drift remains a failure.
4. Single-stream prefill/decode parity, chunk-boundary parity, snapshot restore parity, prompt-cache parity, true batched decode parity, slot isolation, and EOS compaction tests.
5. Real GGUF smoke generation with fixed prompt/seed, finite logits, valid tokens, EOS handling, and qualitative RU/EN prompts.
6. Windows CUDA 12.4 release build, dependency inspection, kernel load smoke test, and server `/v1/chat/completions` end-to-end request.
7. VRAM peak and latency measurements on RTX 3060 12 GB; compare optimized sparse execution against the reference path and reject implementations that materialize all experts or regress to per-expert host synchronization.

Performance results are reported separately for load, prefill tokens/s, first-token latency, decode tokens/s at batch sizes 1 and multiple active slots, and peak VRAM. Correctness gates are mandatory; target throughput is benchmark-driven and must not be fabricated before the kernels exist.

## Migration and Rollback

Dense `qwen35` loading and execution remain unchanged behind the dense feed-forward variant. `qwen35moe` is accepted only when the MoE feature/backend is present. The optimized backend can be disabled with a diagnostic setting to select the reference path for bisecting. Rollback consists of disabling MoE model registration/feature propagation while retaining dense and general `IQ2_XXS` support.

## Optional MTP Phase

After trunk parity and stability, a separate change may load `blk.40`/`nextn` tensors and implement speculative draft/verification. That phase must define tokenizer alignment, draft state/cache handling, acceptance logic, memory cost, and its own parity/performance gates. Trunk-only GGUF files and baseline autoregressive inference remain supported independently.

## Risks / Trade-offs

- Runtime PTX kernels may be slower than a heavily specialized static library initially. This is accepted to preserve Windows dynamic loading and reliable builds; optimize after profiling.
- Router sort may dominate batch-1 decode. Use a small-M path and benchmark thresholds instead of assuming prefill grouping is always faster.
- A 10.8 GB file on 12 GB VRAM leaves narrow headroom. Enforce bounded scratch and publish the validated context/slot envelope.
- Shared-expert omission can produce plausible but incorrect text. It is a mandatory parity component, not an optimization.
- Existing `quantized_qwen3_moe` code contains reusable pieces but also architecture-specific assumptions; extract/reuse routing/backend pieces rather than routing the hybrid model through that transformer.
