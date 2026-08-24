# Qwen3.5 dense CUDA: Yttri comparison and VRAM diagnosis

Date: 2026-08-13

GPU: RTX 3060 12 GB, Windows WDDM, CUDA 12.4, SM86

Models: Qwen3.5-4B/9B Q4_K_M

Reference: llama.cpp `8e7f22b67ef4667b4ddd50230771287f328cfb3f`

## Result

Observed 12 GiB VRAM saturation was benchmark-harness error, not dense runtime allocation. Two ignored tests ran concurrently and each loaded adapters; one test also loaded separate B=1/B=2/B=4 adapters. WDDM paged duplicate 9B weights into shared system memory.

Benchmark now uses one `Qwen35BatchAdapter` with capacity 4. B changes only through active prompt count. Test command must include `--ignored --exact --nocapture --test-threads=1`.

Permanent guard rejects:

- model VRAM delta greater than GGUF size + 1024 MiB;
- B=1 to B=4 growth greater than 256 MiB.

## Isolated memory

Clean GPU baseline: 453 MiB used.

| Model | B=1 total | B=4 total | Runtime delta | Post-exit |
|---|---:|---:|---:|---:|
| 4B | 3448 MiB | 3480 MiB | 2880–2912 MiB | 453 MiB |
| 9B | 5688 MiB | 5720 MiB | 5120–5152 MiB | 453 MiB |

Fresh-server WDDM telemetry stayed resident:

| Model | Loaded dedicated | After B=4 dedicated | Shared |
|---|---:|---:|---:|
| 4B | 3002.9 MiB | 3034.9 MiB | 76.0 MiB |
| 9B | 5210.9 MiB | 5274.9 MiB | 76.0 MiB |

No runtime paging found.

## Yttri prefill comparison

Yttri solved two relevant problems:

1. CUDA prefill attention drift: commit `952bb5d18` moved manual CUDA attention from F16 to F32. Current fork already contains this fallback and defaults to FlashAttention 2 with F32 accumulation (`ec13b1e4`).
2. Repeated prompts: commits `70ab76a30` and `bd7e4b230` place recurrent snapshots at observed common-prefix boundaries. This accelerates repeated shared prefixes, not first unique prompt prefill. Server prefix cache remains disabled until its final-logits/restore contract is corrected.

Yttri CUDA Gated DeltaNet still used a Rust-side per-token loop. Current fork is newer: fused sequence kernels move token loop into CUDA and keep recurrent state in registers (`0b3b5e2f`, `048463cc`). Porting Yttri compute path would regress performance.

Comparable cold 2191-token 4B prefill:

| Path | Time | Throughput |
|---|---:|---:|
| Yttri CUDA | about 5.0 s | 434–439 tok/s |
| Current fork, one forward | 1.593 s | 1375.7 tok/s |
| Current fork, server chunks 512 | 1.789 s | 1224.4 tok/s |
| Current fork, chunks 2048 | 1.647 s | 1330.2 tok/s |

Full/512/2048 produced identical final logits:

```text
argmax   248046
margin   3.3532981872558594
checksum dba55b1794191e1d
```

Chunk 2048 improves 4B throughput only 8.6% over server default 512. Default stays 512 because it bounds 35B transient VRAM and preserves decode interleaving. No model-specific scheduler policy added for this small gain.

## Synchronized Candle vs llama.cpp timing

Both probes use identical 33 prompt tokens and 127 teacher-forced decode calls. llama.cpp timing calls `llama_synchronize(ctx)` after each successful `llama_decode`; enqueue-only timing is invalid.

| Model | Metric | Candle | llama.cpp | Candle/reference |
|---|---|---:|---:|---:|
| 4B | ready-to-prefill load | 1.094 s | 1.674 s | 0.654× |
| 4B | 33-token prefill | 251.0 tok/s | 618.9 tok/s | 0.406× |
| 4B | B=1 decode | 56.57 tok/s | 88.47 tok/s | 0.639× |
| 9B | ready-to-prefill load | 1.856 s | 2.589 s | 0.717× |
| 9B | 33-token prefill | 188.9 tok/s | 521.9 tok/s | 0.362× |
| 9B | B=1 decode | 41.63 tok/s | 55.66 tok/s | 0.748× |

The 33-token prefill row measures launch latency and must not be presented as long-prefill throughput.

## Decode utilization bubbles

A 100 ms `nvidia-smi` trace showed zero GPU utilization in 59.9% of samples during short decode. This confirms host-side bubbles between CUDA bursts, but full-logit readback was not the main cause.

Trace mode inserted one explicit device synchronization after model forward. Median steady-state split:

| Phase | B=1 | B=4 |
|---|---:|---:|
| Forward host wall before final sync | 14.45 ms | 36.47 ms |
| Remaining GPU work at final sync | 1.74 ms | 2.22 ms |
| Full-logit D2H | 0.28 ms | 4.56 ms |
| CPU greedy sampling | 0.30 ms | 1.24 ms |
| Tokenizer/SSE drain | 0.02 ms | 0.05 ms |
| Accounted cycle | 16.87 ms | 44.62 ms |

At B=4, D2H plus sampling consumed about 13% of one cycle. Eliminating both had an approximate 1.15x ceiling; it could not explain the full gap to llama.cpp.

Phase timers inside eight attention blocks located larger host overhead:

| Attention host phase | B=1 | B=4 before batching fixes |
|---|---:|---:|
| RoPE/view construction | 1.03 ms | 5.09 ms |
| Q8 K/V quantization | 2.46 ms | 13.53 ms |
| Cache append | 0.35 ms | 1.43 ms |
| Full-cache Q8 dequantization | 1.49 ms | 4.37 ms |
| FA2 dispatch | 1.10 ms | 2.74 ms |
| Total measured attention phases | 7.71 ms | 29.03 ms |

Three existing-path changes removed repeated host work without a new CUDA kernel:

1. Copy `[B, vocab]` to host once, then split rows on CPU. B=4 D2H fell from 4.56 ms to 2.16 ms.
2. Quantize K and V once as `[B, 1, n_kv, head_dim]`. B=4 Q8 quantization fell from 13.53 ms to 3.36 ms.
3. Apply RoPE once over B when positions match; retain per-slot fallback after batch shrink/admission. B=4 RoPE host time fell from 4.74 ms to 1.45 ms.

Server greedy sampling also returns before cloning the full logits vector when penalties are disabled.

Diagnostic cycle medians improved in sequence:

| Stage | B=4 cycle |
|---|---:|
| Baseline | 45.57 ms |
| One logits D2H | 41.78 ms |
| Batched Q8 K/V quantization | 36.03 ms |
| Batched equal-position RoPE | 31.41 ms |

These traces include explicit synchronization and diagnostic logging. Production throughput below is the promotion metric.

## Final decode matrix

Direct isolated engine, one adapter, fixed GPU baseline:

| Metric | Candle | llama.cpp | Candle/reference |
|---|---:|---:|---:|
| B=1 decode | 55.96 tok/s | 87.08 tok/s | 64.3% |
| B=4 aggregate decode | 104.57 tok/s | 188.11 tok/s | 55.6% |
| B=4/B=1 aggregate gain | 1.869x | 2.160x | — |

The previous isolated B=4 Candle result was 84.50 tok/s. Current B=4 is 23.8% faster. B=4 VRAM remains 3480 MiB; no WDDM paging occurs.

HTTP end-to-end, three 128-token waves per batch size:

| Metric | Before | Current | llama-server | Current/reference |
|---|---:|---:|---:|---:|
| B=1 | 53.35 tok/s | 54.86 tok/s | 80.86 tok/s | 67.8% |
| B=4 aggregate | 83.32 tok/s | 112.17 tok/s | 164.69 tok/s | 68.1% |
| B=4 per request | 20.84 tok/s | 28.07 tok/s | 41.19 tok/s | 68.1% |
| B=4/B=1 aggregate gain | 1.562x | 2.045x | 2.037x | — |

Production gain is 2.8% at B=1 and 34.6% at B=4. Candle batch scaling now matches llama-server closely, although absolute throughput remains lower. HTTP prompt tokenization differs, so direct engine results remain the cleaner reference.

After optimization, remaining B=4 host attention work is concentrated in per-slot full-cache Q8 dequantization and FA2 dispatch. A fused Q8-cache attention path or a layout that permits batched cache reads is the next evidence-based target. GPU argmax is secondary after the one-transfer logits fix.

## Long-context split-K rollback

A user-visible numerical-list degeneration appeared around 2K context tokens and disappeared temporarily after submitting the same conversation again. The exact boundary matched the custom split-K flash-decode threshold rather than the configured 32K context limit.

An isolated 2025-state teacher-forced A/B used identical model state and tokens. FA2 and split-K logits were bit-exact through step 2014, then first diverged at step 2015 when KV length reached 2048. Full-vector metrics at the first affected state:

| Metric | FA2 vs split-K |
|---|---:|
| cosine | 0.9999225 |
| nRMSE | 0.01248 |
| max abs | 0.1730 |
| argmax | equal, reference margin 5.08 |

The high-margin synthetic state kept the same argmax, but stochastic low-margin generation can choose a different token and then permanently diverge through recurrent state. Split-K also provided no measured throughput gain in this run: FA2 49.80 tok/s versus split-K 49.65 tok/s.

FA2 is therefore the production default at every KV length. Split-K requires explicit diagnostic opt-in with `QWEN36_ENABLE_SPLITK_DECODE=1` until a full long-context parity and performance gate passes.

A second server-side defect reset stochastic RNG state every token by cloning the per-slot sampler map before each scheduler step. The sampler now remains installed for the scheduler lifetime and advances shared per-slot RNG state in place. OpenAI Chat also now accepts the sampling fields already sent by the web UI (`top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, and `seed`) instead of silently ignoring them.

Production validation after both fixes:

- two sequential 2300-token outputs crossed KV=2048 with HTTP 200 and no numerical-list degeneration;
- fixed seed produced bit-exact output hashes across both runs;
- direct same-slot reset/replay compared 16 full-logit states and remained bit-exact;
- invalid sampling values are rejected before generation.

## Correctness status

- 4B: 128/128 teacher-forced argmax equal.
- 9B: 126/128 argmax equal; two divergences have llama.cpp margins 0.0585 and 0.0127.
- 4B step 111 remains unresolved: direct quantized path nRMSE 0.4567, max abs 6.3868. Full dequantization plus cuBLAS reduces it to nRMSE 0.1123, max abs 1.9841, locating most drift in quantized matvec. It is diagnostic-only at 7.39 tok/s and is not a production fallback.
- Dense batching changes passed full-logit B=1 versus B=3 comparison: all 248320 logits bit-exact.
- Unequal-position B=2 to B=1 shrink remained bit-exact against sequential generation.
- IQ indexed MoE CUDA suite passed 26/26, including nonzero view offsets and non-contiguous rejection.
- Comparator fail-closed tests passed 10/10; empty/truncated inputs, different step sets, missing tokens, duplicate records, mismatched teacher-forced streams, and invalid full-logit values are rejected.
- Live Qwen3.5-4B passed Chat, Responses, and Anthropic APIs in stream/non-stream modes; no-auth returned 401.

Dense models remain unpromoted until the token-specific numerical spike receives a calibrated gate or runtime fix.
