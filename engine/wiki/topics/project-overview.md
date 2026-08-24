# Project Overview

## Purpose
[coverage: high — 5 sources]
`candle-fork-qwen35-batch` — fork of HuggingFace [candle](https://github.com/huggingface/candle) (v0.9.2, minimalist ML framework for Rust). Fork's purpose: prototype **continuous batching for Qwen3.5-4B** on Candle/Metal — standalone crate `qwen35-batch` adds a `BatchScheduler` + `BatchModel` trait over the existing single-stream `quantized_qwen35` weights. Upstream candle crates (core/nn/transformers/examples/pyo3/wasm) remain available unmodified. Branch: `feat/qwen35-batching`.

## Architecture
[coverage: high — 4 sources]
Rust workspace, edition 2021, resolver 2. Workspace `members` (Cargo.toml:2-14): `candle-core`, `candle-datasets`, `candle-examples`, `candle-nn`, `candle-pyo3`, `candle-transformers`, `candle-ug`, `candle-wasm-examples/*`, `candle-wasm-tests`, `qwen35-batch`, `tensor-tools`. `exclude` (Cargo.toml:15-22): `candle-book`, `candle-flash-attn`, `candle-flash-attn-v3`, `candle-kernels`, `candle-metal-kernels`, `candle-onnx` — built standalone with feature-gated deps. Workspace version 0.9.2, license MIT OR Apache-2.0. Shared dep versions pinned in `[workspace.dependencies]` (Cargo.toml:34-107): `cudarc 0.19.1`, `half 2.5.0`, `memmap2 0.9.3`, `tokenizers 0.22.0`, `objc2 0.6.3`/`objc2-metal 0.3.1`, `gemm 0.19.0`, `safetensors 0.7.0`. Fork entry point: `qwen35-batch/src/lib.rs` re-exports `BatchModel`, `BatchScheduler`, `SchedulerStats`, `Slot`, `SlotStatus` + `DEFAULT_NUM_SLOTS=4`. Profile `release-with-debug` (Cargo.toml:109-111) inherits release + debug=true.

## Talks To
[coverage: medium — 3 sources]
`qwen35-batch` depends on local-path `candle-core` + `candle-nn` (Cargo.toml:8-9) and, on macOS, `candle-metal-kernels` (excluded from workspace but path-dep). Yttri app consumes this fork via `[patch]` in its `Cargo.toml:478-483` (REAL_MODEL.md:130). Python bindings via `candle-pyo3` (maturin). WASM via `candle-wasm-examples`/`candle-wasm-tests`. CUDA path optional via `cudarc` (non-macOS).

## API Surface
[coverage: high — 3 sources]
Public exports of `qwen35-batch` (lib.rs:30-35): `model::{BatchModel, DecodeBatch, PrefillChunk}`, `scheduler::{BatchScheduler, SchedulerStats}`, `slot::{Slot, SlotStatus}`, `const DEFAULT_NUM_SLOTS: usize = 4`. Features (Cargo.toml:31-41): `real-model` (GGUF loader + tokenizers + env_logger), `metal` (candle-core/nn metal), `cuda` (candle-core/nn cuda + candle-kernels + cudarc). Tests: `scheduler_parity` (always), `real_qwen35_batch` + `real_qwen35_quality` (require `real-model`).

## Data
[coverage: low — 1 sources]
No DB. Model data = GGUF weights on disk. Real model GGUF: `qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf` (~2.7 GB; see REAL_MODEL.md:70) currently lives in Yttri resources, loaded zero-copy via `memmap2` on Metal. Per-slot runtime state: ~114 MB/slot snapshot (GDN ssm/conv + KV-cache) — see qwen35-batch topic.

## Key Decisions
[coverage: high — 4 sources]
- **Fork over patch**: standalone crate inside candle workspace (rather than external) to reuse `candle-core`/`candle-nn`/`candle-metal-kernels` via local path + keep upstream examples/pyo3/wasm building.
- **Per-slot state, shared weights**: one `ModelWeights` copy, N independent recurrent-state + KV-cache sets (lib.rs:5-12) — matches LM Studio MLX / llama.cpp design.
- **Greedy sampler default**: sufficient for parity (batched vs sequential) — same logits ⇒ same argmax (model.rs:80-97).
- **First token from prefill logits**: avoid re-forwarding last prompt token into recurrent state (scheduler.rs:9-12; verified by `first_token_comes_from_prefill_logits`).
- **`PREFILL_CHUNK = usize::MAX`**: prototype does sequential prefill (whole prompt in one chunk); chunked prefill needs recurrent-state checkpoints, out of scope (scheduler.rs:62-65).

## Gotchas
[coverage: high — 3 sources]
- **True batched decode is a no-go on current Candle** without Metal-kernel rewrite — see qwen35-batch topic §"Architectural wall". Measured aggregate gain only ×1.07 (vs hypothesized ×1.15-1.35; REAL_MODEL.md:94-111).
- `candle-examples` ships 90+ model READMEs that are mostly one-liners — don't expect deep per-model docs in-repo.
- `make clean` runs `cargo clean`; `make clean-ptx` wipes generated PTX + clobbers `candle-kernels/src/lib.rs` (Makefile:3-8) — destructive, only for kernel rebuilds.
- Workspace `exclude`d crates (kernels/onnx/flash-attn/book) are NOT built by default `cargo build` at workspace root.

## Sources
- [README.md](../README.md)
- [CHANGELOG.md](../CHANGELOG.md)
- [Cargo.toml](../Cargo.toml)
- [Makefile](../Makefile)
- [qwen35-batch/REAL_MODEL.md](../qwen35-batch/REAL_MODEL.md)
