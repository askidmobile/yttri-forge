# candle-core & Framework Crates

## Purpose
[coverage: high — 5 sources]
Upstream HuggingFace candle framework crates (v0.9.2), unmodified in this fork. `candle-core` = minimalist ML framework for Rust with GPU support (README.md:8). Sibling crates: `candle-nn` (neural-net layers), `candle-transformers` (model architectures), `candle-datasets` (dataset loaders), `candle-pyo3` (Python bindings via maturin), `candle-ug` (unstructured CLI), `tensor-tools` (safetensors utilities). Excluded-from-workspace but present: `candle-kernels` (CUDA), `candle-metal-kernels` (Metal), `candle-onnx` (ONNX), `candle-flash-attn`/`-v3` (flash attention), `candle-book` (mdBook docs). These crates are the foundation `qwen35-batch` builds on.

## Architecture
[coverage: medium — 3 sources]
Each crate = independent Cargo package with `version.workspace = true` (0.9.2) / `edition.workspace = true` (2021) — see tensor-tools/Cargo.toml:1-10 for the pattern. `candle-core` exposes `Tensor`, `Device` (Cpu/Metal/Cuda), `DType` (F32/F16/BF16/F64/I64/...), `quantized::gguf_file` (GGUF zero-copy loader), autograd, backends. `candle-nn` builds layers (VarBuilder, conv, RNN GRU/LSTM per CHANGELOG v0.2.1, embedding, rmsnorm, etc.) on top of core. `candle-transformers` ships model architectures (llama, mistral, mixtral, qwen, phi, stable-diffusion, whisper, bert, T5, yolo, sam, blip, clip, mamba, RWKV, etc. — see root README:62-143). `candle-pyo3` = PyO3 bindings, installed via `maturin develop -r`; stub files via `python stub.py` (candle-pyo3/README.md:1-26). `candle-kernels`/`candle-metal-kernels` contain GPU compute kernels (README:1-line each; candle-kernels notes some implementations ported from [dfdx](https://github.com/coreylowman/dfdx)). `candle-onnx` adds ONNX support via prost-build (requires `protoc` in PATH, candle-onnx/README.md:6-21).

## Talks To
[coverage: medium — 3 sources]
- `candle-nn` → `candle-core` (path dep `../candle-core`).
- `candle-transformers` → `candle-core` + `candle-nn`.
- `qwen35-batch` → `candle-core` + `candle-nn` + `candle-metal-kernels` (macOS).
- `candle-pyo3` → `candle-core` (+ Python via maturin).
- `candle-kernels` → CUDA (`cudarc`); `candle-metal-kernels` → Metal (`objc2-metal`).
- `candle-onnx` → `candle-core` + ONNX runtime + `protoc` (build-time).
- `candle-flash-attn`/`-v3` → `candle-core` (flash-attention fused kernels, CUDA).

## API Surface
[coverage: medium — 3 sources]
- `candle-core`: `Tensor`, `Device::{Cpu, Metal, Cuda}`, `DType`, `Storage`, `CpuStorage`, ops (`matmul`, `softmax`, `conv`*`, `silu`, `rope`, ...), `quantized::{gguf_file, QTensor}`, `Result`, `Error`. Docs at [docs.rs/candle-core](https://docs.rs/candle-core).
- `candle-nn`: `VarBuilder`, `Linear`, `Conv`*`, `Embedding`, `RmsNorm`, `LayerNorm`, `RNN`/`GRU`/`LSTM`, `Func`, `Module`.
- `candle-transformers`: `models::{llama, mistral, qwen, ...}` architectures, `generation::LogitsProcessor`.
- `candle-pyo3`: Python module `candle` (import via maturin wheel).
- `tensor-tools`: CLI built on `clap` + `candle` + `safetensors` (tensor-tools/Cargo.toml:12-16).
- `candle-ug`: unstructured CLI over `candle-core` (minimal; contents in `candle-ug/src`).

## Data
[coverage: low — 1 sources]
No crate-level data. Crates operate on `Tensor`s in-memory. Model weights sourced externally as `safetensors` or GGUF (see qwen35-batch topic). Datasets loaded by `candle-datasets` from HF Hub / parquet (workspace dep `parquet 57`).

## Key Decisions
[coverage: medium — 3 sources]
- **Minimalist core**: candle-core is deliberately small; heavy model logic lives in `candle-transformers` (root README:8).
- **Feature-gated backends**: metal/cuda/cuda-f16 behind feature flags (per-crate Cargo.toml), not default — keeps CPU builds lean.
- **Workspace pinning**: all shared dep versions in root `[workspace.dependencies]` (Cargo.toml:34-107) to keep crates in lockstep.
- **Excluded crates** (Cargo.toml:15-22): kernels/onnx/flash-attn/book excluded from default workspace build because they pull heavy native deps (CUDA, protoc, mdBook) — build on demand.
- **dfdx reuse** (candle-kernels/README.md:3): CUDA kernels partly ported from dfdx rather than reimplemented.

## Gotchas
[coverage: medium — 3 sources]
- **Per-crate READMEs are mostly 1-liners** — real docs live in `candle-book/` (mdBook, excluded) and [docs.rs](https://docs.rs/candle-core). Don't grep READMEs for API detail.
- **`candle-onnx` needs `protoc` in PATH** at build time or compilation fails (candle-onnx/README.md:6-21).
- **Excluded crates** (`candle-kernels`, `candle-metal-kernels`, `candle-onnx`, `candle-flash-attn*`, `candle-book`) are NOT built by `cargo build` at workspace root — must `cargo build -p <crate>` or `cd` into them. `qwen35-batch` path-deps `candle-metal-kernels` anyway (macOS).
- **Version drift**: CHANGELOG.md last entry v0.3.1 unreleased / v0.3.0 2023-10-01 (CHANGELOG.md:4-10) — CHANGELOG lags behind workspace version 0.9.2; trust Cargo.toml over CHANGELOG for current version.

## Sources
- [README.md](../README.md)
- [Cargo.toml](../Cargo.toml)
- [CHANGELOG.md](../CHANGELOG.md)
- [candle-core/README.md](../candle-core/README.md)
- [candle-onnx/README.md](../candle-onnx/README.md)
- [candle-kernels/README.md](../candle-kernels/README.md)
- [candle-metal-kernels/README.md](../candle-metal-kernels/README.md)
- [candle-pyo3/README.md](../candle-pyo3/README.md)
- [tensor-tools/Cargo.toml](../tensor-tools/Cargo.toml)
