# candle-examples & WASM

## Purpose
[coverage: high — 4 sources]
`candle-examples` — command-line + WASM example models built on `candle-core`/`candle-nn`/`candle-transformers`. Ships 90+ reference implementations for SOTA models (LLM, vision, audio, multimodal). `candle-wasm-examples` + `candle-wasm-tests` — browser-runnable demos compiled to WASM (trunk/`build-lib.sh`). Used as the canonical "how to run model X on candle" reference and as online demos hosted on HF Spaces. Branch `feat/qwen35-batching` keeps these unmodified; `qwen35-batch` real-model adapter ports `quantized_qwen35.rs` logic from Yttri, not from these examples.

## Architecture
[coverage: high — 4 sources]
`candle-examples/examples/<model>/` — one dir per model, each with `Cargo.toml` (binary) + `README.md` + `src/main.rs`. Run via `cargo run --example <name> --release` (README.md:147). CUDA: add `--features cuda`; cuDNN: `--features cudnn` (README.md:150-151). `candle-wasm-examples/<model>/` — per-model WASM apps built with `trunk serve --release --port <port>` (README.md:165-171) or `sh build-lib.sh` (candle-wasm-examples/bert/README.md:9). WASM examples: bert, blip, chat-template, llama2-c, moondream, phi, quant-qwen3, segment-anything, t5, whisper, yolo. `candle-wasm-tests` — separate test crate for WASM paths.

## Talks To
[coverage: medium — 3 sources]
- All examples → `candle-core` + `candle-nn` + `candle-transformers` (workspace path deps).
- HF Hub (`hf-hub 0.5.0` workspace dep) for weight download.
- WASM examples → trunk/wasm-bindgen; assets fetched from web at runtime (bert/README.md:19).
- `candle-examples/examples/quantized-qwen3` + `quantized-qwen3-moe` — closest in-repo relatives to `qwen35-batch` (Qwen quantized GGUF path).

## API Surface
[coverage: medium — 2 sources]
Not a library — binary examples. Each `examples/<model>/src/main.rs` is a `fn main()`. Common pattern: load tokenizer + weights from HF Hub, build model via `VarBuilder`, run forward, decode/sample. WASM examples export `Model` class to JS (bert/README.md:16). Models covered (README.md:62-143): LLaMA v1/v2/v3 + SOLAR, Falcon, Codegeex4, GLM4, Gemma v1/v2, RecurrentGemma, Phi 1/1.5/2/3, StableLM, Mamba, Mistral 7b, Mixtral 8x7b, StarCoder/2, Qwen1.5, RWKV v5/v6, Replit-code, Yi, Quantized LLaMA/Qwen3/Qwen3-MoE, Stable Diffusion 1.5/2.1/SDXL/Turbo, Wuerstchen, YOLO v3/v8, Segment Anything, SegFormer, Whisper, EnCodec, MetaVoice, Parler-TTS, T5, Bert, JinaBert, DINOv2, VGG, RepVGG, BLIP, CLIP, TrOCR, Marian-MT, Moondream, and more.

## Data
[coverage: low — 1 sources]
No in-repo model weights — all fetched from HF Hub at runtime (or placed manually for WASM demos; see llama2-c wget example, README.md:166-168). Tokenizers + configs also from HF Hub.

## Key Decisions
[coverage: medium — 2 sources]
- **One example dir per model** — discoverable, self-contained, independently buildable.
- **Release build by default for examples** (`cargo run --example X --release`, README.md:147) — debug-mode perf is unusable for inference.
- **Feature-gated backends per-invocation** (`--features cuda`/`cudnn`, README.md:150-151) — no global backend flag.
- **WASM uses HF Spaces for hosting** (README.md:52-60) — online demos at `huggingface.co/spaces/lmz/...`.

## Gotchas
[coverage: medium — 3 sources]
- **Per-model READMEs are 1-liners** (file_glob confirmed) — don't expect usage detail in-repo; read `src/main.rs`.
- **WASM build needs trunk** (not plain `cargo build --target wasm32-unknown-unknown`) for the full demo apps.
- **`quantized-qwen3` example ≠ `qwen35-batch` real adapter** — the adapter was ported from Yttri's `quantized_qwen35.rs` (4854 lines, REAL_MODEL.md:117), which differs from the in-repo `candle-examples/examples/quantized-qwen3` example.
- Some example dirs appear empty/placeholder (`z_image/`) — check before relying.

## Sources
- [README.md](../README.md)
- [candle-examples/README.md](../candle-examples/README.md)
- [candle-wasm-examples/bert/README.md](../candle-wasm-examples/bert/README.md)
- [qwen35-batch/REAL_MODEL.md](../qwen35-batch/REAL_MODEL.md)
