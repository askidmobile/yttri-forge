# Yttri Forge

[Русский](README.md) | English

**A local inference server built with Rust and CUDA: a single `yforge` binary, GGUF models, OpenAI and Anthropic APIs, and optimizations for long contexts and concurrent requests.**

Yttri Forge is being developed as a CUDA engine for Windows and Linux, designed to run language models on NVIDIA GPUs, including the RTX 3060 12 GB. This repository contains the engine and weight converter. The `yforge` HTTP server is built from the separate [qwen36-server](https://github.com/askidmobile/qwen36-server) repository.

[Website (RU)](https://forge.yttri.online/) · [Benchmarks](https://forge.yttri.online/#bench) · [Models](https://forge.yttri.online/#models) · [Windows/Linux installation](https://forge.yttri.online/#install) · [Server](https://github.com/askidmobile/qwen36-server)

`yforge` runs inference without a Python server or container: it needs model weights, an NVIDIA driver, and a suitable CUDA runtime. Python is used by some tools and benchmarks. Custom kernels, an int8 KV cache, CUDA Graphs, and MTP provide gains in specific tested configurations; results and test conditions are documented below.

## Where to start

- **Run a model and connect a client:** follow the [server quick start](#server-quick-start)
- **Deploy a Windows server with Open WebUI:** use [DEPLOY.md](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md)
- **Explore the engine or converter:** start with the [architecture](#architecture) and [forge-convert](#weight-converter-forge-convert)
- **See the results:** read [performance](#performance) and explore the [public comparison scripts](https://github.com/askidmobile/qwen36-server/tree/main/scripts/bench)
- **Explore the protocols:** see [CLI and configuration](https://github.com/askidmobile/qwen36-server#readme) and [API and media contracts](https://github.com/askidmobile/qwen36-server/blob/main/docs/engine-api.md)
- **Read the development story:** see the [article on vc.ru](https://vc.ru/id6112515/3167949-sozdanie-lokalnogo-inference-stack-dlya-llm-na-rtx-3060)

## What's in the stack

| Component | Capabilities |
| --- | --- |
| Engine | Runs supported GGUF models; a CUDA path with FlashAttention-2, quantized matrix multiplication, CUDA Graphs, and a paged KV cache |
| Scheduler | Continuous batching for the `qwen35` / `qwen35moe` architectures, a request queue, and prefix reuse |
| `yforge` server | OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages; SSE streaming and API keys |
| Multimodality | Image and video processing for compatible models and profiles with the required components |
| MTP | Speculative decoding with a separate compatible MTP component; explicitly enabled |
| Web interface | A built-in fallback chat and a proxy to Unsloth Studio; a separate Open WebUI deployment is covered in the server documentation |
| `forge-convert` | Converts source safetensors into an F16 `.ytf16` sidecar and a standalone `.ytf` v2 container; includes tensor verification tools |

Optimization availability depends on the backend, model architecture, and configuration. Compatible HTTP interfaces do not imply full support for every feature of the cloud APIs.

**The model is selected at startup.** To change models, restart the server with a different `--model` or profile. Model hot-swap and unload endpoints are disabled in the current HTTP router; hot-swap descriptions on the website and in older instructions refer to an earlier state of the project.

## Repositories

| Repository | Purpose |
| --- | --- |
| **[yttri-forge](https://github.com/askidmobile/yttri-forge)**, this repository | Active engine development in `engine/`, the `forge-convert` converter, and weight-format experiments |
| **[qwen36-server](https://github.com/askidmobile/qwen36-server)** | HTTP APIs, configuration, scheduler integration, and the `yforge` executable |
| [askidmobile/candle](https://github.com/askidmobile/candle) | Historical fork; current development has moved to `yttri-forge/engine/` |

Building the server requires **both of the first two repositories**. `qwen36-server/Cargo.toml` references the engine through local `path` dependencies. No path edits are needed when the clones are placed side by side:

```text
workspace/
├── yttri-forge/
│   ├── engine/          # separate Cargo workspace: Candle fork
│   └── forge-convert/   # weight converter
└── qwen36-server/       # server → target/release/yforge(.exe)
```

## Models and hardware

The server's primary use case is GGUF models from the **Qwen 3.5 / 3.6 / 3.8 and Ornith 1.0 / 1.5** families that use the supported `qwen35` / `qwen35moe` architectures. The server also has a separate `gemma4` path.

The full [tested-model table on the website](https://forge.yttri.online/#models) includes Qwen 3.5 4B/9B, Qwen 3.6 27B/35B-A3B, Qwen 3.8 27B, Ornith 1.5 9B, and Gemma 4 E4B-it, with specific quantizations and test systems. Useful starting points:

- **Ornith-1.5-9B Q4_K_M:** a working RTX 3060 12 GB profile in the [deployment guide](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md)
- **Gemma 4 E4B Q4_K_M** and **Qwen3.6 35B-A3B IQ2_XXS:** CUDA loading and execution results in the [runtime validation report](https://github.com/askidmobile/qwen36-server/blob/main/docs/lessons/2026-08-21-gguf-runtime-support-gate.md)

A `.gguf` file alone does not establish compatibility. The architecture, quantization, tokenizer, chat template, and required components must all be checked. Support for one model variant does not guarantee support for every size and quantization in that family. Vision/video requires a compatible vision component; MTP requires a matching head. A text model without MTP is enough for a first run.

### Requirements

- **Tested system:** Windows, RTX 3060 12 GB, and 64 GB RAM; tool versions and the profile are listed in `DEPLOY.md`
- **Deployment guide baseline:** an NVIDIA GPU with at least 8 GB VRAM, at least 16 GB RAM, and around 20 GB of disk space for builds and caches **plus** space for models. This does not guarantee that every model or long-context configuration will fit
- **Windows CUDA builds:** Rust MSVC, Visual Studio 2022 Build Tools with C++, CUDA Toolkit, and a compatible NVIDIA driver. The project's tested configuration uses CUDA 13.2; CUDA 12.4 runtime considerations are covered in `DEPLOY.md`
- **Platform scope:** under [decision BD-008 of September 25, 2026](docs/brief/decisions.md), Forge is a CUDA engine for Windows/Linux. On macOS, the Yttri product uses a separate MLX sidecar that is not part of Forge
- **Development and diagnostics:** `qwen36-server/Cargo.toml` retains a Metal option and a build without GPU support; the [server README](https://github.com/askidmobile/qwen36-server#readme) describes them as separate ways to run the server. These code paths do not mean that Forge ships as a macOS/Metal product; CUDA benchmark results do not apply to them

VRAM is used by weights, the KV pool, temporary buffers, and graphs. Context length and the number of slots affect this budget. Start with one slot and a small context window; a 128K profile requires separate tuning and is not a universal setting for a 12 GB GPU.

## Server quick start

This is a short path for **Windows and Linux + NVIDIA** with the build tools already installed. The full walkthrough uses Windows commands, with Linux build and launch alternatives below. For installation from scratch, automatic startup, and Open WebUI, use the [full guide](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md). Its directory layout differs from the side-by-side clones below: do not mix paths from the two layouts.

### 1. Clone the engine and server side by side

In a new working directory:

```console
git clone https://github.com/askidmobile/yttri-forge.git
git clone https://github.com/askidmobile/qwen36-server.git
cd qwen36-server
```

### 2. Build `yforge`

**Windows + CUDA**

In **cmd.exe**, from the `qwen36-server` directory:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set CUDA_COMPUTE_CAP=86
cargo build --release --features cuda --bin yforge
target\release\yforge.exe --help
```

The paths must match your installation. `CUDA_COMPUTE_CAP=86` is set here for the RTX 3060; for another GPU, use its compute capability. The build produces `target\release\yforge.exe`.

**Linux + CUDA**

After installing the NVIDIA driver, CUDA Toolkit, Rust, and C/C++ build tools, use the side-by-side clones from step 1. In the `qwen36-server` directory:

```bash
# Set the actual CUDA path and your GPU's compute capability.
export CUDA_PATH=/usr/local/cuda-12.8
export CUDA_COMPUTE_CAP=86
cargo build --release --features cuda --bin yforge
./target/release/yforge --help
```

In the `local.env` example below, replace the Windows model path with your Linux path. Start the server with `./target/release/yforge --env local.env`; the other configuration settings are the same. Tool installation and running as a systemd service are covered in the website's [Linux installation tab](https://forge.yttri.online/#install).

### 3. Prepare the model and configuration

Download a compatible GGUF, such as an Ornith-1.5-9B Q4_K_M variant, and check the model author's weight distribution terms. Model weights are not included in the repository. Model preparation and conversion from safetensors are covered in the [Model section of DEPLOY.md](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md#6-модель).

Create `local.env` next to the server's `Cargo.toml`. Replace the model path and key value; do not commit this file to Git with a real key:

```dotenv
MODEL=D:\Models\Ornith-1.5-9B-Q4_K_M.gguf
API_KEYS='[{"key":"REPLACE_WITH_YOUR_RANDOM_KEY","name":"local"}]'
HOST=127.0.0.1
PORT=18099
CTX=8192
CONTEXT_LIMIT=8192
SLOTS=1
MAX_TOKENS=2048
KV_POOL_Q8=1
KV_CACHE_DTYPE=q8
CUDA_GRAPHS=1
MTP=0
```

This is a conservative starting example, not a maximum-throughput profile. `CTX` and `CONTEXT_LIMIT` are aligned; increasing only the advertised context window does not automatically increase the engine's actual limit.

```bat
target\release\yforge.exe --env local.env --dry-run
target\release\yforge.exe --env local.env
```

`--dry-run` displays the configuration without loading the model. A successful `--dry-run` does not validate generation: wait for the server-ready message, then send the request below.

Configuration precedence: **CLI → process environment → env file**. Existing environment variables can override `local.env`.

### 4. Test the API

In a separate **PowerShell** window:

```powershell
$headers = @{ Authorization = 'Bearer REPLACE_WITH_YOUR_RANDOM_KEY' }
$models = Invoke-RestMethod -Uri 'http://127.0.0.1:18099/v1/models' -Headers $headers
$models.data

$body = @{
    model = $models.data[0].id
    messages = @(@{ role = 'user'; content = 'Say hello in one sentence.' })
    max_tokens = 256
    stream = $false
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod `
    -Uri 'http://127.0.0.1:18099/v1/chat/completions' `
    -Method Post -Headers $headers -ContentType 'application/json' -Body $body
$response.choices[0].message
```

Use the same key as in `local.env`. For an OpenAI-compatible client, set the Base URL to `http://127.0.0.1:18099/v1`, provide your key, and use the `id` returned by `/v1/models`.

The web interface is available at `http://127.0.0.1:18099/`: the server proxies the configured Studio backend and shows the built-in chat if that backend is unavailable. Open WebUI is installed separately.

### Troubleshooting startup

| Symptom | What to check |
| --- | --- |
| Cargo cannot find `qwen35-batch` or Candle | Both clones are side by side; `yttri-forge/engine/` exists |
| `nvcc` / `cl` not found | CUDA paths and the VS Build Tools environment |
| The Windows process exits before producing a useful log | CUDA runtime dependencies and `PATH` order; see section 9 of `DEPLOY.md` |
| HTTP 401 | The request key matches `API_KEYS`; `Authorization: Bearer …` is present |
| OOM, a sharp slowdown, or a KV-window warning | Reduce the context and number of slots; check VRAM and shared GPU memory |
| The response stops during reasoning | Check the client's `max_tokens`: it also limits the available generation budget |

Keep `HOST=127.0.0.1` for local use. For network access, configure access controls and an HTTPS proxy separately; do not expose the API directly to the internet. Avoid logging request bodies containing private data unless necessary.

## Architecture

```text
Chat / IDE / AI client
        ↓ HTTP + API key
qwen36-server / yforge
        ↓ API handling, queueing, and scheduling
yttri-forge/engine/qwen35-batch + Candle
        ↓ weight loading, KV cache, compute kernels
NVIDIA GPU / CUDA
```

- `engine/` contains a fork of [Hugging Face Candle](https://github.com/huggingface/candle) and remains a separate Cargo workspace
- `engine/qwen35-batch/` contains the runtime and scheduler; `qwen35` / `qwen35moe` use `BatchedEngine` even with a single slot
- In the server, `gemma4` uses a separate sequential `CandleEngine`; do not assume that batching capabilities apply to it
- `forge-convert/` belongs to the root Cargo workspace and prepares weight containers
- `docs/brief/`, `docs/specs/`, and `docs/plans/` preserve design rationale and experiment history; early documents may describe decisions that have since been revised

## Weight converter `forge-convert`

This step is **not required** for a normal GGUF server run. The converter is intended for the project's custom weight pipeline and research.

From the `yttri-forge` root:

```console
cargo build --release -p forge-convert
cargo run --release -p forge-convert -- --help
cargo test -p forge-convert
```

Implemented CLI modes:

- `--f16-heavy`: an F16 `.ytf16` sidecar for selected heavy projections; the engine has a separate dual-read path for prefill
- `--pack`: a standalone `.ytf` v2 container with tensor types, metadata, and an embedded tokenizer
- `--pack-vision` and `--pack-mtp`: package the corresponding components with a reference GGUF
- `--verify-ytf … --gguf …`: verify the container against a reference, tensor by tensor
- `--list-gguf`: inspect tensor names, shapes, and types

Sources: [CLI](forge-convert/src/main.rs), [packing](forge-convert/src/pack.rs), and [container loader](engine/qwen35-batch/src/real/ytf16.rs). The existence of a packing mode does not guarantee compatibility with an arbitrary model. GGUF remains the simplest path for a first run.

## Performance

The results below come from **the project's published reports**. They are not a promise of performance on every system or measurements of the current `main` branch.

### Latest RTX 3060 validation in the published log

The [Ornith measurement log](https://github.com/askidmobile/qwen36-server/blob/main/docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) begins on September 16 but includes later experiments. Its final **§103, dated September 18, 2026**, documents validation after a fix for OOM during long prefill.

System: **RTX 3060 12 GB**, **Ornith-1.5-9B Q4_K_M**, and a working profile with a 128K q8 KV pool and int8-QK. Five requests were run sequentially: 30K, 62K, 30K, 30K, and 127K.

| Post-fix check | yforge result |
| --- | ---: |
| Decode on the final ~127K prompt | **35.8 tok/s** |
| TTFT on that prompt | **115.33 s** |
| VRAM after each of the five requests | **7,863 MiB** |
| Decode on the three ~30K prompts | 48.8–49.3 tok/s |

**7,863 MiB is post-request memory usage, not peak usage.** TTFT measures time to the first token; it is not prefill throughput. This result comes from a specific sequence of requests after a memory-management fix, rather than a load test, a guarantee for every prompt, or a comprehensive quality evaluation. Exact inputs, the profile, and the sequence of changes are documented in §§95–103 of the log.

### Comparison with llama.cpp

The website's [interactive tables](https://forge.yttri.online/#bench) distinguish two test systems. On the RTX 3060 12 GB, using Ornith-1.5-9B Q4_K_M, a 30,232-token prompt, and flash attention in both engines:

| Metric | yforge | llama.cpp b10375 |
| --- | ---: | ---: |
| Prefill, F16 KV, five paired runs | 19.661 s | 19.716 s |
| Decode, F16 KV, five paired runs | 47.11 tok/s | 46.81 tok/s |
| Decode, int8 KV, 128K window, three paired runs | 45.60 tok/s | 45.20 tok/s |
| Prefill, int8 KV, 128K window | 20.40 s | 19.80 s |

The results are close: yforge is about 3% slower in q8 prefill. The comparison table and the final memory validation in §103 cover different stages of development; they must not be treated as a single run.

The second system, described in the [September 18 report](https://github.com/askidmobile/qwen36-server/blob/main/docs/research/2026-09-18-prod-parity-vs-llamacpp.md) as an RTX 4090 with **48 GB**, used Qwen3.8-27B Q8_0 and the same 256K window in both engines:

| Metric | yforge | llama.cpp b10375 |
| --- | ---: | ---: |
| Decode at 256K | 23.4 tok/s | 20.1 tok/s |
| Prefill at 8K | 2,179 tok/s | 1,826 tok/s |
| Steady-state TTFT, short prompt | 0.250 s | 0.342 s |
| VRAM at 256K | 35,241 MiB | 36,659 MiB |

These measurements are the basis for the published **+16% decode at 256K, −27% TTFT, and approximately −1.4 GiB VRAM**. They apply to the stated 48 GB system, not a standard 24 GB RTX 4090 or an RTX 3060. MTP was measured separately: +22% at 8K and +43% at 32K relative to yforge's own baseline; the compared llama.cpp build does not have the same MTP head.

### Reproducing the results

The [public benchmark suite](https://github.com/askidmobile/qwen36-server/tree/main/scripts/bench) was added to the repository on September 19. It includes `final_verify.py`, `batch_bench.py`, `mtp_bench.py`, `peak_vram.py`, and other checks. Read the [requirements and methodology](https://github.com/askidmobile/qwen36-server/blob/main/scripts/bench/README.md) first: model, binary, and log paths are tied to the original Linux test system and must be adapted. The scripts start and stop processes; use a dedicated test system.

After adjusting the paths, run from the `qwen36-server` root:

```bash
python3 scripts/bench/final_verify.py
```

For a comparable test of your own, record both repository commits, the comparison engine version, GPU/driver, GGUF hash, KV type, context, slots, MTP, prefix-cache state, exact inputs, and output length. Measure loading, time to first token, prefill, decode, and peak VRAM separately. Check response quality alongside speed. MTP can either speed up or slow down generation.

## Status and future work

This is an actively evolving engineering project. Code, deployment instructions, and research reports are updated independently, so record the versions used to obtain your results.

- **Already implemented:** GGUF runtime, CUDA optimizations, server APIs, the F16 sidecar, and `.ytf` v2 packing and reading
- **Research and decision history:** [engine plans](docs/plans/), [server research](https://github.com/askidmobile/qwen36-server/tree/main/docs/research), and [server plans](https://github.com/askidmobile/qwen36-server/tree/main/docs/plans)
- **Revised ideas:** the original W4A16 stage is marked as canceled in [TASKS.md](TASKS.md) following an experiment. The early brief is not a commitment to implement that stage or achieve the gains it described

Treat features listed in plans as research until they have been implemented and validated on the target hardware. No delivery dates or universal support for every model are promised.

## Reporting an issue

Use [this repository's Issues](https://github.com/askidmobile/yttri-forge/issues) for engine and conversion bugs; use [qwen36-server Issues](https://github.com/askidmobile/qwen36-server/issues) for HTTP APIs, configuration, and server startup.

Include your OS, GPU and VRAM/RAM capacity, driver/CUDA versions, both repository commits, exact model name and quantization, startup command, and a minimal request example. Before posting, remove API keys, personal paths, private prompts, and other sensitive data from your configuration and logs.

## License

The code is dual-licensed under your choice of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) (`MIT OR Apache-2.0`). The engine in `engine/` is a fork of [Candle](https://github.com/huggingface/candle) under the same terms; its licenses and Candle authors' copyright notices are preserved in `engine/`.

Unless explicitly stated otherwise, any contribution intentionally submitted for inclusion in the project is distributed under the same dual license, without additional terms.

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Model weights are distributed under their authors' own terms; the project's code license does not replace them.

<a id="donate"></a>

## Support the project

If you find the project useful, you can support its development. Donations fund GPU rental for testing new models.

Address for **USDT on the TON network**:

```text
UQAqmiUAf-kCo7pw1HM4KDrf4r8XCAsDDolODnZJnAydZ37O
```

> [!WARNING]
> Send **USDT only**, and only on the **TON network**. TON coins, other tokens, and transfers from other networks (TRC-20, ERC-20, BEP-20) will not be credited to this address and may be lost.

**Donate:** USDT on the TON network only, to the address above. Donations pay for GPU rental to test new models.
