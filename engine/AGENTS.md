# Project Rules — candle-fork

Auto-applied by Warp every conversation. Operational lessons + project conventions. Update when a mistake repeats or a new hard-won fact is learned.

## Repo layout

- Fork of HuggingFace candle (v0.9.2), branch `feat/qwen35-batching`.
- `candle-core/` — quantized CUDA/Metal/CPU dispatch. IQ quant work lives here (`src/quantized/`).
- `candle-kernels/src/quantized.cu` — CUDA dequantize kernels + lookup tables.
- `qwen35-batch/` — continuous batching prototype + real GGUF model (`src/real/model_weights.rs`).
- `qwen36-server` lives in the separate project repo (`Qwen3.6 27B`, path-depends on this fork).
- Remote (fork): `origin https://github.com/askidmobile/candle.git`. Upstream: `huggingface/candle`.
- Push flow: local commit → `git push origin feat/qwen35-batching` → `git pull` on yttri-win.

## yttri-win (Windows build/test machine) — SSH + shell

- SSH host alias: `yttri-win` (192.168.2.89, User Askid). See `~/.ssh/config`.
- **Workspace ownership override.** Never read, write, build, launch, or store results under `D:\Projects\yttri-build`; another agent owns it. Use only `D:\Projects\yttri-inference`. This overrides historical paths below.
- **Default shell is PowerShell.** `&&` is NOT valid in PowerShell — errors with «'&&' is not recognized». Always wrap commands in `cmd /c "..."`.
- **Nested quotes break PowerShell parsing of `(x86)` paths.** `C:\Program Files (x86)\...` inside a PS `-Command` string triggers `ObjectNotFound: (x86:String)`. The `(x86)` parenthesised token gets parsed as a command. Workarounds:
  - For `Test-Path`/file checks: use `powershell -NoProfile -Command "Test-Path 'C:\Program Files (x86)\...\file.bat'"` (single quotes inside double-quoted `-Command`).
  - For commands needing `(x86)` paths + `&&`: **use `cmd /c` instead of PowerShell**, OR run a `.bat` file that contains the `(x86)` `call` line (PowerShell never sees the path).
  - Avoid `&&` entirely inside PS; chain with `;` or separate commands.
- **PATH for CUDA**: `set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;...;%PATH%"` only inside a `.bat`. Production toolkit on yttri-win is CUDA 13.2; set `CUDA_PATH` explicitly.
- **MSVC for nvcc**: nvcc needs `cl.exe` in PATH. Must `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64` before `cargo build/test --features cuda`. Without it: `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`.
- **Build candle with CUDA on yttri-win**: use a `.bat` that sets VsDevCmd + CUDA env then runs cargo. Pattern: `D:\Projects\yttri-build\run_iq_tests.bat`. Do NOT run bare `cargo test --features cuda` over SSH — it lacks MSVC env.
- **No parent Cargo manifest.** Stray `D:\Projects\yttri-build\Cargo.toml` makes server dependency workspace inheritance fail; keep it renamed outside Cargo discovery.
- **Run IQ CUDA tests**: `ssh yttri-win "cmd /c \"D:\\Projects\\yttri-build\\run_iq_tests.bat\""`. The bat does `cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch && cargo test --features cuda --package candle-core --test iq_quant_cuda_tests`.
- Repo on yttri-win: `D:\Projects\yttri-build\candle-fork-qwen35-batch` (separate from `candle-fork` which is the older copy).
- **Build qwen36-server with CUDA**: build under `D:\Projects\yttri-inference` with VsDevCmd, explicit CUDA 13.2 `CUDA_PATH`, and `cargo build --release --features cuda`. Do NOT build without `--features cuda` — the binary falls back to CPU device.

## qwen36-server (Q2_K_XL inference) — launch + diagnostics

- Binary: `D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe`.
- **Server needs env vars** (`QWEN36_MODEL`, `QWEN36_SLOTS`, `QWEN36_CTX`, `QWEN36_PORT`, `QWEN36_API_KEY`, `PATH` with CUDA bin). Launch via a `.bat` that sets them (`run_server_q2.bat`, `start_q2_test.bat`, `start_q2_live.bat`).
- **Foreground launch (диагностика):** `ssh -t yttri-win "cmd /c D:\Projects\yttri-build\start_q2_live.bat"` — показывает stdout/stderr в реальном времени. Сессия жива пока сервер работает. После Ctrl+C — сервер умираст (exe в foreground).
- **Detached launch через SSH не виснет** только если launcher .bat запускает `start /b` и сразу `exit /b 0`. Прямой `ssh yttri-win "cmd /c start /b cmd /c ..."` виснет — SSH ждёт фоновый процесс. Паттерн: `launch_server_detached.bat` → `start /b cmd /c "run_live_inner.bat >live.log 2>&1"` → `exit /b 0`. `run_live_inner.bat` задаёт env vars и запускает exe.
- **curl на yttri-win требует `--noproxy "*"`** — иначе прокси перехватывает localhost запросы, ответ пустой. Рабочий паттерн: `curl -s -m 60 --noproxy "*" http://127.0.0.1:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d @file.json`. JSON-body через `@file.json` — inline JSON ломается PowerShell кавычками.
- **`Start-Process ... -RedirectStandardOutput` does NOT capture the server's stdout** — the exe likely writes to its own log or buffers. Observed: process starts, GPU stays at 375 MiB (baseline), logs stay 0 bytes, process exits silently. Use the `run_test.bat` pattern instead (background `start /b`, poll for "loaded:" in a logfile, curl test, then kill).
- Model: `D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf` (~10 GB Q2_K_XL, 27B). Load takes >90s; 12 GB VRAM (RTX 3060).
- Server logs: `server_q2.log`, `server_q2_iq3.log`, etc. in `D:\Projects\yttri-build\`. All observed 0 bytes → server writes elsewhere or crashes before flush. Check `RUST_BACKTRACE=1` + run foreground with `ssh -t` to see the panic.
- **Detached launch over SSH (2026-08-09, WORKING pattern):** `start /b` и `Start-Process` дети **умирают при закрытии SSH-сессии** (sshd job-object kill). Рабочий способ — **Task Scheduler**: ps1 с `New-ScheduledTaskAction -Execute cmd.exe -Argument '/c run.bat > log 2>&1'` + `Register-ScheduledTask -Force` + `Start-ScheduledTask`. Процесс живёт в сессии 0, переживает SSH. Примеры: `D:\Projects\yttri-build\task_s4t.ps1` (сервер), `task_conc2.ps1` (клиенты), `task_8k.ps1` (stability).
- **WDDM VRAM paging = катастрофа decode (2026-08-09):** Q2_K_XL (11.8 GB) + 4 слота на RTX 3060 12 GB → VRAM 12022/12288 MiB → WDDM прозрачно пейджит аллокации в system RAM → decode step B=4 **13.4s** (vs 0.42s при B=1), медленные блоки «ротируются» (разные attn-блоки 1-3s на каждом шаге), GPU util 1-2%. На IQ2_XXS (9.4 GB, VRAM 9.9 GB занято) тот же шаг = **2.08s**. Правило: если nvidia-smi ≥ ~98% — не дебаж производительность, сначала уменьши VRAM.
- **curl -m timeout маскируется под engine hang:** клиент умер → SSE receiver дропнут → `try_send(Closed)` → слот cancelled → Finished. Снаружи выглядит как «зависшие запросы». Для длинных прогонов: curl `-m` >> ожидаемого времени (8192 tok × 2s = 4.5h → -m 30000), `QWEN36_REQ_TIMEOUT` тоже поднять (default 600s).
- **QWEN36_TRACE=1** — пошаговый trace: `[step] decode begin/end B=N (Xs)`, `[fdb] slow block N attn/delta Xs` (>50ms), `[hb]` каждые 5s в dispatch loop. Всё в stderr → в лог bat-редиректа.
- **Known failure RESOLVED (2026-08-08):** inference bailed with `"CPU matmul is not implemented for IQ3XXS"`. Root cause: `qwen36-server.exe` was built WITHOUT `--features cuda`. `select_device()` (engine.rs:381-394) has `#[cfg(feature = "cuda")]` → `Device::new_cuda(0)`, else falls through to `Ok(Device::Cpu)`. Without the cuda feature, device = CPU → all weights load as `QStorage::Cpu` → `cpu_fwd` → `matmul_t` → bail for RawQuantizedType (IQ types). **Fix: always rebuild via `build_cuda124.bat` (`cargo build --release --features cuda`)** — see build section above.
- **How to verify the binary has cuda:** `dumpbin /dependents qwen36-server.exe`. With cuda, size jumps from ~8.7 MB (no cuda) to ~23.4 MB (cuda kernels embedded). NOTE: `cudart`/`cublas` do NOT appear in dumpbin — candle uses `cudarc` with dynamic-loading (CUDA loaded via `LoadLibrary` at runtime, not statically linked). Size is the reliable indicator.
- **After rebuilding with cuda feature**: server loads Q2_K_XL on CUDA, inference works end-to-end. First curl returned valid chat completion (`"Hello! How can I help you today"`, 8 completion / 13 prompt tokens).

## Attention CUDA decode — F16 (2026-08-10)

- **Batched decode attention на CUDA — F16 matmul** (model_weights.rs, ветка `forward_attn_decode_batch`): KV cache хранится F16 → НЕ конвертировать в F32. q→F16, HGEMM, softmax в F32, out→F32. Дёшево и точно: `gemm_reduced_precision_f16=false` (default) = CUBLAS_COMPUTE_32F — аккумуляция F32.
- **Выигрыш на длинном контексте (35B-A3B, KV 5.4K, B=2): 2.2s → 0.35s/токен (6x).** На коротком контексте разницы нет (доминирует DeltaNet).
- Старый коммент «F16 дал numerical drift» — про ПОЛНУЮ F16-цепочку с F16 softmax. F16 matmul + F32 softmax дрейфа не даёт (проверено: 2×2500 токенов на 5.4K ctx, когерентно).
- Single-slot decode (`forward_attn` seq_len=1) оставлен F32 — там drift-комментарий не оспорен, сервер использует batched путь.
- **PREFILL_CHUNK=512** (scheduler.rs, env QWEN36_PREFILL_CHUNK): цельный prefill создаёт scores N×N×F32×heads (~2GB на 5.6K промпт) → CUDA OOM на 12GB. Чанкинг обязателен. Заодно decode других слотов interleave'ится с prefill.

## Prefill performance (updated 2026-08-13)

- **CUDA prefill already fused.** DeltaNet token loop runs inside sequence kernels with recurrent state held in registers (`0b3b5e2f`, `048463cc`); attention defaults to FlashAttention 2 with F32 accumulation (`ec13b1e4`). Do not restore old Rust-side per-token path.
- **Dense 4B long prefill:** 2191 tokens = 1375.7 tok/s whole, 1224.4 tok/s with server chunk 512; full/512/2048 final logits bit-exact. Yttri CUDA baseline 434–439 tok/s is older. Details: `docs/research/2026-08-13-qwen35-dense-cuda-comparison.md`.
- **Benchmark isolation is mandatory.** One GPU process and one loaded adapter; run ignored tests with `--exact --test-threads=1`. Duplicate model loads trigger WDDM paging and invalidate speed/VRAM results.
- **Tiled dequantize matmul** (`2fdd7a1e`) limits IQ fallback transient; default `PREFILL_CHUNK=512` remains for 35B VRAM and decode fairness.
- **Decode utilization dips need phase timing.** Split host enqueue, explicit GPU sync, D2H, sampling, and drain before blaming transfers; details: `docs/research/2026-08-13-qwen35-dense-cuda-comparison.md`.

## IQ quant CUDA (IQ3XXS, IQ2S, IQ3S, IQ2XS, IQ4XS)

- **2-bit CUDA MoE defaults to PTX after margin-aware parity + 4×8K pass.** Prefill B>4 uses expert-grid route tiles; decode B=1..4 keeps validated direct/grouped path. Use `QWEN36_MOE_BACKEND=reference` only for diagnostic rollback. Details: `docs/lessons/2026-08-13-iq-moe-cuda.md`.
- **8K teacher-forced гейт (2026-08-18, passed на 35B MoE IQ2_XXS):** llama free-run сквозь EOS -> candle forced (`QWEN36_LOGITS_IGNORE_EOS=1`) -> `run_compare.bat --gate` с `QWEN36_GATE_STEPS=8192` и `QWEN36_GATE_FULL_STEPS=16,1024,2048,4096,6144,8191`. Пороги глубинные (`full_vector_thresholds`/`max_reference_margin_for_drift` в qwen35moe_compare.rs): до 256 шага исторические 0.997/0.07/1.3+margin 0.30, глубже 0.988/0.17/3.0+margin 0.75 — дрейф глубины это интеграл q8-KV/q8_1 через рекуррентное состояние, argmax он не ломает (20/8192 низкомаржинальных). 128-гейт с дефолтами бит-в-бит прежний (27B recheck: passed, 0 divergences).
- **Isolated CUDA tests**: `candle-core/tests/iq_quant_cuda_tests.rs`. Run via `run_iq_tests.bat` on yttri-win. Current gate: 24 tests covering required IQ matrix, shared input, B=1/4/5/33, and grouped route-tile boundary.
- **The candle dispatch is CORRECT.** `QMatMul::forward` (mod.rs:1037-1060) → `xs.apply_op1_no_bwd(t)` → `Storage::apply_op1` (storage.rs:205-220) dispatches by INPUT storage → `cuda_fwd` for Cuda, `cpu_fwd` for Cpu. `QStorage::from_data` (mod.rs:87-153) routes IQ types to `cuda::load_quantized_bytes` for CUDA device. Loading path verified correct. The only failure mode is building without the cuda feature (see qwen36-server section above).
- **Dispatch path** (confirmed correct): `QMatMul::forward` → `cuda_fwd` (mod.rs:952-1035 CustomOp1) → for IQ types, fallback to `dequantize + cuBLAS` matmul in `cuda.rs:765`.
- **Model forward path** (`model_weights.rs:5009-5024`): `forward_inner` → `emb_cpu = tok_embeddings.forward(x)` (returns CPU f32) → `layer_in = emb_cpu.to_device(x.device())`. If `x.device()` = CUDA → `layer_in` on CUDA → all `QMatMul` weights already on CUDA (loaded via `load_heavy` with CUDA device).

### CUDA kernel gotchas (2026-08-11)

- **__constant__ + дата-зависимые индексы = до 32x сериализации на варп (2026-08-18).** IQ grid-таблицы в quantized.cu были `__constant__`; constant-кэш вещает один адрес/такт. На реальных весах IQ mmvq падал в 4-19x (iq2_xxs 48 GB/s, iq2_s 13), фикс `5eb10d0d`: `static const __device__`. Итог: 27B IQ2_XXS decode 7.7→21.7 tok/s (0.95x llama), 35B MoE 10.7→67.9 (0.78x), prefill 35B 1.00x llama. LUT с рантайм-индексами — только global/shared, никогда `__constant__`.
- **Перф-бенчи квантованных ядер — только с реалистичной заливкой весов.** Константный филл (0x5A) даёт всем лейнам один индекс таблицы → broadcast → бенч завышает IQ в 4-19x и цифры «hot=cold» врут. `QWEN36_PERF_RANDOM=1` в mmvq_perf_dense (`20d45b52`) воспроизводит модельные тайминги с точностью до процентов.
- **Толеранс дот-продукта с q8_1-активациями — от Σ|w·x|, не от |результата|** (`87cf36da`): при сокращении в доте относительная к результату ошибка взрывается (наблюдали 5% на |res|=10.8 при штатном q8-округлении). Гейт iq_quant_cuda_tests запускать ТОЛЬКО через run_iq_tests_serial.bat (--test-threads=1): 5 cuda_graph_* тестов падают предсуществующе (см. отдельную задачу), MoE/mmvq — зелёные.

- **Split-K flash-decode stays diagnostic-only.** First version had a cross-warp `m/l` race and corrupted generation at KV≥2048. Per-warp registers removed the race, but a 2025-state FA2 A/B still first diverges exactly at KV=2048 (nRMSE 0.01248, max abs 0.173) with no speed gain. FA2 is default; `QWEN36_ENABLE_SPLITK_DECODE=1` is explicit diagnostic opt-in.
- **Проверка деградации текстом**: uniq-3gram НЕ ловит цифро-мусор («2222», «( ( (»). Всегда читать хвост генерации глазами на 3K+ токенов.
- **MTP batched verify на Qwen3.8 — 1.51x (2026-08-19): 28.0 vs 18.6 tok/s, bit-exact.** Verify = один multi-token forward через batched decode-путь (slots=[slot;K], positions=pos..pos+K): attention batched decode уже корректен для K строк одного слота (append по cache_len), DeltaNet сериализуют `_seq`-ядра (delta_rule_batched.cu: внутренний цикл по строкам). Smem-вариант (`delta_rule_kernel_batched_seq_smem`) держит state головы [128×128]=64КБ в динамическом smem: 2 прохода global вместо 6×K (+6% e2e); >48КБ требует CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES (ставится Rust-диспатчем раз, фолбэк на global-вариант). Бит-эксактность: общее __device__-тело с generic-указателем state (global/smem — инструкции те же). При m<K — restore checkpoint + re-run префикса одним чанком (без D2H). **K (QWEN36_MTP_WIDTH, default 3) держать ЧУТЬ НИЖЕ типичной длины всплеска принимаемости: при m==K re-run пропускается.** K=4 на 27B МЕДЛЕННЕЕ baseline (15.1 tok/s — re-run горит в ~90% раундов), K=3 — 28.0. Бюджет раунда мерить QWEN36_MTP_TIMING=1; маржинальная цена verify-строки ~18ms (per-row attention host + state) — батч амортизирует только ~35ms/шаг фиксированных затрат. Rust-цикл по строкам вместо seq-ядер = -1.6K host-ops = хуже последовательного verify (14.2 tok/s, не повторять). Размеры DeltaNet 27B: inner=6144, n_v=48, head_v=128, head_k=128, n_k=16. На 27B гейт/пробу гонять только с 1 слотом.
- **Новый quant-dtype в CUDA = ТРИ whitelist'а**: `mod.rs::QStorage::from_data` (маршрут загрузки), `cuda.rs` (dequant f32/f16/rowslice + mmvq обычный и prequant — это РАЗНЫЕ match'и), `ggml_file.rs::qtensor_from_ggml` (сырые GGUF-байты). Пропустишь один — «not supported yet» с тремя разными формулировками (IQ1M ловил все три по очереди, 2026-08-19).
- **Qwen3.8-27B (UD-IQ2_XXS)**: arch qwen35, 65 блоков = 64 trunk + `nextn_predict_layers=1` (blk.64 — полная MTP-голова: attn+ffn+nextn.*). Транк-вычитание в build_model_common и model_profile. UD-рецепт 3.8 жмёт ssm_beta/ssm_alpha в IQ1_M (96 тензоров [48x5120]) — поддержка добавлена (mmvq+dequant, MMQ не нужен: n%128!=0).
- **cudaforge не видит изменений в .cuh-инклюдах** — кэш по самому .cu; меняешь заголовок, бампни .cu (см. комментарий в candle_mmq_dense.cu).
- **Build trap**: новый `.cu` в candle-kernels без `rerun-if-changed` → ptx.rs не перегенерируется, kernel не найден. build.rs теперь следит за всеми `src/*.cu`.

### Tokenizer GGUF gotchas (2026-08-09)

- **GGUF vocab хранит byte-mapped строки** (GPT-2 bytes_to_unicode): пробел = `Ġ`(U+0120), байт 0xF0 = `ð`(U+00F0), 0x9F = `Ł`(U+0141). Emoji разрезан BPE на частичные UTF-8 куски (`ĠðŁ` + `Ĳ±`).
- **HF tokenizers ByteLevel decoder декодит каждый токен отдельно** → частичные UTF-8 → U+FFFD. Поэтому `decode_text` — ручной: char→byte inverse map по всем токенам, один `from_utf8_lossy` на всю последовательность (tokenizer.rs).
- **Стриминг: holdback U+FFFD-хвоста** — промежуточный decode заканчивается '�' (недостроенный emoji), следующий токен достраивает. Эмитить '�' нельзя: префикс разойдётся. Flush в finish (qwen36-server engine*.rs).
- **vocab_probe.exe** (bin в qwen35-batch): `vocab_probe <gguf> [id...]` — печатает строки/байты токенов + e2e decode_text. Быстрее чем PowerShell-парсинг GGUF.
- **`Tokenizer::decode` skip_special не нужен вручную** — decode_text пропускает `<|...|>` сам.

### candle API gotchas (learned the hard way — do NOT repeat)

- **`QTensor` does NOT impl `Clone`.** Use `Arc<QTensor>` + `QMatMul::from_arc(arc.clone())` when you need the same weights for both `QMatMul` and `dequantize`.
- **`Device` does NOT impl `PartialEq`.** `assert_eq!(res.device(), device)` fails to compile. Use `res.device().same_device(&device)` (returns bool) or compare `location()`.
- **`f32` does not impl `Try`.** `((... )?)` with extra parens around a terminal `to_scalar::<f32>()?` makes the outer `?` apply to `f32` → `E0277: the ? operator cannot only be applied to values that implement Try`. Write `let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;` — no wrapping parens around the final `?`.
- **Zeroed IQ quant blocks dequantize to NONZERO values.** Grid lookup tables have nonzero entries at index 0 (e.g. `iq3xxs_grid[0] = 0x04040404`, `kvalues_iq4nl_f[0] = -127`). With `d=1.0` (f16) and zeroed qs, IQ3XXS/IQ2S/IQ3S/IQ2XS dequantize to ~1.0, IQ4XS to ~4064. Do NOT assert `== 0.0`. Assert finiteness, or compare CUDA matmul against a CUDA-dequantized reference (same kernel path → close match).
- **`from_float` panics for `RawQuantizedType` (IQ types) on CPU.** Can't use `QTensor::quantize` for IQ types. Construct from raw bytes via `QStorage::from_data(Cow::Borrowed(&raw), device, dtype)` — same path as the GGUF loader.

### CUDA kernel struct sizes (quantized.cu)

- `block_iq3_xxs`: `half d` + `uint8_t qs[3*QK_K/8]` = 2 + 96 = 98 bytes.
- `block_iq2_s`: `half d` + `qs[QK_K/4]` + `qh[QK_K/32]` + `scales[QK_K/32]` = 2 + 64 + 8 + 8 = 82 bytes.
- `block_iq3_s`: `half d` + `qs[QK_K/4]` + `qh[QK_K/32]` + `signs[QK_K/8]` + `scales[QK_K/64]` = 2 + 64 + 8 + 32 + 4 = 110 bytes.
- `block_iq2_xs`: `half d` + `uint16_t qs[QK_K/8]` + `scales[QK_K/32]` = 2 + 64 + 8 = 74 bytes.
- `block_iq4_xs`: `half d` + `uint16_t scales_h` + `scales_l[QK_K/64]` + `qs[QK_K/2]` = 2 + 2 + 4 + 128 = 136 bytes.
- `QK_K = 256` for all IQ types. Block size = 256.

## Local environment (macOS)

- **`git`, `ssh`, `ls` NOT in PATH zsh.** `git --no-pager status` → `zsh: command not found: git` (exit 127). Use full paths: `/usr/bin/git`, `/usr/bin/ssh`, `/bin/ls`. `/usr/bin/ssh -V` works. `/usr/bin/git push origin feat/qwen35-batching` works.
- **`cargo` IS in PATH** (via rustup/cargo env). Only system binaries are missing from zsh PATH.

## Reference tests (CPU-only, qwen35-batch)

- **CPU-only reference tests run on yttri-win without CUDA/Metal/MSVC.** `cargo test -p qwen35-batch --test qwen35moe_reference --features real-model` — no `--features metal` or `--features cuda` needed when device=CPU. Runs in 0.01s.
- **Test loop (macOS → yttri-win):** `cargo check --tests --features real-model,metal` locally (compile validation) → commit → `git push origin feat/qwen35-batching` → SSH pull on yttri-win → `cargo test --features real-model` on yttri-win (execution). macOS lacks a linker for the `real-model` test binaries; yttri-win has full toolchain.

## Self-improvement loop (auto)

After a non-trivial task (bugfix, build, deploy, debug, refactor >5 steps) — run the `learn-from-work` skill (`/learn`).
Trigger conditions:
- An error took more than one attempt to fix, or required googling/experimenting.
- The same thing was fixed twice in one session (pattern signal).
- User asks to "learn", "retrospective", "remember", "record lesson", "what did we learn".
Skip for: trivial one-shot fixes, typos, obvious errors. Do not record noise.

`/learn` uses layered memory — writes to the correct tier:
- **Kernel (this file)**: one-line facts that change behavior on most tasks. ≤200 lines total.
- **Wiki**: details, root cause analysis, dates. `docs/lessons/` if no wiki configured.
- **Agent Memory** (Oz, when available): cross-session/cross-project facts.
- **Skill**: multi-step procedures that don't fit in 1-3 lines.
- **Global Rule**: universal patterns (text to user, can't write programmatically).

This file is the kernel — keep it short. If >150 lines, `/learn` triggers an audit:
verbose entries promoted to wiki, one-line pointer left here. Precision test:
"if I remove this line, will the agent make a mistake?" If no → move to wiki.

## Conventions

- `llama.cpp` — только внешний донор знаний и независимый parity-reference. Никогда не добавлять его как Cargo/build/FFI/subprocess/runtime dependency `candle-fork`.
- Respond in Russian (per Global Rule).
- Do NOT add `Co-Authored-By` to commits unless explicitly told to (per Global Rule).
- Commit co-author line only on explicit user request.
