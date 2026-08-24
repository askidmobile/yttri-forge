# MoE logits parity tools

Candle tools:

```text
qwen35moe_logits <model.gguf> 128 rust <forced.jsonl>
qwen35moe_compare --gate <llama.jsonl> <candidate.jsonl>
```

Set `QWEN36_LOGITS_FULL_STEPS=16,45,50,92,111` for gate runs. Teacher forcing uses `fed_ids` from one fixed baseline JSONL, so every backend evaluates identical model states.

`llama-logits.cpp` is diagnostic source for pinned llama.cpp commit `8e7f22b67ef4667b4ddd50230771287f328cfb3f`. Configure that tree with NMake, CUDA 12.4, and `CMAKE_CUDA_ARCHITECTURES=86`; then build probe:

```bat
build-llama-logits-windows.bat D:\path\to\llama.cpp
llama-logits.exe <model.gguf> prompt-tokens.txt forced-tokens.txt 128 16,45,50,92,111
```

Use `-` instead of `forced-tokens.txt` for greedy diagnosis. Copy `prompt_tokens` and `fed_ids` from Candle JSONL into whitespace-separated text files.

Both probes append a `performance` JSON record with ready-to-prefill load time, prefill time/throughput, and teacher-forced decode time/throughput. llama.cpp calls `llama_synchronize()` inside measured regions; enqueue-only CUDA timing is invalid. Runs with one requested step report zero decode calls and `decode_tokens_per_s=0.0`.

Gate thresholds, calibrated on RTX 3060 / CUDA 12.4 / target UD-IQ2_XXS GGUF:

- 128 contiguous teacher-forced steps required;
- full logits required at steps `16,45,50,92,111`;
- cosine `>= 0.997`, nRMSE `<= 0.07`, max absolute error `<= 1.3`;
- at most 5 argmax differences, each with external-reference margin `<= 0.30`.

Comparator input is fail-closed: both files need nonempty identical step sets, exactly one nonempty `tokens.ids` record matching logits count, numeric finite margins, and numeric finite full-logit values when `values` is present. Gate mode also requires identical 128-token `fed_ids` teacher-forcing streams. Duplicate steps/tokens, truncated runs, one-sided full vectors, mismatched teacher forcing, and invalid values return nonzero.

Exact greedy sequence remains diagnostic. Low-margin decisions vary among llama.cpp, Candle dequantize+cuBLAS, and sparse PTX despite close logits.
