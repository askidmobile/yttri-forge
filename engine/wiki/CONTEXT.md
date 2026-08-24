# Codebase Wiki — Navigation Guide

This project has a compiled knowledge wiki. Use it instead of scanning raw files.

## How to use this wiki

1. Start at INDEX.md — scan the topic table to find relevant modules
2. Read 1-3 topic articles relevant to your current task
3. Check coverage tags:
   - [coverage: high] — trust this section, skip raw files
   - [coverage: medium] — good overview, check raw sources for implementation details
   - [coverage: low] — read the raw source files listed in Sources
4. Check concepts/ for cross-cutting patterns (batch-axis wall, parity-first prototyping)
5. Only read raw source files when you need code-level detail

## When NOT to use the wiki
- Writing new code (read the actual source files for exact syntax/types)
- Debugging a specific function (go to the file directly)
- The wiki article says [coverage: low] for what you need

## Stats
Compiled: 2026-08-08 | Topics: 5 | Concepts: 2 | Sources: 23 | Auto-updates on session start

## Topic map
- **project-overview** — fork purpose, workspace layout, branch feat/qwen35-batching
- **qwen35-batch** — the continuous batching prototype (BatchScheduler + BatchModel + real GGUF adapter); contains the central negative result
- **candle-core** — upstream candle framework crates (core/nn/transformers/datasets/pyo3/ug + excluded kernels/onnx/flash-attn)
- **candle-examples** — 90+ model examples + WASM demos
- **infrastructure** — CI workflows, Makefile, pre-commit hooks

## Concept map
- **batch-axis-wall** — Candle Metal/CUDA kernels are per-token; true batched decode needs a kernel rewrite (the fork's central finding)
- **parity-first-prototyping** — prove correctness on a deterministic mock before touching real weights; makes the negative speedup result trustworthy
