# candle-fork-qwen35-batch Wiki — Schema

Last updated: 2026-08-08

## Topics

| Slug | File | Description |
|------|------|-------------|
| `project-overview` | topics/project-overview.md | Fork purpose, workspace layout, branch feat/qwen35-batching |
| `qwen35-batch` | topics/qwen35-batch.md | Continuous batching prototype crate — BatchScheduler + BatchModel + real GGUF adapter |
| `candle-core` | topics/candle-core.md | Upstream candle framework crates (core/nn/transformers/datasets/pyo3/ug + excluded kernels/onnx/flash-attn) |
| `candle-examples` | topics/candle-examples.md | 90+ model examples + WASM demos |
| `infrastructure` | topics/infrastructure.md | CI workflows, Makefile, pre-commit hooks |

## Concepts

| Slug | File | Connects |
|------|------|----------|
| `batch-axis-wall` | concepts/batch-axis-wall.md | qwen35-batch, candle-core, project-overview |
| `parity-first-prototyping` | concepts/parity-first-prototyping.md | qwen35-batch, project-overview, infrastructure |

## Naming conventions
- Topic slugs: lowercase-kebab-case.
- Concept slugs: lowercase-kebab-case, descriptive of the pattern.
- Cross-references: markdown links (per `link_style: markdown` in .wiki-compiler.json).
- Coverage tags on every section: `[coverage: high/medium/low — N sources]`.

## Evolution Log
- 2026-08-08: Initial schema generated from 5 topics, 2 concepts.
