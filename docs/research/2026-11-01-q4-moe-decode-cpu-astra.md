# Q4_K MoE decode on RTX 2080 Ti: CPU, NUMA and scheduling experiments

**Date:** 2026-11-01  
**Stand:** RTX 2080 Ti 11 GiB, PCIe 3.0 x16, 2× Xeon E5-2686 v4 (36 physical / 72 logical cores), 125 GiB RAM, CUDA 12.4  
**Model:** `Ornith-1.5-35B-Q4_K_M.gguf`, 41 MoE layers

This note records rejected experiments so they are not repeated. No experimental path from this document remains enabled in production code.

## Corrected model geometry

The current GGUF was inspected directly instead of relying on the earlier profile note:

| Field | Actual value |
|---|---:|
| MoE layers | 41 |
| Experts per layer | 256 |
| Top-k per token | **8** |
| Expert feed-forward width | 512 |
| Hidden width | 2048 |
| Gate / up dtype | Q4_K |
| Down dtype | **Q4_K** |

Earlier notes assumed top-4 and a Q6_K down matrix. Both assumptions are wrong for this file. This matters because every estimate must count eight experts and three Q4_K matrices per active route.

## Baselines on the stand

| Configuration | Decode |
|---|---:|
| yforge CPU-MoE, 18 threads, node 1 pinned | 11.35 tok/s |
| yforge CPU-MoE, 36 threads, node 1 pinned | 14.13–14.4 tok/s |
| llama.cpp `-ngl 0`, 36 threads | 4.70 tok/s |
| llama.cpp `-ngl 99 -ot exps=CPU`, 18 threads, node 1 pinned | 28.88 tok/s |
| llama.cpp `-ngl 99 -ot exps=CPU`, 36 threads | 31.40 tok/s |
| llama.cpp `-ngl 99 -ot exps=CPU --numa distribute`, 36 threads | 27.95 tok/s |

The llama.cpp short-context result is 31.40 tok/s, not the historical 45.75 tok/s figure. A `llama-bench` depth run at 131k did not finish within nine minutes and was stopped. The old 45.75 result must therefore be reproduced with the exact binary and filled KV state before it is used as a release gate.

## Measured critical path

With 36 threads on one NUMA node, yforge's CPU-MoE interior averaged:

| Phase | ms/layer |
|---|---:|
| Quantize input | 0.014 |
| Gate + up | 0.542–0.562 |
| Down | 0.298–0.375 (depends on kernel variant) |
| CPU-MoE total | 0.87–0.93 |

The full hybrid boundary averaged 1.32 ms/layer:

| Phase | ms/layer |
|---|---:|
| D2H + stream synchronization | 0.196 |
| CPU expert computation | 0.880 |
| H2D + shared expert + residual tail | 0.245 |

At 41 MoE layers the CPU expert computation alone is roughly 36 ms/token. A 45 tok/s token budget is 22.2 ms, so changing only the matrix layout or a single SIMD kernel cannot reach the target.

## Rejected experiments

### 1. Reorder complete Q4_K blocks for eight-row GEMV

The experiment preserved each 144-byte `BlockQ4K` and changed the order from `[row][K-block]` to `[K-block][row-in-group]`. The x8 reader and fallback paths were checked for index consistency.

Result without NUMA pinning: packed **7.78 tok/s** versus raw **7.98 tok/s**. With correct NUMA placement both paths were within noise at about 11.8 tok/s. The layout was removed.

### 2. Native `Q4_Kx8` AVX2 kernel

A separate `[d][dmin][scales][qs]` layout matching llama.cpp's native Q4_Kx8 representation was packed and a new AVX2 kernel was written. It matched raw Q4_K dot products for eight different rows, multiple K-blocks and non-constant Q8_K activations.

The parity test passed, but full server decode did not improve. The extra packed copy was removed.

### 3. Direct model mmap for CPU experts

Loading expert tensors through the model mmap reduced RSS, but the VMA pages were split between NUMA nodes: about 8.7 GiB on node 0 and 31 GiB on node 1. Decode fell to 12.43 tok/s. The mmap path was removed.

### 4. NUMA movement of copied weights

`numactl --cpunodebind=1 --membind=1` was essential. Without it, 18-thread yforge was about 7.98 tok/s; with node-1 placement it reached 11.83 tok/s. However, two-socket interleaving and remote reads were slower:

| Threads / placement | tok/s |
|---|---:|
| 18 threads / node 1 | 11.83 |
| 36 threads / node 1 | 14.13–14.4 |
| 54 threads / node 1 | 14.76 |
| 72 threads / node 1 | 12.78 |
| 36 threads / interleaved nodes | 8.82 |

### 5. Two node-local weight copies

A read-only probe loaded all expert weights twice, one copy bound to each NUMA node, for a total of 19.1 GiB per node. It used real gate/up/down tensors and verified that direct one-node and split 4+4 outputs agreed within `2e-3`.

Measured with top-8 routing:

| Scheduling | ms/layer |
|---|---:|
| One node, contiguous row ranges | 0.641 |
| Two node-local copies, static ranges, two pools | 0.658 |
| Two node-local copies, per-iteration dispatch | 0.818 |
| Two node-local copies, persistent dispatcher | 0.811 |

The memory replicated correctly, but a single node with static contiguous ranges remained as fast or faster. The probe was not promoted into the runtime.

### 6. Static row partitioning in production

A persistent Rayon pool with contiguous row ranges was tested behind `MOE_CPU_STATIC=1`. It replaced work stealing with one fixed range per worker, matching llama.cpp's row distribution strategy.

Result: **12.98 tok/s**, slower than the existing 14.4 tok/s path. The flag and code path were removed.

## What is established

- The gap is not caused by a missing eight-row Q4_K arithmetic kernel; parity was achieved without a decode gain.
- Crossing Q4 weights through PCIe cannot reach 45 tok/s. At 540 MiB/token, 45 tok/s needs about 24 GB/s, above practical PCIe 3.0 x16 bandwidth.
- Copying weights to both NUMA nodes does not solve the serialized dispatch and synchronization path.
- Fixed row ranges alone do not reproduce llama.cpp's advantage.
- The actual model consumes eight experts across 41 layers, so all prior top-4 CPU estimates were too optimistic.

## Required next architecture

To exceed llama.cpp on this machine, the decode path needs more than another layout variant:

1. CPU expert compute must fall below roughly 0.3 ms/layer.
2. GPU work, transfers, synchronization and sampling must fit in about 10 ms/token.
3. The scheduling unit should batch several routes or accepted speculative tokens so that CPU/GPU dispatch is amortized and INT8 kernels receive enough rows to become efficient.
4. The comparison baseline must be reproduced on the same binary and KV state at 128k and 256k.

The current per-layer hybrid architecture has a measured ceiling well below 45 tok/s. No production change from this investigation is justified yet.
