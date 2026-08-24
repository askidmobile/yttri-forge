@echo off
setlocal
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64;%PATH%"
set "QWEN36_CUDA_GRAPHS=1"
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
nsys profile -o D:\Projects\yttri-build\trace27b --force-overwrite=true -t cuda --cuda-graph-trace=node --delay 160 --duration 15 target\release\bench_candle_direct.exe D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf
nsys stats --report cuda_gpu_kern_sum --format table D:\Projects\yttri-build\trace27b.nsys-rep > D:\Projects\yttri-build\trace27b_kernels.txt 2>&1
