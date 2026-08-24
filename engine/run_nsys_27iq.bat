@echo off
setlocal
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64;%PATH%"
set "QWEN36_CUDA_GRAPHS=1"
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
nsys profile -o D:\Projects\yttri-build\trace27iq --force-overwrite=true -t cuda --cuda-graph-trace=node --delay 15 --duration 150 target\release\bench_candle_direct.exe D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf
nsys stats --report cuda_gpu_kern_sum --format table --force-export=true D:\Projects\yttri-build\trace27iq.nsys-rep > D:\Projects\yttri-build\trace27iq_kernels.txt 2>&1
