@echo off
setlocal
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64;%PATH%"
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
nsys profile -o D:\Projects\yttri-build\trace4b --force-overwrite=true -t cuda --delay 0 --duration 3 target\release\bench_candle_direct.exe D:\Models\yttri\qwen3.5-4b\Qwen3.5-4B-Q4_K_M.gguf
nsys stats --report cuda_gpu_kern_sum --format table --force-export=true D:\Projects\yttri-build\trace4b.nsys-rep > D:\Projects\yttri-build\trace4b_kernels.txt 2>&1
