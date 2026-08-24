@echo off
setlocal
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64;%PATH%"
cd /d D:\Projects\yttri-build
rem tg-only, 8 повторов (~45s decode). Прежнее окно (delay 11, dur 6) попадало
rem в загрузку модели — трейс был без единого ядра.
nsys profile -o D:\Projects\yttri-build\trace_llama --force-overwrite=true -t cuda --delay 35 --duration 60 D:\Projects\yttri-build\llama.cpp\build-nmake-cuda124\bin\llama-bench.exe -m D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf -ngl 99 -n 128 -r 8
nsys stats --report cuda_gpu_kern_sum --format table --force-export=true D:\Projects\yttri-build\trace_llama.nsys-rep > D:\Projects\yttri-build\trace_llama_kernels.txt 2>&1
