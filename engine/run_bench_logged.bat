@echo off
setlocal
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "QWEN36_CUDA_GRAPHS=1"
set "QWEN36_TRACE=1"
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
target\release\bench_candle_direct.exe %* > D:\Projects\yttri-build\bench_live.log 2>&1
