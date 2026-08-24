@echo off
setlocal
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
target\release\bench_candle_direct.exe %*
