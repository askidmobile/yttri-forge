@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "CUDA_COMPUTE_CAP=86"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_INCLUDE_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include"
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
cargo test --features cuda --package candle-core --test iq_quant_cuda_tests 2>&1
echo === EXIT CODE: %ERRORLEVEL% ===
