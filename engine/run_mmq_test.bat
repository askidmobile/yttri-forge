@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64;%LIB%"
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch
cargo test --release --features cuda --package candle-core --test iq_quant_cuda_tests -- --ignored --exact mmq_mma_matches_reference --nocapture --test-threads=1
