@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\libnvvp;%PATH%"
set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64;%LIB%"
cd /d D:\Projects\yttri-forge\engine
cargo test --release --features cuda -p candle-core --test moe_table_cuda_tests -- --nocapture --test-threads=1
