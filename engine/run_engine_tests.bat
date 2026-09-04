@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64;%LIB%"
cd /d D:\Projects\yttri-forge\engine
cargo test --release --features cuda,real-model -p qwen35-batch --test expert_store --test qwen35moe_reference --test model_profile
