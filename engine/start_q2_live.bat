@echo off
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "QWEN36_MODEL=D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf"
set "QWEN36_SLOTS=1"
set "QWEN36_CTX=2048"
set "QWEN36_PORT=18099"
set "QWEN36_API_KEY=test"
set "RUST_BACKTRACE=1"
D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe
