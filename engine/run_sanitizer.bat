@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64;%LIB%"
set "QWEN36_CUDA_GRAPHS=1"
set "QWEN36_TRACE=1"
set "QWEN36_GRAPH_MAX_LAYERS=4"
cd /d D:\Projects\yttri-build
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\compute-sanitizer\compute-sanitizer.exe" --tool memcheck --launch-timeout 300 --error-exitcode 1 D:\Projects\yttri-build\candle-fork-qwen35-batch\target\release\qwen35moe_logits.exe D:\Models\lmstudio-community\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf 4 rust > D:\Projects\yttri-build\sanitizer-out.txt 2>&1
