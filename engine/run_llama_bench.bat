@echo off
setlocal
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
D:\Projects\yttri-build\llama.cpp\build-nmake-cuda124\bin\llama-bench.exe %*
