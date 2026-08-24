@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch\candle-kernels\src\mmq_gguf
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -ptx -std=c++17 -O3 --expt-relaxed-constexpr -arch=compute_86 candle_mmq_dense.cu -o D:\Projects\yttri-build\test_mmq_dense.ptx
echo EXITCODE=%ERRORLEVEL%
