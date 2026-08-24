@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64;%LIB%"
rem отдельный target для worktree — не конфликтует с master-сборками.
set "CARGO_TARGET_DIR=D:\Projects\yttri-build\candle-fork-vramfit\target"
cd /d D:\Projects\yttri-build\candle-fork-vramfit
cargo build --release --features real-model,cuda --bin bench_candle_direct
