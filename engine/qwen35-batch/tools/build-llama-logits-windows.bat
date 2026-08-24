@echo off
setlocal EnableExtensions
if "%~1"=="" (
  echo usage: build-llama-logits-windows.bat D:\path\to\llama.cpp
  exit /b 2
)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
set "ROOT=%~1"
set "BUILD=%ROOT%\build-nmake-cuda124"
cl /nologo /std:c++17 /EHsc /O2 /MD /I"%ROOT%\include" /I"%ROOT%\ggml\include" "%~dp0llama-logits.cpp" /link /LIBPATH:"%BUILD%\src" /LIBPATH:"%BUILD%\ggml\src" /LIBPATH:"%BUILD%\ggml\src\ggml-cuda" llama.lib ggml.lib ggml-cpu.lib ggml-cuda.lib ggml-base.lib /OUT:"%~dp0llama-logits.exe"
exit /b %ERRORLEVEL%
