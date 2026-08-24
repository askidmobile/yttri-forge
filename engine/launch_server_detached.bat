@echo off
REM Launches qwen36-server in background via start /b, then exits immediately.
REM SSH-friendly: calling this .bat returns control to SSH which can disconnect.
REM Redirect is on the start line itself — the inner bat runs the exe directly.
start "qwen36" /b cmd /c "D:\Projects\yttri-build\run_live_inner.bat >D:\Projects\yttri-build\live.log 2>&1"
exit /b 0
