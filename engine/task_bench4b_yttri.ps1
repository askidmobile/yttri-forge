$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_direct_bench.bat D:\Models\yttri\qwen3.5-4b\Qwen3.5-4B-Q4_K_M.gguf > D:\Projects\yttri-build\bench4b_yttri.log 2>&1'
Register-ScheduledTask -TaskName "Bench4bYttri" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench4bYttri"
