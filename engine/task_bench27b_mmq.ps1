$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c set QWEN36_FORCE_MMQ=1&& D:\Projects\yttri-build\candle-fork-qwen35-batch\run_bench_nographs.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf > D:\Projects\yttri-build\bench27b_mmq.log 2>&1'
Register-ScheduledTask -TaskName "Bench27bMMQ" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27bMMQ"
