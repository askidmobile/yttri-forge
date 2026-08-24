$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_bench_nographs.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf > D:\Projects\yttri-build\bench27b_nographs.log 2>&1'
Register-ScheduledTask -TaskName "Bench27bNG" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27bNG"
