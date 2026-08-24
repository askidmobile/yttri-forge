$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_direct_bench.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf > D:\Projects\yttri-build\bench27b_iq.log 2>&1'
Register-ScheduledTask -TaskName "Bench27bIQ" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27bIQ"
