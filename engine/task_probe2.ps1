$action1 = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_direct_bench.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf > D:\Projects\yttri-build\bench27b_probe2.log 2>&1'
$action2 = New-ScheduledTaskAction -Execute "powershell.exe" -Argument '-NoProfile -ExecutionPolicy Bypass -File D:\Projects\yttri-build\candle-fork-qwen35-batch\gpu_shared_probe.ps1'
Register-ScheduledTask -TaskName "Bench27bP2" -Action $action1 -Force | Out-Null
Register-ScheduledTask -TaskName "GpuProbe" -Action $action2 -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27bP2"
Start-Sleep -Seconds 20
Start-ScheduledTask -TaskName "GpuProbe"
