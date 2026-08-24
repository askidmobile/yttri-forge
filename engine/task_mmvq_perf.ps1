$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_mmvq_perf.bat > D:\Projects\yttri-build\mmvq_perf.log 2>&1'
Register-ScheduledTask -TaskName "MmvqPerf" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "MmvqPerf"
