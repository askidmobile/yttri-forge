$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c set QWEN36_PERF_FILL_GB=10&& D:\Projects\yttri-build\candle-fork-qwen35-batch\run_mmvq_perf.bat > D:\Projects\yttri-build\mmvq_perf_fill.log 2>&1'
Register-ScheduledTask -TaskName "MmvqPerfFill" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "MmvqPerfFill"
