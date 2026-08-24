$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_mmq_test.bat > D:\Projects\yttri-build\mmq_test.log 2>&1'
Register-ScheduledTask -TaskName "MmqTest" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "MmqTest"
