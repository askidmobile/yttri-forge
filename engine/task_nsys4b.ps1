$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_nsys_4b.bat > D:\Projects\yttri-build\nsys4b.log 2>&1'
Register-ScheduledTask -TaskName "Nsys4b" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Nsys4b"
