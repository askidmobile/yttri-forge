$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_nsys_27b.bat > D:\Projects\yttri-build\nsys27b.log 2>&1'
Register-ScheduledTask -TaskName "Nsys27b" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Nsys27b"
