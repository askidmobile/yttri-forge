$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_nsys_27iq.bat > D:\Projects\yttri-build\nsys27iq.log 2>&1'
Register-ScheduledTask -TaskName "Nsys27IQ" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Nsys27IQ"
