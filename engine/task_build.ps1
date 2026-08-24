$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\build_bench.bat > D:\Projects\yttri-build\build.log 2>&1'
Register-ScheduledTask -TaskName "CandleBuild" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "CandleBuild"
