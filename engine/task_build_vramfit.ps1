$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-vramfit\build_vramfit.bat > D:\Projects\yttri-build\build_vramfit.log 2>&1'
Register-ScheduledTask -TaskName "BuildVramfit" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "BuildVramfit"
