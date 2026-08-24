$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_nsys_llama.bat > D:\Projects\yttri-build\nsys_llama.log 2>&1'
Register-ScheduledTask -TaskName "NsysLlama" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "NsysLlama"
