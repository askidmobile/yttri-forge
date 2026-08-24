$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_llama_bench.bat -m D:\Models\yttri\qwen3.5-4b\Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 1 > D:\Projects\yttri-build\llama4b_yttri.log 2>&1'
Register-ScheduledTask -TaskName "Llama4bYttri" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Llama4bYttri"
