$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_llama_bench.bat -m D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf -ngl 99 -p 512 -n 128 -r 1 > D:\Projects\yttri-build\llama27b_iq.log 2>&1'
Register-ScheduledTask -TaskName "Llama27bIQ" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Llama27bIQ"
