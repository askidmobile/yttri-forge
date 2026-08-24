$log = 'D:\Projects\yttri-build\vram_27iq_timeline.log'
Remove-Item $log -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_direct_bench.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf > D:\Projects\yttri-build\bench27iq_vram.log 2>&1'
Register-ScheduledTask -TaskName "Bench27IQVram" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27IQVram"
# Семплируем VRAM каждые 2с до ~180с, с меткой фазы из лога
$deadline = (Get-Date).AddSeconds(190)
while ((Get-Date) -lt $deadline) {
    $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader) -replace '\D',''
    $phase = ''
    if (Test-Path D:\Projects\yttri-build\bench27iq_vram.log) {
        $c = Get-Content D:\Projects\yttri-build\bench27iq_vram.log -Raw -ErrorAction SilentlyContinue
        if ($c -match 'CANDLE pp512') { $phase = 'tg/decode' }
        elseif ($c -match 'batched KV cache') { $phase = 'loaded/prefill' }
    }
    "$(Get-Date -Format HH:mm:ss) vram=${used}MiB $phase" | Out-File $log -Append
    Start-Sleep -Seconds 2
}
