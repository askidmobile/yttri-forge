$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c set QWEN36_PREFILL_CHUNK=128&& D:\Projects\yttri-build\candle-fork-qwen35-batch\run_direct_bench.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf > D:\Projects\yttri-build\bench27iq_c128.log 2>&1'
Register-ScheduledTask -TaskName "Bench27IQC128" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27IQC128"
$deadline = (Get-Date).AddSeconds(190)
$max = 0
while ((Get-Date) -lt $deadline) {
    $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader) -replace '\D','')
    if ($used -gt $max) { $max = $used }
    Start-Sleep -Seconds 3
}
"PEAK=$max MiB" | Out-File D:\Projects\yttri-build\vram_c128_peak.log
