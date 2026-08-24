$log = 'D:\Projects\yttri-build\vram_27iq_ng.log'
Remove-Item $log -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_bench_nographs.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf > D:\Projects\yttri-build\bench27iq_ng.log 2>&1'
Register-ScheduledTask -TaskName "Bench27IQNG" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27IQNG"
$deadline = (Get-Date).AddSeconds(190)
$max = 0
while ((Get-Date) -lt $deadline) {
    $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader) -replace '\D','')
    if ($used -gt $max) { $max = $used }
    "$(Get-Date -Format HH:mm:ss) vram=${used}MiB" | Out-File $log -Append
    Start-Sleep -Seconds 2
}
"PEAK=$max MiB" | Out-File $log -Append
