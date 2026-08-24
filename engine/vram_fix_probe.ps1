$log = 'D:\Projects\yttri-build\bench27iq_fix_peak.log'
$benchlog = 'D:\Projects\yttri-build\bench27iq_fix.log'
Remove-Item $log, $benchlog -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c set RUST_LOG=info&& D:\Projects\yttri-build\candle-fork-qwen35-batch\run_direct_bench.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf > D:\Projects\yttri-build\bench27iq_fix.log 2>&1'
Register-ScheduledTask -TaskName "Bench27IQFix" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27IQFix"
$deadline = (Get-Date).AddSeconds(200)
$max = 0
while ((Get-Date) -lt $deadline) {
    $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader) -replace '\D','')
    if ($used -gt $max) { $max = $used }
    Start-Sleep -Seconds 2
}
"PEAK=$max MiB" | Out-File $log
