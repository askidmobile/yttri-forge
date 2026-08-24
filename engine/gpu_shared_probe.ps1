$ErrorActionPreference = 'SilentlyContinue'
$log = 'D:\Projects\yttri-build\gpu_shared.log'
Remove-Item $log -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddSeconds(200)
while ((Get-Date) -lt $deadline) {
    $p = Get-Process bench_candle_direct -ErrorAction SilentlyContinue
    if ($p) {
        $counters = Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
        $sum = 0
        if ($counters) {
            foreach ($c in $counters.CounterSamples) { if ($c.InstanceName -match "bench_candle") { $sum += $c.CookedValue } }
        }
        $ded = Get-Counter '\GPU Process Memory(*)\Non Dedicated Usage' -ErrorAction SilentlyContinue
        $dedsum = 0
        if ($ded) {
            foreach ($c in $ded.CounterSamples) { if ($c.InstanceName -match "bench_candle") { $dedsum += $c.CookedValue } }
        }
        "$(Get-Date -Format HH:mm:ss) pid=$($p.Id) ramMB=$([math]::Round($p.WorkingSet64/1MB)) sharedGPU_MB=$([math]::Round($sum/1MB)) nonDedicated_MB=$([math]::Round($dedsum/1MB))" | Out-File $log -Append
    }
    Start-Sleep -Seconds 5
}
