$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c D:\Projects\yttri-build\candle-fork-qwen35-batch\run_direct_bench.bat D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf > D:\Projects\yttri-build\bench27b_tg.log 2>&1'
Register-ScheduledTask -TaskName "Bench27bTg" -Action $action -Force | Out-Null
Start-ScheduledTask -TaskName "Bench27bTg"
# Ждём появления строки pp512 в логе (= старт tg фазы), затем семплируем 35с
$deadline = (Get-Date).AddSeconds(320)
$found = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path D:\Projects\yttri-build\bench27b_tg.log) {
        $c = Get-Content D:\Projects\yttri-build\bench27b_tg.log -Raw -ErrorAction SilentlyContinue
        if ($c -match 'CANDLE pp512') { $found = $true; break }
    }
    Start-Sleep -Seconds 3
}
if ($found) {
    $out = @()
    for ($i = 0; $i -lt 18; $i++) {
        $s = nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader
        $out += "$(Get-Date -Format HH:mm:ss) $s"
        Start-Sleep -Seconds 2
    }
    $out | Out-File D:\Projects\yttri-build\tg_util.log
}
