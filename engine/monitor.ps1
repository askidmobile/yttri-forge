# Monitor server: launch, poll every 30s for 10min, capture log on death.
Stop-Process -Name qwen36-server -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Remove-Item 'D:\Projects\yttri-build\live.log' -Force -ErrorAction SilentlyContinue
Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'D:\Projects\yttri-build\run_live_inner.bat >D:\Projects\yttri-build\live.log 2>&1' -WindowStyle Hidden

for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 30
    $p = Get-Process qwen36-server -ErrorAction SilentlyContinue
    if ($p) {
        $mb = [math]::Round($p.WorkingSet64 / 1MB)
        Write-Host "poll $i ($($i*30)s): PID=$($p.Id) RSS=${mb}MB"
    } else {
        Write-Host "poll $i ($($i*30)s): PROCESS DIED"
        Write-Host "=== LOG ==="
        if (Test-Path 'D:\Projects\yttri-build\live.log') {
            Get-Content 'D:\Projects\yttri-build\live.log'
        } else {
            Write-Host "no log file"
        }
        Write-Host "=== END ==="
        exit 0
    }
}
Write-Host "=== STILL ALIVE after 600s ==="
Get-Content 'D:\Projects\yttri-build\live.log' -ErrorAction SilentlyContinue
exit 0