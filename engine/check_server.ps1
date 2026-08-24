Start-Sleep -Seconds 120
$p = Get-Process qwen36-server -ErrorAction SilentlyContinue
if ($p) {
    $mb = [math]::Round($p.WorkingSet64 / 1MB)
    Write-Host "PID=$($p.Id) RSS=${mb}MB"
} else {
    Write-Host "PROCESS NOT FOUND"
}
Write-Host "---LOG---"
if (Test-Path 'D:\Projects\yttri-build\live.log') {
    $sz = (Get-Item 'D:\Projects\yttri-build\live.log').Length
    Write-Host "live.log size=$sz bytes"
    Get-Content 'D:\Projects\yttri-build\live.log' -Tail 10
} else {
    Write-Host "no live.log"
}