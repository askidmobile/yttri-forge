# Wait for IQ2_XXS model to load, then curl test.
$logFile = "D:\Projects\yttri-build\iq2xxs.log"
$body = '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Hello, write hello world in Python"}],"max_tokens":64,"stream":false}'
$tmp = "$env:TEMP\q36iq2.json"
$body | Out-File -FilePath $tmp -Encoding utf8 -NoNewline

for ($i = 1; $i -le 14; $i++) {
    Start-Sleep -Seconds 30
    $p = Get-Process qwen36-server -ErrorAction SilentlyContinue
    if ($p) {
        $mb = [math]::Round($p.WorkingSet64 / 1MB)
        Write-Host "poll $i ($($i*30)s): PID=$($p.Id) RSS=${mb}MB"
        $log = Get-Content $logFile -ErrorAction SilentlyContinue
        if ($log -match "loaded:") {
            Write-Host "MODEL LOADED - running curl test..."
            $curl = "C:\Windows\System32\curl.exe"
            & $curl -s -m 120 --noproxy "*" http://127.0.0.1:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d "@$tmp"
            exit 0
        }
        if ($log) { Write-Host "log: $log" }
    } else {
        Write-Host "poll $i PROCESS DIED"
        Write-Host "=== LOG ==="
        Get-Content $logFile -ErrorAction SilentlyContinue
        Write-Host "=== END ==="
        exit 1
    }
}
Write-Host "Still loading after 420s..."
Get-Content $logFile -ErrorAction SilentlyContinue