$p = Get-Process qwen36-server -ErrorAction SilentlyContinue
if ($p) {
    $mb = [math]::Round($p.WorkingSet64 / 1MB)
    Write-Host "PID=$($p.Id) RSS=${mb}MB"
} else {
    Write-Host "PROCESS NOT FOUND"
}
# Try multiple nvidia-smi locations
$smi = @(
    "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "C:\Windows\System32\nvidia-smi.exe"
)
foreach ($s in $smi) {
    if (Test-Path $s) {
        Write-Host "nvidia-smi: $s"
        & $s "--query-gpu=memory.used,memory.total" "--format=csv,noheader"
        break
    }
}
# curl test
$body = '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":false}'
$body | Out-File -FilePath "$env:TEMP\test.json" -Encoding utf8 -NoNewline
$curl = "C:\Windows\System32\curl.exe"
if (Test-Path $curl) {
    & $curl -s -m 10 --noproxy "*" http://127.0.0.1:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d "@$env:TEMP\test.json"
}