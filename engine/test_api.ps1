try {
    $body = @{model='qwen3.6-27b'; messages=@(@{role='user'; content='Say hello'}); max_tokens=16; temperature=0; stream=$false} | ConvertTo-Json -Depth 5 -Compress
    $r = Invoke-RestMethod -Uri 'http://localhost:18099/v1/chat/completions' -Method Post -Headers @{Authorization='Bearer smoke-key'; 'Content-Type'='application/json'} -Body $body -TimeoutSec 120
    Write-Output ('OK: ' + $r.choices[0].message.content)
    Write-Output ('Usage: ' + ($r.usage | ConvertTo-Json -Compress))
} catch {
    Write-Output ('ERR: ' + $_.Exception.Message)
}
