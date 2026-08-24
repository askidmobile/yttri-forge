$p = Get-Process qwen36-server -ErrorAction SilentlyContinue
if ($p) {
    $mb = [math]::Round($p.WorkingSet64 / 1MB)
    Write-Host "PID=$($p.Id) RSS=${mb}MB"
} else {
    Write-Host "PROCESS NOT FOUND"
}
$nvidia = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvidia-smi.exe"
if (Test-Path $nvidia) {
    & $nvidia "--query-gpu=memory.used,memory.total" "--format=csv,noheader"
} else {
    Write-Host "nvidia-smi not found at $nvidia"
}