# Launch server with stderr capture, poll 30s for 10min, capture log on death.
# Uses ProcessStart with redirect to capture ALL output including stderr.
Stop-Process -Name qwen36-server -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

$exe = "D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;" + $env:PATH
$env:QWEN36_MODEL = "D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf"
$env:QWEN36_SLOTS = "1"
$env:QWEN36_CTX = "2048"
$env:QWEN36_PORT = "18099"
$env:QWEN36_API_KEY = "test"
$env:RUST_BACKTRACE = "full"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true

$proc = [System.Diagnostics.Process]::Start($psi)
Write-Host "Started PID=$($proc.Id)"

$logFile = "D:\Projects\yttri-build\capture.log"
"" | Out-File $logFile -Encoding utf8

$script = {
    param($proc, $logFile)
    while (-not $proc.StandardOutput.EndOfStream) {
        $line = $proc.StandardOutput.ReadLine()
        Add-Content -Path $logFile -Value $line -Encoding utf8
    }
}
$errScript = {
    param($proc, $logFile)
    while (-not $proc.StandardError.EndOfStream) {
        $line = $proc.StandardError.ReadLine()
        Add-Content -Path $logFile -Value "[STDERR] $line" -Encoding utf8
    }
}

$job1 = Start-Job -ScriptBlock $script -ArgumentList $proc, $logFile
$job2 = Start-Job -ScriptBlock $errScript -ArgumentList $proc, $logFile

for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 30
    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($p) {
        $mb = [math]::Round($p.WorkingSet64 / 1MB)
        Write-Host "poll $i ($($i*30)s): PID=$($p.Id) RSS=${mb}MB"
    } else {
        Write-Host "poll $i ($($i*30)s): PROCESS DIED (exit=$($proc.ExitCode))"
        Write-Host "=== LOG ==="
        Get-Content $logFile
        Write-Host "=== END ==="
        Stop-Job $job1, $job2 -ErrorAction SilentlyContinue
        Remove-Job $job1, $job2 -Force -ErrorAction SilentlyContinue
        exit 0
    }
}
Write-Host "=== STILL ALIVE after 600s ==="
Get-Content $logFile
Stop-Job $job1, $job2 -ErrorAction SilentlyContinue
Remove-Job $job1, $job2 -Force -ErrorAction SilentlyContinue
exit 0