# pgraph_check.ps1 - Phase 2 gate: capture/replay prefill + parity vs eager.
# Usage: pgraph_check.ps1 [check|on|off] [ctxK] [decodeTokens]
$ErrorActionPreference = "Continue"
$mode = if ($args.Count -gt 0) { $args[0] } else { "check" }
$ctxK = if ($args.Count -gt 1) { [int]$args[1] } else { 2 }
$dec  = if ($args.Count -gt 2) { [int]$args[2] } else { 16 }
$model = "D:\Models\yttri\qwen3.5-4b\Qwen3.5-4B-Q4_K_M.gguf"
$modelName = "qwen3.5-4b"
$port = 18097
$base = "http://localhost:$port"

$env:MODEL = $model
$env:CTX = "16384"; $env:SLOTS = "2"; $env:PORT = "$port"
$env:API_KEYS = '[{"key":"x","name":"pg"}]'
if ($mode -eq "plain") {
  Remove-Item Env:QWEN36_CUDA_GRAPHS -ErrorAction SilentlyContinue
  $env:QWEN36_PGRAPH = "off"
} else {
  $env:QWEN36_CUDA_GRAPHS = "1"
  $env:QWEN36_PGRAPH = $mode
}
Remove-Item Env:MTP -ErrorAction SilentlyContinue
Remove-Item Env:QWEN36_MTP -ErrorAction SilentlyContinue

$out = "$env:TEMP\pg.out"; $err = "$env:TEMP\pg.err"
Remove-Item $out,$err -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath D:\Projects\yttri-inference\target\release\qwen36-server.exe `
  -WorkingDirectory D:\Projects\yttri-inference -RedirectStandardOutput $out `
  -RedirectStandardError $err -WindowStyle Hidden -PassThru

function Wait-Ready($b, $mn) {
  foreach ($i in 1..120) {
    try {
      $probe = @{model=$mn;stream=$false;max_tokens=1;messages=@(@{role="user";content="hi"})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod "$b/v1/chat/completions" -Headers @{Authorization="Bearer x"} -Method Post -ContentType "application/json" -Body $probe -TimeoutSec 120 | Out-Null
      return $true
    } catch { Start-Sleep 3 }
  }
  return $false
}
if (-not (Wait-Ready $base $modelName)) {
  "SERVER NOT READY"; Get-Content $err -Tail 30
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; exit 1
}

$sb = New-Object System.Text.StringBuilder
for ($w=0; $w -lt ($ctxK*500); $w++) { [void]$sb.Append("word$w ") }
$content = $sb.ToString() + " Count from 1 to 20."

# 1) prefill only
$b1 = @{model=$modelName;stream=$false;max_tokens=1;messages=@(@{role="user";content=$content})} | ConvertTo-Json -Depth 5 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\pg1.json", $b1)
$t1 = [double](curl.exe -s -X POST "$base/v1/chat/completions" -H "Content-Type: application/json" -H "Authorization: Bearer x" -d "@$env:TEMP\pg1.json" -o "$env:TEMP\pg1.out" -w "%{time_total}")
$ptok = (Select-String -Path "$env:TEMP\pg1.out" -Pattern '"prompt_tokens":(\d+)' | Select-Object -First 1).Matches[0].Groups[1].Value

# 2) same prompt again -> LRU hit on every chunk
$t2 = [double](curl.exe -s -X POST "$base/v1/chat/completions" -H "Content-Type: application/json" -H "Authorization: Bearer x" -d "@$env:TEMP\pg1.json" -o "$env:TEMP\pg2.out" -w "%{time_total}")

# 3) decode sanity
$b3 = @{model=$modelName;stream=$false;max_tokens=$dec;messages=@(@{role="user";content=$content})} | ConvertTo-Json -Depth 5 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\pg3.json", $b3)
$t3 = [double](curl.exe -s -X POST "$base/v1/chat/completions" -H "Content-Type: application/json" -H "Authorization: Bearer x" -d "@$env:TEMP\pg3.json" -o "$env:TEMP\pg3.out" -w "%{time_total}")

"=== mode=$mode prompt_tokens=$ptok prefill_1=$([math]::Round($t1,2))s prefill_2=$([math]::Round($t2,2))s decode_call=$([math]::Round($t3,2))s ==="
"--- answer ---"
Get-Content "$env:TEMP\pg3.out" -Raw | Select-Object -First 1
"--- [pg] lines ---"
Select-String -Path $err -Pattern '\[pg\]' | ForEach-Object { $_.Line }
"--- errors ---"
Select-String -Path $err -Pattern 'panic|illegal|CUDA_ERROR|ERROR' | Select-Object -First 15 | ForEach-Object { $_.Line }
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
