$f = 'D:\Projects\yttri-build\candle-fork-qwen35-batch\qwen35-batch\src\real\model_weights.rs'
$lines = Get-Content $f
# The CUDA prefill block is the SECOND occurrence of 'if let Some(ctx) = &self.cuda_ctx {'
# First (line ~1586) is in forward() decode path. Second (line ~2082) is in forward_prefill().
$occurrence = 0
$startIdx = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'if let Some\(ctx\) = &self\.cuda_ctx \{') {
        $occurrence++
        if ($occurrence -eq 2) {
            $startIdx = $i
            break
        }
    }
}
if ($startIdx -lt 0) { Write-Host "ERROR: CUDA prefill block not found"; exit 1 }
# Find closing brace: the block ends at a line that is exactly "        }" at same indent
$braceDepth = 0
for ($i = $startIdx; $i -lt $lines.Count; $i++) {
    $opens = ([regex]::Matches($lines[$i], '\{')).Count
    $closes = ([regex]::Matches($lines[$i], '\}')).Count
    $braceDepth += $opens - $closes
    if ($braceDepth -le 0 -and $i -gt $startIdx) {
        $endIdx = $i
        break
    }
}
if ($endIdx -lt 0) { Write-Host "ERROR: closing brace not found"; exit 1 }
Write-Host "CUDA prefill block: lines $($startIdx+1) to $($endIdx+1)"
# Comment out lines startIdx..endIdx with //
for ($i = $startIdx; $i -le $endIdx; $i++) {
    if ($lines[$i].Trim() -ne '') {
        $lines[$i] = '        // DISABLED CPU-FB-TEST: ' + $lines[$i].TrimStart()
    }
}
Set-Content $f -Value $lines
Write-Host "PATCHED: disabled CUDA DeltaNet prefill (CPU fallback active)"