$ErrorActionPreference = 'Stop'
$exe = 'D:\Projects\yttri-build\candle-fork-qwen35-batch\target\release\qwen36_inspect.exe'
$model = $args[0]
$out = 'D:\Projects\yttri-build\inspect_model.json'
& $exe $model > $out
$j = Get-Content -Raw $out | ConvertFrom-Json
$t = $j.tensors
$byBlock = @{}
foreach ($x in $t) {
    if ($x.name -match '^blk\.(\d+)\.') {
        $n = [int]$Matches[1]
        if (-not $byBlock[$n]) { $byBlock[$n] = @{ iq = $false; bytes = 0 } }
        if ($x.dtype -like 'IQ*') { $byBlock[$n].iq = $true }
        $byBlock[$n].bytes += [long]$x.byte_length
    }
}
$iqBlocks = @($byBlock.Keys | Where-Object { $byBlock[$_].iq } | Sort-Object)
$free = @($byBlock.Keys | Where-Object { -not $byBlock[$_].iq } | Sort-Object)
$totalIqFree = 0
$line = foreach ($n in $free) { $totalIqFree += $byBlock[$n].bytes; "$n($([math]::Round($byBlock[$n].bytes/1MB))MB)" }
Write-Output ("IQ-BLOCKS count=" + $iqBlocks.Count + ": " + ($iqBlocks -join ','))
Write-Output ("IQ-FREE count=" + $free.Count + " total=" + [math]::Round($totalIqFree/1GB,2) + "GB")
Write-Output ($line -join ' ')
