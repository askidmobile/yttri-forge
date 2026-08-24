$base = @{}
Get-Content 'D:\Projects\yttri-build\logits-4b-baseline128.jsonl' | ForEach-Object {
    if ($_.StartsWith('{')) {
        $r = $_ | ConvertFrom-Json
        if ($r.type -eq 'logits') { $base[[int]$r.step] = $r.checksum }
    }
}
$head = @{}
Get-Content 'D:\Projects\yttri-build\logits-4b-head128.jsonl' | ForEach-Object {
    if ($_.StartsWith('{')) {
        $r = $_ | ConvertFrom-Json
        if ($r.type -eq 'logits') { $head[[int]$r.step] = $r.checksum }
    }
}
$mismatch = 0
$first = -1
foreach ($s in ($base.Keys | Sort-Object)) {
    if ($head[$s] -ne $base[$s]) {
        $mismatch++
        if ($first -lt 0) { $first = $s }
        Write-Host ("step {0}: baseline={1} head={2}" -f $s, $base[$s], $head[$s])
    }
}
Write-Host ("steps={0} mismatches={1} first={2}" -f $base.Count, $mismatch, $first)
