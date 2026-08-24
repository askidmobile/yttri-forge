$ids = (Get-Content 'D:\Projects\yttri-build\llama-forced-tokens.txt' -Raw).Trim() -split '\s+'
$arr = @($ids | ForEach-Object { [int]$_ })
$json = '{"type":"tokens","ids":[' + ($arr -join ',') + ']}'
[System.IO.File]::WriteAllText('D:\Projects\yttri-build\forced-27b.jsonl', $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
Write-Host "wrote $($arr.Count) tokens, no BOM"
