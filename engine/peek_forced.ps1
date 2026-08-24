Get-Content 'D:\Projects\yttri-build\forced-27b.jsonl' -TotalCount 1 | ForEach-Object { $_.Substring(0, [Math]::Min(200, $_.Length)) }
