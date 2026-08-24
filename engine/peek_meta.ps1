$j = Get-Content -Raw 'D:\Projects\yttri-build\inspect_model.json' | ConvertFrom-Json
$md = $j.metadata
$md.PSObject.Properties | Where-Object { $_.Name -match 'head|dimension|length|expert|interval|kv' } | ForEach-Object { $_.Name + ' = ' + ($_.Value -join ',') }
