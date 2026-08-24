$j = Get-Content -Raw 'D:\Projects\yttri-build\inspect_model.json' | ConvertFrom-Json
$j.tensors | Where-Object { $_.name -match '^blk\.(3|15|60|61|62|63)\.' -and $_.name -like '*.weight' } | ForEach-Object { $_.name + '  ' + $_.dtype }
