$j = Get-Content -Raw 'D:\Projects\yttri-build\inspect4b.json' | ConvertFrom-Json
$j.tensors | Group-Object dtype | ForEach-Object {
    $bytes = ($_.Group | Measure-Object byte_length -Sum).Sum
    "$($_.Name) x $($_.Count)  =  $([math]::Round($bytes/1GB,3)) GB"
}