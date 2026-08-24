$exe = 'D:\Projects\yttri-build\candle-fork-qwen35-batch\target\release\qwen36_inspect.exe'
& $exe 'D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-IQ2_XXS.gguf' > 'D:\Projects\yttri-build\inspect27b_iq.json'
$j = Get-Content -Raw 'D:\Projects\yttri-build\inspect27b_iq.json' | ConvertFrom-Json
$j.tensors | Group-Object dtype | ForEach-Object {
    $bytes = ($_.Group | Measure-Object byte_length -Sum).Sum
    "$($_.Name) x $($_.Count)  =  $([math]::Round($bytes/1GB,2)) GB"
}
