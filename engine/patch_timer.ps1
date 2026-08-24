$f = 'D:\Projects\yttri-build\qwen36-server\src\engine.rs'
$raw = Get-Content $f -Raw
$marker = "let mut last_logits_t: Option<candle_core::Tensor> = None;"
if ($raw -notmatch '_t_pf') {
    $nl = [char]10
    $replacement = $marker + $nl + "            let _t_pf = std::time::Instant::now();"
    $raw = $raw -replace [regex]::Escape($marker), $replacement
    Set-Content $f -Value $raw -NoNewline
    Write-Host "PATCHED: added _t_pf timer"
} else {
    Write-Host "SKIP: _t_pf already present"
}