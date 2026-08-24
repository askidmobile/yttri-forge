$f = 'D:\Projects\yttri-build\qwen36-server\src\engine.rs'
$raw = Get-Content $f -Raw
$old = "last_logits_t = Some(model.forward(&ids, start)?);"
$new = "last_logits_t = Some(model.forward(&ids, start)?);" + [char]10 + "                eprintln!(" + [char]34 + "[prefill] chunk start={} end={} elapsed={:.1}ms" + [char]34 + ", start, end, _t_pf.elapsed().as_secs_f64() * 1000.0);"
if ($raw -notmatch '\[prefill\] chunk') {
    $raw = $raw -replace [regex]::Escape($old), $new
    Set-Content $f -Value $raw -NoNewline
    Write-Host "PATCHED: added eprintln per chunk"
} else {
    Write-Host "SKIP: prefill chunk eprintln already present"
}