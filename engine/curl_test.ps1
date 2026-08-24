$body = '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Напиши hello world на Python с markdown"}],"max_tokens":100,"stream":false}'
$tmp = "$env:TEMP\q36test.json"
$body | Out-File -FilePath $tmp -Encoding utf8 -NoNewline
$curl = "C:\Windows\System32\curl.exe"
& $curl -v -s -m 600 --noproxy "*" http://127.0.0.1:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d "@$tmp"
