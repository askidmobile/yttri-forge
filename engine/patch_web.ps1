$path = 'D:\Projects\yttri-build\qwen36-server\web\index.html'
$f = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)

# 1. Add CSS for markdown elements + typing indicator (before details.think)
$cssOld = 'details.think {'
$cssNew = @'
.typing { color: var(--dim); font-style: italic; }
.msg.assistant .bubble { white-space: normal; }
.msg.assistant .bubble pre { background: var(--bg2); padding: 8px; border-radius: 4px; overflow-x: auto; white-space: pre; margin: 6px 0; }
.msg.assistant .bubble code { background: var(--bg2); padding: 1px 4px; border-radius: 3px; font-family: ui-monospace, "Cascadia Code", monospace; font-size: 13px; }
.msg.assistant .bubble pre code { background: none; padding: 0; }
.msg.assistant .bubble h1, .msg.assistant .bubble h2, .msg.assistant .bubble h3 { margin: 10px 0 4px; color: var(--fg); }
.msg.assistant .bubble h1 { font-size: 18px; } .msg.assistant .bubble h2 { font-size: 16px; } .msg.assistant .bubble h3 { font-size: 15px; }
.msg.assistant .bubble ul, .msg.assistant .bubble ol { margin: 4px 0; padding-left: 22px; }
.msg.assistant .bubble li { margin: 2px 0; }
.msg.assistant .bubble p { margin: 4px 0; }
.msg.assistant .bubble a { color: var(--accent); }
.msg.assistant .bubble blockquote { border-left: 3px solid var(--border); margin: 6px 0; padding-left: 10px; color: var(--dim); }
details.think {
'@
$f = $f.Replace($cssOld, $cssNew)

# 2. Add marked.js CDN + inline fallback before </head>
$headOld = '</head>'
$headNew = @'
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
if (typeof marked !== "undefined") marked.setOptions({ breaks: true, gfm: true });
function mdRender(t) {
  if (typeof marked !== "undefined") {
    try { return marked.parse(t).replace(/<script[^>]*>[\s\S]*?<\/script>/gi,""); } catch(e) {}
  }
  // Fallback: escape + basic formatting
  var h = t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, function(m,l,c){ return '<pre><code>'+c+'</code></pre>'; });
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  h = h.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  h = h.replace(/\n/g, '<br>');
  return h;
}
</script>
</head>
'@
$f = $f.Replace($headOld, $headNew)

# 3. Markdown rendering: change textContent to innerHTML for assistant text parts
$f = $f.Replace('span.textContent = part.text;', 'span.innerHTML = mdRender(part.text);')

# 4. Fix duplicate assistant block: use lastElementChild instead of append
$f = $f.Replace('let msgEl = renderMsg(asst);
  el.append(msgEl);', 'let msgEl = el.lastElementChild;')

# 5. Typing indicator for empty assistant content
$f = $f.Replace('if (msg.role === "assistant") {
    for (const part of parseThinking(msg.content || "")) {', 'if (msg.role === "assistant") {
    if (!(msg.content || "").trim()) {
      const t = document.createElement("span"); t.className = "typing"; t.textContent = "...";
      bubble.append(t);
    } else
    for (const part of parseThinking(msg.content || "")) {')

[System.IO.File]::WriteAllText($path, $f, [System.Text.Encoding]::UTF8)
Write-Host "Patched OK"