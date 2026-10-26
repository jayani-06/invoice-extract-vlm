"""A deliberately small reviewer page over the review API.

One static HTML file, no build step and no framework, served at `GET /review`.
It only calls the public endpoints, so anything it can do a script or another
UI can do too. It shows, per item, exactly what Phase 5 requires a reviewer to
see: the page image, the extracted value, the retrieved neighbours behind a
category, and the `explain()` text from vendor validation.
"""

REVIEW_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Invoice review queue</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5b6475; --line:#d9dde5;
          --fail:#b42318; --review:#b54708; --pass:#067647; --accent:#2f5bd3; }
  body { margin:0; font:14px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:16px 20px; background:var(--card); border-bottom:1px solid var(--line);
           display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:18px; margin:0 12px 0 0; }
  main { padding:16px 20px; display:grid; gap:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px;
          display:grid; grid-template-columns:minmax(0,320px) minmax(0,1fr); gap:16px; }
  .card img { width:100%; border:1px solid var(--line); border-radius:4px; }
  .tag { display:inline-block; padding:1px 8px; border-radius:10px; font-weight:600; color:#fff; }
  .FAIL { background:var(--fail); } .REVIEW { background:var(--review); } .PASS { background:var(--pass); }
  pre { background:#f1f3f7; padding:8px; border-radius:4px; overflow:auto; max-height:220px; margin:6px 0; }
  table { border-collapse:collapse; width:100%; } td, th { border-bottom:1px solid var(--line);
          padding:3px 6px; text-align:left; }
  .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; align-items:center; }
  button { border:1px solid var(--line); background:#fff; padding:5px 10px; border-radius:4px; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  input, select { padding:5px; border:1px solid var(--line); border-radius:4px; }
  .muted { color:var(--muted); } .err { color:var(--fail); font-weight:600; }
  @media (max-width:760px) { .card { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>Review queue</h1>
  <label>Reviewer <input id="reviewer" placeholder="your name" required></label>
  <button onclick="load()">Refresh</button>
  <span id="status" class="muted"></span>
</header>
<main id="queue"></main>
<script>
let CATEGORIES = [];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const reviewer = () => document.getElementById('reviewer').value.trim();
try { document.getElementById('reviewer').value = localStorage.getItem('reviewer') || ''; } catch (e) {}

async function post(url, body) {
  if (!body.reviewer) { alert('Enter your name first: every decision is recorded against it.'); return; }
  try { localStorage.setItem('reviewer', body.reviewer); } catch (e) {}
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await r.json();
  if (!r.ok) { alert(data.detail || JSON.stringify(data)); return; }
  load();
}

function lineCard(it) {
  const opts = CATEGORIES.filter(c => c.code !== '9999')
    .map(c => `<option value="${c.code}">${c.code} ${esc(c.name)}</option>`).join('');
  const rows = it.neighbours.map(n =>
    `<tr><td>${esc(n.description)}</td><td>${esc(n.category)}</td><td>${esc(n.vendor)}</td><td>${n.score.toFixed(2)}</td></tr>`).join('');
  return `<div class="card"><img src="${it.image_url}" alt="page ${esc(it.doc_id)}">
    <div><div><span class="tag ${it.overall_verdict}">${it.overall_verdict}</span>
      <b>${esc(it.doc_id)}</b> &middot; line ${it.line_index + 1} &middot; category abstained</div>
      <pre>${esc(JSON.stringify(it.extracted, null, 1))}</pre>
      <div>${esc(it.explanation)}</div>
      <table><tr><th>Retrieved neighbour</th><th>Category</th><th>Vendor</th><th>Score</th></tr>
      ${rows || '<tr><td colspan=4 class="muted">no neighbour above the similarity floor</td></tr>'}</table>
      <div class="actions"><select id="cat-${it.item_id}">${opts}</select>
        <button class="primary" onclick="post('/review/lines/${it.item_id}',
          {reviewer:reviewer(), action:'correct', category:document.getElementById('cat-${it.item_id}').value})">
          Set category</button></div></div></div>`;
}

function docCard(it) {
  const checks = it.checks.map(c => `<li><b>${c.verdict}</b> ${esc(c.code)}: ${esc(c.message)}</li>`).join('');
  const fail = it.overall_verdict === 'FAIL';
  const accept = fail
    ? `<input id="why-${it.item_id}" placeholder="reason for override (required)" size="34">
       <button onclick="post('/review/documents/${it.doc_id}', {reviewer:reviewer(), action:'override',
         reason:document.getElementById('why-${it.item_id}').value, vendor_name:document.getElementById('vn-${it.item_id}').value || null})">
         Override FAIL</button>`
    : `<button class="primary" onclick="post('/review/documents/${it.doc_id}', {reviewer:reviewer(), action:'accept',
         vendor_name:document.getElementById('vn-${it.item_id}').value || null})">Accept</button>`;
  const vendor = (it.extracted.vendor || {}).name || '';
  return `<div class="card"><img src="${it.image_url}" alt="page ${esc(it.doc_id)}">
    <div><div><span class="tag ${it.overall_verdict}">${it.overall_verdict}</span> <b>${esc(it.doc_id)}</b></div>
      <div>${esc(it.explanation)}</div>
      ${checks ? `<ul>${checks}</ul>` : ''}
      <pre>${esc(JSON.stringify(it.extracted, null, 1))}</pre>
      ${fail ? '<div class="err">FAIL: not payable without a recorded override.</div>' : ''}
      <div class="actions"><label>Vendor name <input id="vn-${it.item_id}" value="${esc(vendor)}" size="30"></label>
        ${accept}
        <button onclick="post('/review/documents/${it.doc_id}', {reviewer:reviewer(), action:'reject',
          reason:prompt('Reason for rejecting?') || null})">Reject</button></div></div></div>`;
}

async function load() {
  if (!CATEGORIES.length) CATEGORIES = await (await fetch('/categories')).json();
  const items = await (await fetch('/review/queue')).json();
  document.getElementById('status').textContent = items.length + ' open item(s)';
  document.getElementById('queue').innerHTML = items.length
    ? items.map(it => it.kind === 'document' ? docCard(it) : lineCard(it)).join('')
    : '<p class="muted">Nothing to review.</p>';
}
load();
</script>
</body>
</html>
"""
