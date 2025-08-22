// static/app.js

// ---------- DOM refs ----------
const form = document.getElementById('scrape-form');
const statusEl = document.getElementById('status');

const progress = document.getElementById('progress');       // needs the markup you added earlier
const bar = progress ? progress.querySelector('.bar') : null;
const label = progress ? progress.querySelector('.label') : null;

const tbody = document.querySelector('#results tbody');
const runBtn = document.getElementById('run-btn');
const dlBtn = document.getElementById('download-csv');

const logBox = document.getElementById('log'); // optional <pre id="log">
function appendLog(line){
  if (logBox) {
    logBox.textContent += line + '\n';
    logBox.scrollTop = logBox.scrollHeight;
  }
  console.log('[SSE log]', line); // always shows in DevTools
}

// ---------- state ----------
let rows = [];
let es = null; // EventSource

// ---------- helpers ----------

function prettifyLog(line, maxLen = 140){
  if (!line) return '';
  // remove leading “[progress] …” if you ever forward those too
  line = line.replace(/^\[progress\]\s*/,'').trim();
  return line.length > maxLen ? line.slice(0, maxLen - 1) + '…' : line;
}

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg || '';
}

function clearTable() {
  if (tbody) tbody.innerHTML = '';
}

function showProgress() {
  if (!progress) return;
  progress.classList.remove('hidden');
}
function hideProgress() {
  if (!progress) return;
  progress.classList.add('hidden');
  progress.classList.remove('indeterminate');
  if (bar) bar.style.width = '0%';
  if (label) label.textContent = '';
}
function setProgressPct(pct, text) {
  if (!progress || !bar) return;
  showProgress();
  progress.classList.remove('indeterminate');
  bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  if (label && text) label.textContent = text;
}
function setIndeterminate(text) {
  if (!progress) return;
  showProgress();
  progress.classList.add('indeterminate');
  if (label && text) label.textContent = text;
}

function renderTable(data) {
  clearTable();
  if (!tbody) return;
  for (const r of data) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${escapeHtml(r.name ?? '')}</td>
      <td>${escapeHtml(r.category ?? '')}</td>
      <td>${fmtMoney(r.price_cad)}</td>
      <td>${fmtNum(r.abv_percent)}</td>
      <td>${fmtInt(r.volume_ml)}</td>
      <td>${fmtNum(r.score)}</td>
      <td>${r.url ? `<a href="${r.url}" target="_blank" rel="noopener">View</a>` : '-'}</td>
    `;
    tbody.appendChild(tr);
  }
}

function fmtMoney(x) { return x == null ? '-' : Number(x).toFixed(2); }
function fmtNum(x)    { return x == null ? '-' : Number(x).toFixed(2); }
function fmtInt(x)    { return x == null ? '-' : Math.round(Number(x)); }

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]
  ));
}

function toCSV(data) {
  const cols = ['name','category','price_cad','abv_percent','volume_ml','score','url'];
  const lines = [cols.join(',')];
  for (const r of data) {
    const row = cols.map(k => `"${String(r[k] ?? '').replace(/"/g,'""')}"`).join(',');
    lines.push(row);
  }
  return lines.join('\n');
}

function downloadCSV() {
  const blob = new Blob([toCSV(rows)], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'nslc_value.csv';
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// ---------- sorting ----------
document.querySelectorAll('#results th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    const isAlreadySorted = th.classList.contains('sorted');
    const wasAsc = th.classList.contains('asc');

    // clear headers
    document.querySelectorAll('#results th').forEach(h => h.classList.remove('sorted','asc','desc'));

    // toggle direction (default score to desc on first click)
    const asc = isAlreadySorted ? !wasAsc : (key !== 'score');
    th.classList.add('sorted', asc ? 'asc' : 'desc');

    rows.sort((a, b) => {
      const va = a[key];
      const vb = b[key];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return asc ? (va > vb ? 1 : va < vb ? -1 : 0)
                 : (va < vb ? 1 : va > vb ? -1 : 0);
    });

    renderTable(rows);
  });
});

function openStream(categories, concurrency) {
  // Close any previous stream
  if (es) { try { es.close(); } catch(_) {} es = null; }

  const qs = new URLSearchParams({
    categories: categories.join(','),
    concurrency: String(concurrency)
  });

  es = new EventSource(`/api/scrape/stream?${qs.toString()}`);

  // --- state ---
  let pctShown   = 0;           // 0–100 we display
  let finished   = false;
  let latestLog  = 'Starting…'; // <-- define it!
  let totalCount = null;        // from progress events (or logs if you parse there)
  let doneCount  = 0;

  // helpers
  const fmtPct = v => `${Math.max(0, Math.min(100, v)).toFixed(1)}%`;
  const setLabel = () => {
    const parts = [fmtPct(pctShown)];
    if (Number.isFinite(doneCount) && Number.isFinite(totalCount) && totalCount) {
      parts.push(`${doneCount}/${totalCount}`);
    }
    if (latestLog) parts.push(latestLog);
    const label = parts.join(' — ');
    setProgressPct(pctShown, label);
    setStatus(latestLog); // status div shows the raw console line
  };

  // initial UI
  setIndeterminate(latestLog);
  setStatus(latestLog);

  // watchdog
  const t0 = Date.now();
  let lastTick = t0;
  const watchdog = setInterval(() => {
    const now = Date.now();
    const sinceStart = (now - t0) / 1000;
    const sinceTick  = (now - lastTick) / 1000;
    if ((sinceStart > 20 && pctShown <= 1) || (sinceTick > 45 && !finished)) {
      try { es.close(); } catch(_) {}
      clearInterval(watchdog);
      fetchResults(categories, concurrency);
    }
  }, 5000);

  // keep watchdog alive but don't overwrite label
  es.addEventListener('status', () => { lastTick = Date.now(); });

  // console lines drive the label
  es.addEventListener('log', (e) => {
    const d = JSON.parse(e.data);
    if (!d.line) return;
    latestLog = String(d.line);
    lastTick = Date.now();
    setLabel();
  });

  // real progress moves the bar (prefer done/total)
  es.addEventListener('progress', (e) => {
    const d = JSON.parse(e.data);
    lastTick = Date.now();

    if (Number.isFinite(d.done) && Number.isFinite(d.total) && d.total > 0) {
      doneCount  = Math.max(doneCount, Number(d.done));
      totalCount = Number(d.total);
      const pct  = (doneCount / totalCount) * 100;
      pctShown   = Math.round(Math.max(pctShown, Math.min(pct, 99)) * 10) / 10;
    } else if (Number.isFinite(d.pct)) {
      pctShown = Math.round(Math.max(pctShown, Math.min(Number(d.pct), 99)) * 10) / 10;
    }

    setLabel();
  });

  es.addEventListener('done', () => {
    finished = true;
    try { es.close(); } catch(_) {}
    es = null;
    clearInterval(watchdog);
    pctShown = 100;
    setProgressPct(100, `100.0% — ${totalCount ? `${doneCount}/${totalCount} — ` : ''}Done — loading results…`);
    setStatus('Done — loading results…');
    fetchResults(categories, concurrency);
  });

  es.addEventListener('error', () => {
    try { es.close(); } catch(_) {}
    es = null;
    clearInterval(watchdog);
    fetchResults(categories, concurrency);
  });
}


async function fetchResults(categories, concurrency) {
  try {
    const res = await fetch('/api/scrape', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ categories, concurrency })
    });

    if (!res.ok) {
      let msg = await res.text();
      try {
        const j = JSON.parse(msg);
        msg = j.detail || j.message || msg;
      } catch (_) {}
      throw new Error(`${res.status} ${msg}`);
    }

    const data = await res.json();
    rows = data.items || [];
    renderTable(rows);
    setStatus(`${data.cached ? 'Loaded from cache' : 'Fresh scrape'} — ${rows.length} items`);
    dlBtn.disabled = rows.length === 0;
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + (err?.message || String(err)));
  } finally {
    hideProgress();
    runBtn.disabled = false;
  }
}

// ---------- main submit handler ----------
form.addEventListener('submit', (e) => {
  e.preventDefault();

  const categories = Array.from(document.querySelectorAll("input[name='category']:checked"))
    .map(cb => cb.value);
  const concurrency = Number(document.getElementById('concurrency')?.value) || 5;

  if (categories.length === 0) {
    setStatus('Choose at least one category.');
    return;
  }

  setStatus('Starting… this may take several minutes on a fresh run.');
  runBtn.disabled = true;
  dlBtn.disabled = true;
  clearTable();

  // Stream progress; when stream says "done", we fetch the results.
  openStream(categories, concurrency);
});

// ---------- CSV download ----------
dlBtn.addEventListener('click', downloadCSV);
