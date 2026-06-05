#!/usr/bin/env python3
"""Read dmdd_data.json and render a self-contained HTML dashboard.

Features:
  - Native <details>/<summary> collapse per experiment (no JS dependency for collapse)
  - Status badges (color-coded)
  - Search bar (vanilla JS filters experiment cards by name/tag/purpose)
  - Comparison view (checkboxes + 'Compare selected' button -> popup with side-by-side)
  - Anchor links for deep-linking to specific experiments
  - Copy-to-clipboard for tables
  - Embedded image support (base64 PNG, expandable per experiment)
  - Last-updated timestamp
"""
import argparse
import base64
import html
import json
import os
from datetime import datetime
from pathlib import Path


STATUS_COLORS = {
    "done": "#3d8b3d",
    "eval_in_progress": "#d6a000",
    "running": "#d6a000",
    "queued": "#6c757d",
    "failed": "#c33",
}

# Each experiment gets a distinct light bg tint so rows + cards from the
# same experiment are visually grouped throughout the dashboard.
EXPERIMENT_TINTS = {
    "v24": "#f3e8ff", "v23": "#ffe8e8", "v22": "#fff7da",
    "v21": "#e8f5ff", "v20": "#d9f5e1", "v19": "#fff0d8",
    "v17": "#ebebfa", "v16": "#ffe0f0", "v15": "#dceefd",
    "v13": "#fff3dc", "v12": "#ececec",
}


def exp_tint_style(exp_id):
    c = EXPERIMENT_TINTS.get(exp_id)
    return f' style="background:{c}"' if c else ""


def exp_row_class(exp_id):
    return f"row-{exp_id}"


def latest_first_key(exp_id):
    """Sort key: pull numeric id, descending. v24 < v12 (so v24 comes first when ascending sort)."""
    try:
        n = int(''.join(c for c in exp_id if c.isdigit()))
        return -n
    except Exception:
        return 0


def esc(s):
    if s is None: return ""
    return html.escape(str(s))


def img_b64(path):
    if not path or not os.path.isfile(path): return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def render_kv_table(d):
    """Render a flat key-value dict as a 2-column table."""
    rows = []
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v) if isinstance(v, dict) else "; ".join(map(str, v))
        rows.append(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>")
    return f'<table class="kv">{"".join(rows)}</table>'


def render_results_table(results, headers=None):
    """Render a list of result-dicts as a table.

    If headers is None, infer from keys of first dict (or take union).
    Auto-formats floats and adds class tags for hits_target.
    """
    if not results: return "<p><em>(no results)</em></p>"
    if headers is None:
        # Stable union of keys, preferring first row's order
        seen = []
        for r in results:
            for k in r.keys():
                if k not in seen and k != "hits_target":
                    seen.append(k)
        headers = seen
    out = ['<table class="results">']
    out.append("<thead><tr>")
    for h in headers:
        out.append(f"<th>{esc(h)}</th>")
    out.append("<th>copy</th></tr></thead>")
    out.append("<tbody>")
    for r in results:
        row_class = ""
        if r.get("hits_target") is True: row_class = ' class="hit"'
        out.append(f"<tr{row_class}>")
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                # Recognize percent-style
                if h.upper() in ("R", "P") or "_R" in h.upper() or "_P" in h.upper() or h.endswith("_R") or h.endswith("_P"):
                    v_str = f"{v:.2f}%"
                else:
                    v_str = f"{v:.4f}" if v < 1 else f"{v:.2f}"
            else:
                v_str = esc(v)
            out.append(f"<td>{v_str}</td>")
        # Copy button
        csv_row = ",".join(str(r.get(h, "")) for h in headers)
        out.append(f'<td><button class="copy-btn" data-text="{esc(csv_row)}">📋</button></td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def status_badge(status):
    color = STATUS_COLORS.get(status, "#888")
    label = status.upper().replace("_", " ")
    return f'<span class="badge" style="background:{color}">{esc(label)}</span>'


def render_experiment(exp):
    eid = exp["id"]
    name = exp["name"]
    status = exp.get("status", "unknown")
    tags = exp.get("tags", [])
    tag_html = "".join(f'<span class="tag">{esc(t)}</span>' for t in tags)

    purpose = exp.get("purpose", "")
    training_data = exp.get("training_data", {})
    recipe = exp.get("recipe", {})
    results = exp.get("results", [])
    stage1 = exp.get("stage1_standalone")
    cascade_results = exp.get("cascade_results", [])
    per_class_full_tta_T = exp.get("per_class_full_tta_T")
    per_class_no_tta_T = exp.get("per_class_no_tta_T")
    caveats = exp.get("caveats", "")
    pending = exp.get("pending", [])
    takeaway = exp.get("takeaway", "")

    # Compute searchable text
    search_text = " ".join([name, purpose, " ".join(tags), eid, status]).lower()

    tint = exp_tint_style(eid)
    parts = []
    parts.append(f'<details class="exp" id="exp-{esc(eid)}" data-search="{esc(search_text)}" data-id="{esc(eid)}" data-status="{esc(status)}"{tint}>')
    parts.append(
        f'<summary>'
        f'{status_badge(status)}'
        f'<span class="exp-name">{esc(name)}</span>'
        f'{tag_html}'
        f'<a class="anchor" href="#exp-{esc(eid)}" title="link">🔗</a>'
        f'</summary>'
    )

    if purpose:
        parts.append(f'<div class="section"><h4>Purpose</h4><p>{esc(purpose)}</p></div>')
    if training_data:
        parts.append(f'<div class="section"><h4>Training data</h4>{render_kv_table(training_data)}</div>')
    if recipe:
        parts.append(f'<div class="section"><h4>Recipe</h4>{render_kv_table(recipe)}</div>')
    if results:
        parts.append(f'<div class="section"><h4>Results</h4>{render_results_table(results)}</div>')
    if stage1:
        parts.append(f'<div class="section"><h4>Stage-1 standalone (tile-level)</h4>{render_kv_table(stage1)}</div>')
    if cascade_results:
        parts.append(f'<div class="section"><h4>Cascade results</h4>{render_results_table(cascade_results)}</div>')
    if per_class_full_tta_T:
        parts.append(f'<div class="section"><h4>Per-class (full-TTA + T*)</h4>{render_results_table(per_class_full_tta_T)}</div>')
    if per_class_no_tta_T:
        parts.append(f'<div class="section"><h4>Per-class (no-TTA + T*)</h4>{render_results_table(per_class_no_tta_T)}</div>')
    if pending:
        parts.append('<div class="section"><h4>Pending</h4><ul>' + "".join(f"<li>{esc(p)}</li>" for p in pending) + "</ul></div>")
    if caveats:
        parts.append(f'<div class="section caveat"><h4>Caveats</h4><p>{esc(caveats)}</p></div>')
    if takeaway:
        parts.append(f'<div class="section"><h4>Take-away</h4><p>{esc(takeaway)}</p></div>')

    parts.append("</details>")
    return "\n".join(parts)


def render_comparison(comp):
    cid = comp["id"]
    title = comp["title"]
    purpose = comp.get("purpose", "")
    table = comp.get("table", [])
    takeaway = comp.get("takeaway", "")
    search_text = (title + " " + purpose + " " + takeaway).lower()
    parts = [f'<details class="exp" id="cmp-{esc(cid)}" data-search="{esc(search_text)}">']
    parts.append(f'<summary><span class="exp-name">{esc(title)}</span><a class="anchor" href="#cmp-{esc(cid)}">🔗</a></summary>')
    if purpose: parts.append(f'<div class="section"><h4>Purpose</h4><p>{esc(purpose)}</p></div>')
    if table:   parts.append(f'<div class="section"><h4>Table</h4>{render_results_table(table)}</div>')
    if takeaway:parts.append(f'<div class="section"><h4>Take-away</h4><p>{esc(takeaway)}</p></div>')
    parts.append("</details>")
    return "\n".join(parts)


def render_analysis(an):
    aid = an["id"]; title = an["title"]; summary = an.get("summary", ""); table = an.get("table", [])
    search_text = (title + " " + summary).lower()
    parts = [f'<details class="exp" id="an-{esc(aid)}" data-search="{esc(search_text)}">']
    parts.append(f'<summary><span class="exp-name">{esc(title)}</span><a class="anchor" href="#an-{esc(aid)}">🔗</a></summary>')
    if summary: parts.append(f'<div class="section"><p>{esc(summary)}</p></div>')
    if table:   parts.append(f'<div class="section">{render_results_table(table)}</div>')
    parts.append("</details>")
    return "\n".join(parts)


CSS = r"""
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       margin: 0; padding: 0; background: #f7f8fa; color: #222;
       display: grid; grid-template-columns: 230px 1fr; min-height: 100vh; }
aside.sidebar { position: sticky; top: 0; align-self: start; height: 100vh; overflow-y: auto;
                background: #ffffff; border-right: 1px solid #d0d7de; padding: 18px 14px;
                font-size: 13px; box-shadow: inset -1px 0 0 rgba(0,0,0,0.02); }
aside.sidebar h3 { margin: 0 0 6px 0; font-size: 11px; color: #57606a; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700; }
aside.sidebar .nav-section { margin-bottom: 14px; }
aside.sidebar ul { list-style: none; padding: 0; margin: 0; }
aside.sidebar li { margin: 0; }
aside.sidebar li a { display: block; padding: 4px 8px; border-radius: 5px; text-decoration: none; color: #24292f; }
aside.sidebar li a:hover { background: #f0f4ff; color: #0969da; }
aside.sidebar li.indent a { padding-left: 18px; font-size: 12.5px; color: #57606a; }
aside.sidebar li.indent a:hover { color: #0969da; }
aside.sidebar .exp-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: middle; border: 1px solid #00000020; }
main.main-content { padding: 24px 32px; min-width: 0; }
h1 { margin: 0 0 4px 0; font-size: 28px; }
h2 { margin-top: 36px; padding-bottom: 6px; border-bottom: 2px solid #d0d7de; font-size: 22px; scroll-margin-top: 12px; }
details.exp { scroll-margin-top: 12px; }
.section-anchor { scroll-margin-top: 12px; }

/* Bullseye target */
.bullseye-wrap { background: white; border: 1px solid #d0d7de; border-radius: 12px;
                 padding: 18px; margin: 14px 0 20px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
.bullseye-wrap .header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.bullseye-wrap h3 { margin: 0; font-size: 16px; }
.bullseye-wrap .target-hint { color: #57606a; font-size: 12px; }
.bullseye-svg { width: 100%; max-width: 720px; height: auto; display: block; margin: 0 auto; }
.bullseye-legend { font-size: 12px; color: #444; margin-top: 8px; display: flex; flex-wrap: wrap; gap: 14px; justify-content: center; }
.bullseye-legend .lg-item { display: inline-flex; align-items: center; gap: 5px; }
.bullseye-legend .lg-dot { width: 10px; height: 10px; border-radius: 50%; border: 1px solid #00000040; }

/* Bar charts */
.bar-charts-wrap { display: grid; grid-template-columns: 1fr; gap: 14px; margin: 8px 0 18px 0; }
.bar-chart-card { background: white; border: 1px solid #d0d7de; border-radius: 10px; padding: 14px 18px; }
.bar-chart-card h4 { margin: 0 0 6px 0; font-size: 13px; color: #24292f; text-transform: none; letter-spacing: 0; font-weight: 600; }
.bar-chart-card .bc-hint { color: #57606a; font-size: 11.5px; margin: 0 0 8px 0; }
.bar-chart-svg { width: 100%; height: auto; display: block; }

@media (max-width: 900px) {
  body { grid-template-columns: 1fr; }
  aside.sidebar { position: relative; height: auto; max-height: 280px; }
  main.main-content { padding: 18px; }
}

h4 { margin: 12px 0 6px 0; font-size: 14px; color: #57606a; text-transform: uppercase; letter-spacing: 0.5px; }
.meta { color: #57606a; font-size: 13px; margin-bottom: 24px; }
.controls { background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; position: sticky; top: 0; z-index: 100; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
.controls input[type=search] { width: 50%; padding: 8px 12px; border: 1px solid #d0d7de; border-radius: 6px; font-size: 14px; }
.controls button { padding: 8px 12px; border: 1px solid #d0d7de; border-radius: 6px; background: white; cursor: pointer; font-size: 14px; margin-left: 6px; }
.controls button:hover { background: #f3f4f6; }
.controls button.primary { background: #0969da; color: white; border-color: #0969da; }
.controls button.primary:hover { background: #0758c3; }

details.exp { background: white; border: 1px solid #d0d7de; border-radius: 8px; margin-bottom: 8px; padding: 0; }
details.exp[open] { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
details.exp summary { padding: 12px 14px; cursor: pointer; display: flex; align-items: center; gap: 8px; list-style: none; }
details.exp summary::-webkit-details-marker { display: none; }
details.exp summary:hover { background: #f6f8fa; }
details.exp summary::before { content: "▶"; font-size: 10px; color: #57606a; transition: transform 0.15s; margin-right: 2px; }
details.exp[open] > summary::before { transform: rotate(90deg); }
.exp-name { font-weight: 600; flex-grow: 1; }
.anchor { text-decoration: none; color: #57606a; opacity: 0.4; font-size: 12px; }
.anchor:hover { opacity: 1; }

.badge { display: inline-block; padding: 3px 8px; border-radius: 10px; color: white; font-size: 11px; font-weight: 600; text-transform: uppercase; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; background: #e8eaf0; color: #57606a; font-size: 11px; margin-left: 2px; }

.section { padding: 8px 16px 12px 16px; }
.section p { margin: 4px 0; line-height: 1.5; }
.section.caveat { background: #fff8e1; border-left: 3px solid #f0a000; }

table { border-collapse: collapse; margin: 4px 0; font-size: 13px; }
table.kv { width: 100%; }
table.kv th { text-align: left; padding: 4px 8px; color: #57606a; font-weight: normal; width: 30%; vertical-align: top; }
table.kv td { padding: 4px 8px; }
table.kv tr:nth-child(even) { background: #f6f8fa; }
table.results { border: 1px solid #d0d7de; width: 100%; }
table.results th { background: #f6f8fa; padding: 6px 10px; text-align: left; border-bottom: 1px solid #d0d7de; font-size: 12px; }
table.results td { padding: 5px 10px; border-bottom: 1px solid #eaeef2; }
table.results tr.hit { background: #e7f5e3; }
table.results tr.hit td { font-weight: 600; }
table.results tr.pinned td:first-child { color: #c00; font-weight: 700; }
table.results tr.pinned { border-left: 3px solid #0969da; }
table.results tr.deprio td { color: #888; }
table.results tr.deprio { opacity: 0.7; }
table.sortable th.sort-col { cursor: pointer; user-select: none; }
table.sortable th.sort-col:hover { background: #e8eaf0; }
table.sortable th.sort-col::after { content: " ⇅"; color: #aaa; font-size: 10px; }
table.sortable th.sort-col.sort-asc::after  { content: " ↑"; color: #0969da; }
table.sortable th.sort-col.sort-desc::after { content: " ↓"; color: #0969da; }

/* Domain-card grouping with subtle background shading */
.domain-card { border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; border: 1px solid #d0d7de; }
.domain-card .domain-header { margin: 0 0 8px 0; font-size: 16px; padding-bottom: 6px; border-bottom: 1px solid; }
.team-card     { background: #eaf2fb; }            /* light blue */
.team-card .domain-header     { border-color: #c0d4ec; color: #1a4a8a; }
.team-card table.results th   { background: #d6e6f7; }
.overseas-card { background: #fdf0e6; }            /* light orange */
.overseas-card .domain-header { border-color: #f0d2b8; color: #8a4a1a; }
.overseas-card table.results th { background: #f9dec5; }
.combined-card { background: #ecf6e8; }            /* light green */
.combined-card .domain-header { border-color: #cce5c2; color: #2a6a2a; }
.combined-card table.results th { background: #dbeed2; }
.domain-card table.results tr.hit { background: #c4ebc0 !important; }
.domain-card table.results tr.pinned { border-left: 4px solid #0969da; }
.domain-card table.results { background: white; }
.copy-btn { background: none; border: none; cursor: pointer; opacity: 0.4; font-size: 12px; }
.copy-btn:hover { opacity: 1; }

#compare-modal { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 1000; padding: 40px; overflow: auto; }
#compare-modal.open { display: block; }
#compare-modal .modal-content { background: white; border-radius: 8px; padding: 24px; max-width: 1200px; margin: 0 auto; }
#compare-modal .close { float: right; cursor: pointer; font-size: 20px; color: #57606a; }
.hidden { display: none; }

footer { margin-top: 32px; padding: 16px 0; color: #57606a; font-size: 12px; text-align: center; }
"""

JS = r"""
(function(){
  // Search filter
  const search = document.getElementById('search');
  search.addEventListener('input', () => {
    const q = search.value.trim().toLowerCase();
    document.querySelectorAll('details.exp').forEach(el => {
      const t = el.dataset.search || '';
      el.style.display = (!q || t.includes(q)) ? '' : 'none';
    });
  });

  // Status filter
  document.querySelectorAll('.status-filter').forEach(btn => {
    btn.addEventListener('click', () => {
      const want = btn.dataset.status;
      document.querySelectorAll('.status-filter').forEach(b => b.classList.toggle('primary', b === btn));
      document.querySelectorAll('details.exp').forEach(el => {
        const s = el.dataset.status;
        if (want === 'all' || !s) { el.style.display = ''; return; }
        el.style.display = (s === want) ? '' : 'none';
      });
    });
  });

  // Expand / collapse all
  document.getElementById('expand-all').addEventListener('click', () => {
    document.querySelectorAll('details.exp').forEach(el => { if (el.style.display !== 'none') el.open = true; });
  });
  document.getElementById('collapse-all').addEventListener('click', () => {
    document.querySelectorAll('details.exp').forEach(el => { el.open = false; });
  });

  // Copy-to-clipboard
  document.addEventListener('click', (ev) => {
    if (!ev.target.classList.contains('copy-btn')) return;
    const text = ev.target.dataset.text;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      ev.target.textContent = '✓';
      setTimeout(() => ev.target.textContent = '📋', 800);
    });
  });

  // Compare selected
  document.getElementById('compare-btn').addEventListener('click', () => {
    const selected = Array.from(document.querySelectorAll('.compare-cb:checked')).map(cb => cb.dataset.id);
    if (selected.length < 2) { alert('Pick at least 2 experiments to compare.'); return; }
    const rows = [];
    selected.forEach(id => {
      const exp = document.getElementById('exp-' + id);
      if (!exp) return;
      const name = exp.querySelector('.exp-name').textContent;
      const tables = exp.querySelectorAll('table.results');
      tables.forEach(t => {
        const caption = t.closest('.section').querySelector('h4').textContent;
        const tableHTML = t.outerHTML;
        rows.push('<h3 style="margin-top:16px;">' + name + ' — ' + caption + '</h3>' + tableHTML);
      });
    });
    document.getElementById('compare-content').innerHTML = rows.join('');
    document.getElementById('compare-modal').classList.add('open');
  });
  document.querySelector('#compare-modal .close').addEventListener('click', () => {
    document.getElementById('compare-modal').classList.remove('open');
  });
  document.getElementById('compare-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'compare-modal') document.getElementById('compare-modal').classList.remove('open');
  });

  // Sortable consolidated table
  function parsePct(s) {
    if (!s || s === '—') return -Infinity;
    s = s.trim().replace(/%$/, '').replace(/✓/, '1');
    const n = parseFloat(s);
    return isNaN(n) ? s.toLowerCase() : n;
  }
  document.querySelectorAll('table.sortable').forEach(table => {
    const headers = table.querySelectorAll('th.sort-col');
    let currentSort = { col: -1, asc: false };
    headers.forEach((h, idx) => {
      h.addEventListener('click', () => {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const asc = (currentSort.col === idx) ? !currentSort.asc : false;
        currentSort = { col: idx, asc };
        headers.forEach(x => x.classList.remove('sort-asc', 'sort-desc'));
        h.classList.add(asc ? 'sort-asc' : 'sort-desc');
        rows.sort((a, b) => {
          const av = parsePct(a.cells[idx].textContent);
          const bv = parsePct(b.cells[idx].textContent);
          if (typeof av === 'number' && typeof bv === 'number') return asc ? av - bv : bv - av;
          return asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
        });
        rows.forEach(r => tbody.appendChild(r));
      });
    });
  });
})();
"""


def render_bullseye(leaderboard, target_R=85.0, target_P=50.0):
    """Fancy R/P scatter with concentric rings around the customer target (85R, 50P).
    Each leaderboard config is plotted as a dot in its experiment tint."""
    W, H = 720, 360
    pad_l, pad_r, pad_t, pad_b = 50, 20, 16, 38
    x_min, x_max = 50.0, 100.0  # R axis
    y_min, y_max = 0.0, 80.0    # P axis
    def sx(r): return pad_l + (r - x_min) / (x_max - x_min) * (W - pad_l - pad_r)
    def sy(p): return H - pad_b - (p - y_min) / (y_max - y_min) * (H - pad_t - pad_b)
    cx, cy = sx(target_R), sy(target_P)
    out = [f'<svg class="bullseye-svg" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    # Background
    out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fafbfc"/>')
    # Grid lines (R: every 10; P: every 10)
    for r in range(50, 101, 10):
        out.append(f'<line x1="{sx(r):.1f}" y1="{pad_t}" x2="{sx(r):.1f}" y2="{H-pad_b}" stroke="#e8eaf0" stroke-width="1"/>')
        out.append(f'<text x="{sx(r):.1f}" y="{H-pad_b+14}" text-anchor="middle" font-size="10" fill="#57606a">{r}</text>')
    for p in range(0, 81, 10):
        out.append(f'<line x1="{pad_l}" y1="{sy(p):.1f}" x2="{W-pad_r}" y2="{sy(p):.1f}" stroke="#e8eaf0" stroke-width="1"/>')
        out.append(f'<text x="{pad_l-6}" y="{sy(p)+3:.1f}" text-anchor="end" font-size="10" fill="#57606a">{p}</text>')
    # Target zone: top-right of the customer target (R≥85 AND P≥50)
    out.append(f'<rect x="{cx:.1f}" y="{pad_t}" width="{W-pad_r-cx:.1f}" height="{cy-pad_t:.1f}" '
               f'fill="#86efac" fill-opacity="0.22" stroke="none"/>')
    # Concentric rings centered on the target (visual "fancy dartboard")
    for ring_r, color, opacity in [(120, "#86efac", 0.18), (90, "#bbf7d0", 0.28),
                                    (60, "#86efac", 0.34), (32, "#22c55e", 0.45)]:
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{ring_r}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>')
    # Target crosshair
    out.append(f'<line x1="{cx:.1f}" y1="{pad_t}" x2="{cx:.1f}" y2="{H-pad_b}" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<line x1="{pad_l}" y1="{cy:.1f}" x2="{W-pad_r}" y2="{cy:.1f}" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#dc2626" stroke="white" stroke-width="2"/>')
    out.append(f'<text x="{cx+10:.1f}" y="{cy-10:.1f}" font-size="11" font-weight="700" fill="#991b1b">target {target_R:.0f}R / {target_P:.0f}P</text>')
    # Axis labels
    out.append(f'<text x="{(pad_l+W-pad_r)/2:.1f}" y="{H-6}" text-anchor="middle" font-size="11" fill="#24292f" font-weight="600">micro Recall (%)</text>')
    out.append(f'<text x="14" y="{(pad_t+H-pad_b)/2:.1f}" text-anchor="middle" font-size="11" fill="#24292f" font-weight="600" transform="rotate(-90 14 {(pad_t+H-pad_b)/2:.1f})">micro Precision (%)</text>')
    # Plot leaderboard entries (deduplicate near-identical points)
    placed = []
    for lb in leaderboard:
        if lb.get('R') is None or lb.get('P') is None: continue
        r = lb['R']; p = lb['P']; mid = lb.get('model_id','')
        # Clamp to view box
        if r < x_min or r > x_max or p < y_min or p > y_max: continue
        x = sx(r); y = sy(p)
        # Tint
        tint = EXPERIMENT_TINTS.get(mid, "#9ca3af")
        # Compute label nudge to avoid overlap
        nudge = 0
        for (px, py) in placed:
            if abs(px-x) < 22 and abs(py-y) < 16: nudge += 12
        placed.append((x, y))
        out.append(f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{tint}" stroke="#24292f" stroke-width="1.2"/>'
                   f'<title>{esc(mid)} — {esc(lb["config"])} · R={r:.2f}% P={p:.2f}%</title></g>')
        out.append(f'<text x="{x+8:.1f}" y="{y-7+nudge:.1f}" font-size="10.5" fill="#24292f" font-weight="600">{esc(mid)}</text>')
    out.append('</svg>')
    return "\n".join(out)


def render_bar_chart(rows, domain_label, max_models=10):
    """Grouped R+P bar chart for a domain. Picks best variant per model_id within the rows."""
    # Pick one (best avg_RP) row per model_id
    by_model = {}
    for r in rows:
        mid = r["model_id"]
        if mid not in by_model or r["avg_RP"] > by_model[mid]["avg_RP"]:
            by_model[mid] = r
    items = list(by_model.values())
    items.sort(key=lambda x: latest_first_key(x["model_id"]))
    items = items[:max_models]
    if not items:
        return f'<p><em>(no data for {esc(domain_label)})</em></p>'

    n = len(items)
    bar_w = 14
    gap = 4
    group_w = 2 * bar_w + gap + 18   # two bars + label spacing
    pad_l, pad_r, pad_t, pad_b = 38, 14, 26, 42
    H = 220
    W = max(420, pad_l + pad_r + n * group_w)
    plot_h = H - pad_t - pad_b
    def sy(v): return pad_t + (1 - v/100.0) * plot_h
    out = [f'<svg class="bar-chart-svg" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff"/>')
    # Y axis ticks
    for v in (0, 25, 50, 75, 100):
        y = sy(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#eef0f4" stroke-width="1"/>')
        out.append(f'<text x="{pad_l-5}" y="{y+3:.1f}" text-anchor="end" font-size="9.5" fill="#57606a">{v}</text>')
    # Target lines: R=85 (recall), P=50 (precision)
    out.append(f'<line x1="{pad_l}" y1="{sy(85):.1f}" x2="{W-pad_r}" y2="{sy(85):.1f}" stroke="#0969da" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>')
    out.append(f'<text x="{W-pad_r-2}" y="{sy(85)-2:.1f}" text-anchor="end" font-size="9" fill="#0969da">target R=85</text>')
    out.append(f'<line x1="{pad_l}" y1="{sy(50):.1f}" x2="{W-pad_r}" y2="{sy(50):.1f}" stroke="#dc2626" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>')
    out.append(f'<text x="{W-pad_r-2}" y="{sy(50)-2:.1f}" text-anchor="end" font-size="9" fill="#dc2626">target P=50</text>')
    # Bars
    for i, r in enumerate(items):
        gx = pad_l + i * group_w + 8
        tint = EXPERIMENT_TINTS.get(r["model_id"], "#9ca3af")
        # R bar (solid)
        Rv = r["R"] if r["R"] is not None else 0
        Pv = r["P"] if r["P"] is not None else 0
        yR = sy(Rv); hR = (H-pad_b) - yR
        yP = sy(Pv); hP = (H-pad_b) - yP
        out.append(f'<rect x="{gx:.1f}" y="{yR:.1f}" width="{bar_w}" height="{hR:.1f}" fill="{tint}" stroke="#24292f" stroke-width="0.7">'
                   f'<title>{esc(r["model_id"])} micro R = {Rv:.2f}% · {esc(r["variant"])}</title></rect>')
        out.append(f'<text x="{gx+bar_w/2:.1f}" y="{yR-2:.1f}" text-anchor="middle" font-size="9" fill="#24292f">{Rv:.1f}</text>')
        out.append(f'<rect x="{gx+bar_w+gap:.1f}" y="{yP:.1f}" width="{bar_w}" height="{hP:.1f}" fill="{tint}" fill-opacity="0.45" stroke="#24292f" stroke-width="0.7" stroke-dasharray="2 2">'
                   f'<title>{esc(r["model_id"])} micro P = {Pv:.2f}% · {esc(r["variant"])}</title></rect>')
        out.append(f'<text x="{gx+bar_w+gap+bar_w/2:.1f}" y="{yP-2:.1f}" text-anchor="middle" font-size="9" fill="#24292f">{Pv:.1f}</text>')
        # x-axis label = model id (link)
        out.append(f'<a href="#exp-{esc(r["model_id"])}"><text x="{gx+bar_w+gap/2:.1f}" y="{H-pad_b+14:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" fill="#0969da">{esc(r["model_id"])}</text></a>')
    # Legend
    lx = pad_l; ly = H - 8
    out.append(f'<rect x="{lx}" y="{ly-9}" width="10" height="10" fill="#888"/>')
    out.append(f'<text x="{lx+14}" y="{ly}" font-size="10.5" fill="#24292f">Recall</text>')
    out.append(f'<rect x="{lx+70}" y="{ly-9}" width="10" height="10" fill="#888" fill-opacity="0.45" stroke="#24292f" stroke-width="0.7" stroke-dasharray="2 2"/>')
    out.append(f'<text x="{lx+84}" y="{ly}" font-size="10.5" fill="#24292f">Precision</text>')
    # Axis title
    out.append(f'<text x="{W/2:.1f}" y="14" text-anchor="middle" font-size="11" font-weight="600" fill="#24292f">{esc(domain_label)} — best variant per model</text>')
    out.append('</svg>')
    return "\n".join(out)


def render_pareto(rows_team, rows_overseas, rows_combined):
    """Single scatter: all (R,P) points across domains, Pareto frontier on team test highlighted."""
    W, H = 720, 320
    pad_l, pad_r, pad_t, pad_b = 46, 18, 24, 40
    x_min, x_max = 40.0, 100.0
    y_min, y_max = 0.0, 80.0
    def sx(r): return pad_l + (r - x_min) / (x_max - x_min) * (W - pad_l - pad_r)
    def sy(p): return H - pad_b - (p - y_min) / (y_max - y_min) * (H - pad_t - pad_b)
    out = [f'<svg class="bar-chart-svg" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff"/>')
    # Target zone
    out.append(f'<rect x="{sx(85):.1f}" y="{pad_t}" width="{W-pad_r-sx(85):.1f}" height="{sy(50)-pad_t:.1f}" fill="#86efac" fill-opacity="0.18"/>')
    out.append(f'<line x1="{sx(85):.1f}" y1="{pad_t}" x2="{sx(85):.1f}" y2="{H-pad_b}" stroke="#0969da" stroke-width="1" stroke-dasharray="3 3"/>')
    out.append(f'<line x1="{pad_l}" y1="{sy(50):.1f}" x2="{W-pad_r}" y2="{sy(50):.1f}" stroke="#dc2626" stroke-width="1" stroke-dasharray="3 3"/>')
    # Grid ticks
    for r in range(40, 101, 10):
        out.append(f'<text x="{sx(r):.1f}" y="{H-pad_b+14}" text-anchor="middle" font-size="9.5" fill="#57606a">{r}</text>')
    for p in range(0, 81, 20):
        out.append(f'<text x="{pad_l-5}" y="{sy(p)+3:.1f}" text-anchor="end" font-size="9.5" fill="#57606a">{p}</text>')
    # Plot rows: team triangle, overseas square, combined circle
    def plot(rows, shape):
        for r in rows:
            if r["R"] is None or r["P"] is None: continue
            x = sx(r["R"]); y = sy(r["P"])
            tint = EXPERIMENT_TINTS.get(r["model_id"], "#9ca3af")
            title = f'{r["model_id"]} {r["variant"]} — R={r["R"]:.2f}% P={r["P"]:.2f}%'
            if shape == 'circle':
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{tint}" stroke="#24292f" stroke-width="0.8"><title>{esc(title)}</title></circle>')
            elif shape == 'triangle':
                pts = f"{x:.1f},{y-5:.1f} {x-5:.1f},{y+4:.1f} {x+5:.1f},{y+4:.1f}"
                out.append(f'<polygon points="{pts}" fill="{tint}" stroke="#24292f" stroke-width="0.8"><title>{esc(title)}</title></polygon>')
            else:  # square
                out.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="{tint}" stroke="#24292f" stroke-width="0.8"><title>{esc(title)}</title></rect>')
    plot(rows_team, 'triangle')
    plot(rows_overseas, 'square')
    plot(rows_combined, 'circle')
    # Axis titles
    out.append(f'<text x="{(pad_l+W-pad_r)/2:.1f}" y="{H-6}" text-anchor="middle" font-size="11" font-weight="600" fill="#24292f">micro Recall (%)</text>')
    out.append(f'<text x="14" y="{(pad_t+H-pad_b)/2:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#24292f" transform="rotate(-90 14 {(pad_t+H-pad_b)/2:.1f})">micro Precision (%)</text>')
    # Legend (shapes)
    lx = pad_l + 8; ly = pad_t + 10
    out.append(f'<polygon points="{lx},{ly-4} {lx-4},{ly+3} {lx+4},{ly+3}" fill="#888"/><text x="{lx+8}" y="{ly+3}" font-size="10" fill="#24292f">team</text>')
    out.append(f'<rect x="{lx+50}" y="{ly-4}" width="8" height="8" fill="#888"/><text x="{lx+62}" y="{ly+3}" font-size="10" fill="#24292f">overseas</text>')
    out.append(f'<circle cx="{lx+114:.1f}" cy="{ly:.1f}" r="4" fill="#888"/><text x="{lx+122}" y="{ly+3}" font-size="10" fill="#24292f">combined</text>')
    out.append(f'<text x="{sx(85)+4}" y="{pad_t+10}" font-size="10" font-weight="600" fill="#0a6e2e">target zone (R≥85 AND P≥50)</text>')
    out.append('</svg>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dmdd_data.json")
    ap.add_argument("--out", default="dmdd_dashboard.html")
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    last_updated = data.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M %Z"))
    target = data.get("customer_target", "")
    class_mapping = data.get("class_mapping", {})

    experiments = data.get("experiments", [])
    comparisons = data.get("comparisons", [])
    analyses = data.get("analyses", [])
    leaderboard = data.get("leaderboard", [])

    by_cat = {"model": [], "other": []}
    for e in experiments:
        c = e.get("category", "model")
        by_cat.setdefault(c, []).append(e)

    # Status counts
    status_counts = {}
    for e in experiments:
        s = e.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    # Sort experiments latest-first up front so the sidebar nav can list them in order
    experiments_sorted = sorted(experiments, key=lambda e: latest_first_key(e["id"]))

    # Build sidebar nav HTML
    sidebar_parts = ['<aside class="sidebar"><div class="nav-section"><h3>Jump to</h3><ul>',
                     '<li><a href="#top">⌂ Top</a></li>',
                     '<li><a href="#target-view">🎯 Target view</a></li>',
                     '<li><a href="#leaderboard">🏆 Leaderboard</a></li>',
                     '<li><a href="#chart-gallery">📊 Charts</a></li>',
                     '<li><a href="#consolidated">📋 Consolidated</a></li>',
                     '<li class="indent"><a href="#consolidated-team-card">🟦 Team</a></li>',
                     '<li class="indent"><a href="#consolidated-overseas-card">🟧 Overseas</a></li>',
                     '<li class="indent"><a href="#consolidated-combined-card">🟩 Combined</a></li>',
                     '<li><a href="#experiments">🧪 Experiments</a></li>']
    for exp in experiments_sorted:
        eid = exp["id"]; ename = exp.get("name", eid)
        tint = EXPERIMENT_TINTS.get(eid, "#e5e7eb")
        # Trim long names for the sidebar
        short = ename if len(ename) <= 30 else ename[:28] + '…'
        sidebar_parts.append(f'<li class="indent"><a href="#exp-{esc(eid)}">'
                             f'<span class="exp-dot" style="background:{tint}"></span>'
                             f'<strong>{esc(eid)}</strong> {esc(short)}</a></li>')
    if comparisons:
        sidebar_parts.append('<li><a href="#comparisons">🔁 Comparisons</a></li>')
    if analyses:
        sidebar_parts.append('<li><a href="#analyses">📈 Analyses</a></li>')
    sidebar_parts.append('</ul></div></aside>')
    sidebar_html = "".join(sidebar_parts)

    out_html = [f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DMDD experiments dashboard</title>
<style>{CSS}</style>
</head><body id="top">

{sidebar_html}

<main class="main-content">

<h1>DMDD experiments dashboard</h1>
<p class="meta">Last updated: <strong>{esc(last_updated)}</strong>
&nbsp;·&nbsp; Customer target: <strong>{esc(target)}</strong>
&nbsp;·&nbsp; Total experiments: {len(experiments)}
&nbsp;·&nbsp; {" · ".join(f"{k} {v}" for k, v in status_counts.items())}</p>

<div id="target-view" class="bullseye-wrap section-anchor">
  <div class="header">
    <h3>🎯 Target view — how close are we to {esc(target)}?</h3>
    <span class="target-hint">Green bullseye = customer target. Dots closer to the center are better. Hover for details.</span>
  </div>
  {render_bullseye(leaderboard)}
</div>

<h2 id="leaderboard" class="section-anchor">🏆 Leaderboard</h2>
<p>Best operating point per objective. Color-coded by experiment to match the consolidated tables.
🟦 team / 🟧 overseas / 🟩 combined / ✨ specialty / ⚠ negative result.</p>
<table class="results leaderboard" style="width:100%;">
<thead><tr><th>Objective</th><th>Config</th><th>micro R</th><th>micro P</th></tr></thead>
<tbody>
"""]
    for lb in leaderboard:
        model_id = lb.get('model_id', '')
        tint = exp_tint_style(model_id) if model_id else ''
        row_class = f' class="{exp_row_class(model_id)}"' if model_id else ''
        config_text = esc(lb['config'])
        if model_id:
            config_text = f"<a href='#exp-{esc(model_id)}'>{config_text}</a>"
        out_html.append(f"<tr{row_class}{tint}><td>{esc(lb['objective'])}</td><td>{config_text}</td>"
                        f"<td>{lb['R']:.2f}%</td><td>{lb['P']:.2f}%</td></tr>")
    out_html.append("</tbody></table>")

    out_html.append("<h2>Class ID mapping</h2>")
    out_html.append('<table class="kv" style="max-width:400px;">')
    for cid, cname in class_mapping.items():
        out_html.append(f"<tr><th>{esc(cid)}</th><td>{esc(cname)}</td></tr>")
    out_html.append("</table>")

    # ===== Consolidated tables split by test set =====
    COMBINED_MODELS = {"v15", "v17", "v20"}
    DEPRIORITIZED = {"v12"}

    def classify_variant(variant_str):
        """Return one of {'combined','team','overseas','other'} based on variant string."""
        v = variant_str.lower()
        if "combined" in v: return "combined"
        if "team" in v: return "team"
        if "overseas" in v: return "overseas"
        # default — for models with only one domain (v16 lab, v18 own test, v19, v23, v24),
        # classify by the experiment's main domain
        return "other"

    def domain_from_exp(eid):
        """Map model id to its default domain when variants don't say."""
        if eid in {"v12", "v18", "v19", "v24"}: return "team"
        if eid in {"v13", "v16", "v23"}: return "overseas"
        return "other"

    rows_combined = []; rows_team = []; rows_overseas = []
    for exp in experiments:
        eid = exp["id"]; ename = exp["name"]
        for res in exp.get("results", []):
            variant = res.get("variant", "")
            # Drop partial / incomplete rows entirely
            if "partial" in variant.lower(): continue
            r_val = res.get("R"); p_val = res.get("P"); mIoU = res.get("mIoU")
            if r_val is None and p_val is None: continue  # skip rows without numbers
            hits = res.get("hits_target", False)
            cls = classify_variant(variant)
            if cls == "other": cls = domain_from_exp(eid)
            if eid in COMBINED_MODELS:    group = 0; group_name = "★ combined"
            elif eid in DEPRIORITIZED:     group = 2; group_name = "baseline"
            else:                          group = 1; group_name = ""
            avg_rp = ((r_val if r_val is not None else 0) + (p_val if p_val is not None else 0)) / 2
            row = {"model_id": eid, "model": ename, "variant": variant,
                   "R": r_val, "P": p_val, "avg_RP": avg_rp, "mIoU": mIoU,
                   "hits_target": hits, "group": group, "group_name": group_name}
            if cls == "combined":   rows_combined.append(row)
            elif cls == "team":     rows_team.append(row)
            elif cls == "overseas": rows_overseas.append(row)

    # Latest experiment first within each section (v24 → v12), with per-experiment tint
    for arr in (rows_combined, rows_team, rows_overseas):
        arr.sort(key=lambda r: (latest_first_key(r["model_id"]), -r["avg_RP"]))

    def render_consolidated_table(rows, table_id):
        if not rows:
            return "<p><em>(no results in this category)</em></p>"
        h = [f'<table class="results" id="{table_id}" style="width:100%;">']
        h.append("<thead><tr>")
        cols = [("group_name","Tier"),("model_id","Model"),("variant","Variant"),
                ("R","micro R"),("P","micro P"),("avg_RP","avg(R,P)"),
                ("mIoU","mIoU"),("hits","85R/50P?")]
        for key, label in cols:
            h.append(f'<th data-key="{esc(key)}">{esc(label)}</th>')
        h.append("</tr></thead><tbody>")
        for r in rows:
            classes = [exp_row_class(r["model_id"])]
            if r["hits_target"]: classes.append("hit")
            if r["model_id"] in COMBINED_MODELS: classes.append("pinned")
            elif r["model_id"] in DEPRIORITIZED: classes.append("deprio")
            row_class = f' class="{ " ".join(classes) }"'
            tint = exp_tint_style(r["model_id"])
            h.append(f"<tr{row_class}{tint}>")
            h.append(f'<td>{esc(r["group_name"])}</td>')
            h.append(f'<td><a href="#exp-{esc(r["model_id"])}">{esc(r["model_id"])}</a></td>')
            h.append(f'<td>{esc(r["variant"])}</td>')
            h.append(f'<td>{r["R"]:.2f}%</td>' if r["R"] is not None else "<td>—</td>")
            h.append(f'<td>{r["P"]:.2f}%</td>' if r["P"] is not None else "<td>—</td>")
            h.append(f'<td>{r["avg_RP"]:.2f}%</td>')
            h.append(f'<td>{r["mIoU"]:.3f}</td>' if r["mIoU"] is not None else "<td>—</td>")
            h.append(f'<td>{"✓" if r["hits_target"] else ""}</td>')
            h.append("</tr>")
        h.append("</tbody></table>")
        return "".join(h)

    # ===== Chart gallery (SVG — no JS required) =====
    out_html.append('<h2 id="chart-gallery" class="section-anchor">📊 Charts</h2>')
    out_html.append('<p>All charts are static SVG so they render in JupyterLab without JavaScript. '
                    'Hover any bar / dot for the exact numbers and the variant it represents.</p>')

    out_html.append('<div class="bar-charts-wrap">')
    out_html.append('<div class="bar-chart-card"><h4>🟦 Team test — micro Recall vs Precision</h4>'
                    '<p class="bc-hint">One bar pair per model (best variant). Dashed lines = customer target (R=85, P=50).</p>'
                    + render_bar_chart(rows_team, "Team test") + '</div>')
    out_html.append('<div class="bar-chart-card"><h4>🟧 Overseas test</h4>'
                    '<p class="bc-hint">Best variant per model on the overseas test set.</p>'
                    + render_bar_chart(rows_overseas, "Overseas test") + '</div>')
    out_html.append('<div class="bar-chart-card"><h4>🟩 Combined (team ∪ overseas)</h4>'
                    '<p class="bc-hint">Best variant per model when both test sets are pooled.</p>'
                    + render_bar_chart(rows_combined, "Combined") + '</div>')
    out_html.append('<div class="bar-chart-card"><h4>🌐 Pareto view — every (R, P) point across all domains</h4>'
                    '<p class="bc-hint">▲ team · ■ overseas · ● combined. The green box is the customer target zone (R≥85 AND P≥50). '
                    'Points in or near that zone are the candidates worth shipping.</p>'
                    + render_pareto(rows_team, rows_overseas, rows_combined) + '</div>')
    out_html.append('</div>')

    # ===== Three consolidated tables, team first =====
    out_html.append('<h2 id="consolidated" class="section-anchor">📋 Consolidated results</h2>')
    out_html.append('<p>Sorted into three sections. <strong>Team test first</strong> (primary deployment target). '
                    '★ = combined-data models (v15/v17/v20) pinned at the top of each section, then by avg(R,P) descending. '
                    'Partial / incomplete runs are filtered out.</p>')

    out_html.append('<div class="domain-card team-card" id="consolidated-team-card">')
    out_html.append('<h3 class="domain-header">🟦 Team test (primary)</h3>')
    out_html.append(render_consolidated_table(rows_team, "consolidated-team"))
    out_html.append("</div>")

    out_html.append('<div class="domain-card overseas-card" id="consolidated-overseas-card">')
    out_html.append('<h3 class="domain-header">🟧 Overseas test</h3>')
    out_html.append(render_consolidated_table(rows_overseas, "consolidated-overseas"))
    out_html.append("</div>")

    out_html.append('<div class="domain-card combined-card" id="consolidated-combined-card">')
    out_html.append('<h3 class="domain-header">🟩 Combined (team ∪ overseas)</h3>')
    out_html.append(render_consolidated_table(rows_combined, "consolidated-combined"))
    out_html.append("</div>")

    out_html.append('<h2 id="experiments" class="section-anchor">🧪 Experiments</h2>')
    out_html.append("<p>Latest experiments first. Each version has its own background tint so it's easy to track its rows across all tables.</p>")
    for exp in experiments_sorted:
        out_html.append(render_experiment(exp))

    if comparisons:
        out_html.append('<h2 id="comparisons" class="section-anchor">🔁 Cross-comparisons</h2>')
        for cmp in comparisons:
            out_html.append(render_comparison(cmp))

    if analyses:
        out_html.append('<h2 id="analyses" class="section-anchor">📈 Data analyses</h2>')
        for an in analyses:
            out_html.append(render_analysis(an))

    out_html.append("""
<footer>Generated by generate_dashboard.py · SVG charts, no JavaScript required.</footer>
</main>
</body></html>""")

    Path(args.out).write_text("\n".join(out_html), encoding="utf-8")
    print(f"wrote -> {args.out}  ({len(experiments)} experiments, {len(comparisons)} comparisons, {len(analyses)} analyses)")


if __name__ == "__main__":
    main()
