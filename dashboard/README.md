# DMDD experiments dashboard

A self-contained HTML dashboard summarising every SAM 3.1 LoRA wind-blade
defect experiment to date — single-file, no JavaScript required, opens in any
browser or inside JupyterLab via `IFrame` / `display(HTML(...))`.

## Files

| file | role |
|---|---|
| `dmdd_data.json` | source of truth — every experiment / leaderboard / consolidated row lives here |
| `generate_dashboard.py` | renders the JSON into a single self-contained `dmdd_dashboard.html` |
| `dmdd_dashboard.html` | the generated page (commit so it's viewable directly on GitHub raw / Pages) |

## What's in it

- **Sticky left sidebar** — jump to any section (leaderboard, charts, each version's card, comparisons, analyses).
- **🎯 Target view** — bullseye chart with concentric rings around the customer target (R=85% / P=50%); each model is plotted as a dot in its own color. Closer to center = better.
- **🏆 Leaderboard** — best operating point per objective (team / overseas / combined / specialty), each row tinted by experiment and linked to the model's card.
- **📊 Chart gallery** — per-domain bar charts (Recall vs Precision with target dashed lines) plus a Pareto scatter that overlays every (R, P) point across all domains. Shapes encode domain: ▲ team · ■ overseas · ● combined.
- **📋 Consolidated tables** — three sections (team / overseas / combined), latest experiment first, each version with its own background tint so rows are easy to track across tables.
- **🧪 Per-experiment cards** — recipe, training data, results, cascade results, per-class numbers, caveats, take-away.

Every interactive piece is anchor links or hover-tooltips (SVG `<title>`); no
`<script>` blocks, so JupyterLab's HTML sandbox does not strip anything.

## Regenerating

```bash
cd dashboard
python generate_dashboard.py        # reads dmdd_data.json → writes dmdd_dashboard.html
```

To update results, edit `dmdd_data.json` and rerun. Schema mirrors what each
helper in `generate_dashboard.py` reads — the simplest reference is the existing
entries.

## Viewing in JupyterLab

```python
from IPython.display import IFrame
IFrame('dmdd_dashboard.html', width='100%', height=900)
```

`display(HTML(open('dmdd_dashboard.html').read()))` also works — the dashboard
is built so nothing depends on JavaScript.
