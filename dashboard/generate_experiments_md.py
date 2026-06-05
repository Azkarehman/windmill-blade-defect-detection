#!/usr/bin/env python3
"""Render dmdd_data.json into a comprehensive EXPERIMENTS.md.

Same source of truth as the HTML dashboard, just markdown-formatted so it
renders well on GitHub. Run from the dashboard/ directory:

    python generate_experiments_md.py --out ../EXPERIMENTS.md
"""
import argparse, json, re
from pathlib import Path


def latest_first_key(eid):
    m = re.match(r"v(\d+)", eid or "")
    return -int(m.group(1)) if m else 0


def fmt_pct(v):
    if v is None: return "—"
    return f"{v:.2f}%" if isinstance(v, (int, float)) else str(v)


def fmt_iou(v):
    if v is None: return "—"
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def md_kv(d, keep_keys=None):
    out = []
    for k, v in d.items():
        if keep_keys and k not in keep_keys: continue
        if v is None or v == "": continue
        out.append(f"- **{k}**: {v}")
    return "\n".join(out)


def md_results_table(rows, cols=None):
    """Render a list of result dicts as a pipe-delimited markdown table."""
    if not rows: return ""
    if cols is None:
        seen = []
        for r in rows:
            for k in r.keys():
                if k not in seen: seen.append(k)
        cols = seen
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None or v == "":
                cells.append("—")
            elif isinstance(v, float):
                if c == "mIoU": cells.append(f"{v:.3f}")
                elif "%" in c.lower() or c in ("R", "P"): cells.append(f"{v:.2f}%")
                else: cells.append(f"{v:.2f}")
            elif isinstance(v, bool):
                cells.append("✓" if v else "")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_experiment(e):
    parts = [f"### {e['name']}"]
    if e.get("status"):
        parts.append(f"_Status: **{e['status']}**_ · tags: {', '.join(e.get('tags', [])) or '—'}")
    if e.get("purpose"):
        parts.append(f"\n**Purpose.** {e['purpose']}")
    if e.get("training_data"):
        parts.append("\n**Training data**\n\n" + md_kv(e["training_data"]))
    if e.get("recipe"):
        parts.append("\n**Recipe**\n\n" + md_kv(e["recipe"]))
    if e.get("results"):
        parts.append("\n**Results**\n\n" + md_results_table(
            e["results"],
            cols=[c for c in ("variant", "mIoU", "R", "P", "hits_target", "note") if any(c in r for r in e["results"])]
        ))
    if e.get("stage1_standalone"):
        parts.append("\n**Stage-1 standalone (tile classifier)**\n\n" + md_kv(e["stage1_standalone"]))
    if e.get("cascade_results"):
        parts.append("\n**Cascade results**\n\n" + md_results_table(
            e["cascade_results"],
            cols=[c for c in ("pair", "records", "baseline_R", "baseline_P", "best_mode", "best_t_gate", "best_R", "best_P", "note") if any(c in r for r in e["cascade_results"])]
        ))
    for pc_key, pc_label in (("per_class_full_tta_T", "Per-class (full-TTA + T*)"),
                             ("per_class_no_tta_T", "Per-class (no-TTA + T*)")):
        if e.get(pc_key):
            parts.append(f"\n**{pc_label}**\n\n" + md_results_table(
                e[pc_key],
                cols=["class", "TP", "FP", "FN", "R", "P"]
            ))
    if e.get("caveats"):
        c = e["caveats"]
        text = c if isinstance(c, str) else "\n".join(f"- {x}" for x in c)
        parts.append(f"\n**Caveats.** {text}")
    if e.get("pending"):
        parts.append("\n**Pending**\n\n" + "\n".join(f"- {p}" for p in e["pending"]))
    if e.get("takeaway"):
        parts.append(f"\n**Take-away.** {e['takeaway']}")
    return "\n".join(parts)


def render_leaderboard(lb):
    rows = []
    for e in lb:
        rows.append({"Objective": e.get("objective", ""),
                     "Config":   e.get("config", ""),
                     "R":        e.get("R"),
                     "P":        e.get("P")})
    return md_results_table(rows, cols=["Objective", "Config", "R", "P"])


def render_comparison(c):
    parts = [f"### {c['title']}"]
    if c.get("purpose"): parts.append(f"\n**Purpose.** {c['purpose']}")
    if c.get("table"):
        parts.append("\n" + md_results_table(c["table"]))
    if c.get("takeaway"): parts.append(f"\n**Take-away.** {c['takeaway']}")
    return "\n".join(parts)


def render_analysis(a):
    parts = [f"### {a['title']}"]
    if a.get("summary"): parts.append(f"\n{a['summary']}")
    if a.get("table"):
        parts.append("\n" + md_results_table(a["table"]))
    if a.get("takeaway"): parts.append(f"\n**Take-away.** {a['takeaway']}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dmdd_data.json")
    ap.add_argument("--out",  default="../EXPERIMENTS.md")
    args = ap.parse_args()

    data = json.load(open(args.data))
    last_updated = data.get("last_updated", "")
    target = data.get("customer_target", "")
    class_map = data.get("class_mapping", {})

    md = []
    md.append("# DMDD wind-blade defect experiments")
    md.append("")
    md.append(f"_Auto-generated from `dashboard/dmdd_data.json` · last updated **{last_updated}**._")
    md.append("")
    md.append("> Full interactive view (bullseye target chart, per-domain bar charts, Pareto scatter, per-experiment cards): see [`dashboard/dmdd_dashboard.html`](dashboard/dmdd_dashboard.html).")
    md.append("")
    md.append(f"**Customer target.** {target}")
    md.append("")
    md.append("**Class IDs.** ")
    md.append("")
    md.append("| id | class |")
    md.append("|---|---|")
    for cid, cname in class_map.items():
        md.append(f"| {cid} | {cname} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🏆 Leaderboard — best operating point per objective")
    md.append("")
    md.append(render_leaderboard(data.get("leaderboard", [])))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🧪 Experiments (latest first)")
    md.append("")
    experiments = sorted(data.get("experiments", []), key=lambda e: latest_first_key(e["id"]))
    for e in experiments:
        md.append(render_experiment(e))
        md.append("")
        md.append("---")
        md.append("")
    if data.get("comparisons"):
        md.append("## 🔁 Cross-comparisons")
        md.append("")
        for c in data["comparisons"]:
            md.append(render_comparison(c))
            md.append("")
            md.append("---")
            md.append("")
    if data.get("analyses"):
        md.append("## 📈 Data analyses")
        md.append("")
        for a in data["analyses"]:
            md.append(render_analysis(a))
            md.append("")
            md.append("---")
            md.append("")
    if data.get("notes"):
        md.append("## Notes")
        md.append("")
        notes = data["notes"]
        if isinstance(notes, list):
            for n in notes: md.append(f"- {n}")
        else:
            md.append(str(notes))

    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"wrote -> {args.out}  ({len(experiments)} experiments)")


if __name__ == "__main__":
    main()
