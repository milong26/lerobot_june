#!/usr/bin/env python3
"""Create two PPT-ready figures from current partial quick-eval results."""

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_ROOT = SCRIPT_DIR / "quick_surrogate_eval"


def sf(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def rankdata(values: list[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def corr(rows: list[dict[str, Any]], xk: str, yk: str) -> dict[str, Any]:
    pairs = []
    for r in rows:
        x, y = sf(r.get(xk)), sf(r.get(yk))
        if x is not None and y is not None:
            pairs.append((x, y))
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "pearson": pearson(x, y),
        "spearman": pearson(rankdata(x).tolist(), rankdata(y).tolist()),
    }


def fmt(v: float | None, d: int = 3) -> str:
    return "NA" if v is None else f"{v:.{d}f}"


def camera_group(name: str) -> str:
    if name.startswith("corner3"):
        return "corner3"
    if name.startswith("corner2"):
        return "corner2"
    if name.startswith("corner"):
        return "corner"
    return name or "unknown"


def load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    files = sorted(run_dir.glob("shard_*_of_*/quick_surrogate_results.csv"))
    if not files:
        raise FileNotFoundError(f"No shard results found under {run_dir}")

    by_path: dict[str, dict[str, str]] = {}
    for p in files:
        for r in read_csv(p):
            cp = r.get("checkpoint_path", "")
            if cp:
                by_path[cp] = r

    status: dict[str, int] = {}
    valid: list[dict[str, Any]] = []
    for r in by_path.values():
        s = r.get("eval_status", "")
        status[s] = status.get(s, 0) + 1
        if s != "ok":
            continue
        vals = {}
        ok = True
        for k in ("quick_success", "quick_grasp_success", "full_success", "full_grasp_success"):
            v = sf(r.get(k))
            if v is None:
                ok = False
                break
            vals[k] = v
        if not ok:
            continue
        item: dict[str, Any] = dict(r)
        item.update(vals)
        item["camera_group"] = camera_group(r.get("camera_name", ""))
        valid.append(item)

    valid.sort(key=lambda r: (r.get("model", ""), int(r.get("checkpoint", "0"))))
    return valid, status


def examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    fn = [r for r in rows if r["quick_success"] == 0 and r["full_success"] >= 20]
    ov = [r for r in rows if r["quick_success"] >= 40 and r["full_success"] <= 25]
    return {
        "false_negative": max(fn, key=lambda r: r["full_success"]) if fn else None,
        "overestimate": max(ov, key=lambda r: r["quick_success"] - r["full_success"]) if ov else None,
    }


def save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[SAVE] {png_path}")


def plot_slide1(rows, stats, ex, outdir):
    fig = plt.figure(figsize=(13.333, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0],
                          left=0.07, right=0.96, top=0.82, bottom=0.13, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    tx = fig.add_subplot(gs[0, 1])
    tx.axis("off")

    groups = ["corner", "corner2", "corner3"]
    markers = {"corner": "o", "corner2": "s", "corner3": "^"}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for gi, g in enumerate(groups):
        sub = [r for r in rows if r["camera_group"] == g]
        if not sub:
            continue
        x = np.asarray([r["quick_success"] for r in sub])
        y = np.asarray([r["full_success"] for r in sub])
        jitter = np.asarray([((i % 5) - 2) * 0.75 for i in range(len(sub))])
        ax.scatter(x + jitter, y, s=72, alpha=0.82, marker=markers[g],
                   color=colors[gi], edgecolors="white", linewidths=0.7,
                   label=f"{g} (n={len(sub)})")

    ax.plot([0, 100], [0, 100], "--", color="0.55", linewidth=1)
    ax.set_xlim(-6, 106)
    ymax = max(40, math.ceil(max(r["full_success"] for r in rows) / 10) * 10 + 5)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Quick task success (5 episodes, %)")
    ax.set_ylabel("Historical full-eval task success (200 episodes, %)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")

    s = stats["quick_task_vs_full_task"]
    ax.text(0.98, 0.96,
            f"n = {len(rows)}\\nPearson r = {fmt(s['pearson'])}\\nSpearman rho = {fmt(s['spearman'])}",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8"))

    fig.suptitle("Preliminary Validation: 5-Episode Success Does Not Reliably Rank Checkpoints",
                 y=0.95, fontsize=21, fontweight="bold")

    tx.text(0, 0.96, "What the scatter shows", fontsize=16, fontweight="bold", va="top")
    tx.text(0, 0.86,
            "• Five episodes quantize success into\\n"
            "  20-point steps.\\n\\n"
            "• Current rank agreement with the\\n"
            "  200-episode result is weak.\\n\\n"
            "• The same quick score can map to\\n"
            "  very different full-eval performance.",
            fontsize=13, va="top", linespacing=1.35)

    y = 0.48
    tx.text(0, y, "Concrete failure modes", fontsize=16, fontweight="bold", va="top")
    y -= 0.09
    fn = ex["false_negative"]
    if fn:
        tx.text(0, y, "False negative", fontsize=13, fontweight="bold", va="top")
        tx.text(0, y - 0.05,
                f"Quick {fn['quick_success']:.0f}% vs. full {fn['full_success']:.1f}%\\n"
                f"{fn['model'].split('/')[0]} @ {fn['checkpoint']}",
                fontsize=11.3, va="top")
        y -= 0.16
    ov = ex["overestimate"]
    if ov:
        tx.text(0, y, "Overestimation", fontsize=13, fontweight="bold", va="top")
        tx.text(0, y - 0.05,
                f"Quick {ov['quick_success']:.0f}% vs. full {ov['full_success']:.1f}%\\n"
                f"{ov['model'].split('/')[0]} @ {ov['checkpoint']}",
                fontsize=11.3, va="top")

    tx.text(0, 0.035,
            "Preliminary finding:\\nRaw 5-episode task success is too sparse and state-dependent\\n"
            "to serve as a reliable checkpoint ranking signal.",
            fontsize=12.2, fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="0.96", edgecolor="0.82"))

    save(fig, outdir, "slide1_quick_vs_full")
    plt.close(fig)


def plot_slide2(rows, stats, outdir):
    fig = plt.figure(figsize=(13.333, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.5], width_ratios=[1.35, 1.0],
                          left=0.07, right=0.96, top=0.82, bottom=0.10,
                          hspace=0.34, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")

    fig.suptitle("Preliminary Direction: Intermediate Grasp Signals Are More Informative",
                 y=0.95, fontsize=21, fontweight="bold")

    keys = ["quick_task_vs_full_task", "quick_grasp_vs_full_task", "quick_grasp_vs_full_grasp"]
    labels = ["Quick task\\n→ Full task", "Quick grasp\\n→ Full task", "Quick grasp\\n→ Full grasp"]
    p = [stats[k]["pearson"] for k in keys]
    s = [stats[k]["spearman"] for k in keys]
    x = np.arange(3)
    w = 0.34
    b1 = ax1.bar(x - w/2, p, w, label="Pearson r")
    b2 = ax1.bar(x + w/2, s, w, label="Spearman rho")
    ax1.axhline(0, color="0.35", linewidth=1)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Correlation")
    ax1.set_ylim(-0.45, 0.8)
    ax1.set_title("Which cheap signal tracks full evaluation?")
    ax1.grid(axis="y", alpha=0.18)
    ax1.legend(frameon=False, loc="upper left")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x()+b.get_width()/2, h + (0.025 if h >= 0 else -0.04),
                     f"{h:+.2f}", ha="center",
                     va="bottom" if h >= 0 else "top", fontsize=10.5)

    cstats = stats["camera_quick_task_vs_full_task"]
    cams = ["corner", "corner2", "corner3"]
    vals = [cstats.get(c, {}).get("spearman", np.nan) for c in cams]
    ns = [cstats.get(c, {}).get("n", 0) for c in cams]
    bars = ax2.bar(cams, vals)
    ax2.axhline(0, color="0.35", linewidth=1)
    ax2.set_ylim(-0.45, 0.65)
    ax2.set_ylabel("Spearman rho")
    ax2.set_title("Reliability varies across camera settings")
    ax2.grid(axis="y", alpha=0.18)
    for b, v, n in zip(bars, vals, ns):
        if np.isnan(v):
            continue
        ax2.text(b.get_x()+b.get_width()/2, v + (0.02 if v >= 0 else -0.035),
                 f"{v:+.2f}\\nn={n}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=10.5)

    stages = [
        ("Checkpoint", "candidate"),
        ("Few episodes", "cheap probe"),
        ("Grasp signal", "intermediate competence"),
        ("Action / transport", "quality signal"),
        ("Full eval", "selected only"),
    ]
    xs = np.linspace(0.08, 0.92, len(stages))
    y = 0.48
    bw, bh = 0.15, 0.42
    for i, ((title, sub), xp) in enumerate(zip(stages, xs)):
        box = FancyBboxPatch((xp-bw/2, y-bh/2), bw, bh,
                             boxstyle="round,pad=0.012,rounding_size=0.018",
                             edgecolor="0.45", facecolor="0.97", linewidth=1.2,
                             transform=ax3.transAxes)
        ax3.add_patch(box)
        ax3.text(xp, y+0.055, title, transform=ax3.transAxes,
                 ha="center", va="center", fontsize=11.5, fontweight="bold")
        ax3.text(xp, y-0.085, sub, transform=ax3.transAxes,
                 ha="center", va="center", fontsize=9.2)
        if i < len(stages)-1:
            ax3.add_patch(FancyArrowPatch((xp+bw/2+0.006, y),
                                          (xs[i+1]-bw/2-0.006, y),
                                          arrowstyle="-|>", mutation_scale=14,
                                          color="0.45", linewidth=1.2,
                                          transform=ax3.transAxes))
    ax3.text(0.5, 0.98,
             "Implication: use stage-wise competence signals for pre-screening instead of raw 5-episode task success.",
             transform=ax3.transAxes, ha="center", va="top",
             fontsize=12.5, fontweight="bold")

    save(fig, outdir, "slide2_surrogate_signals")
    plt.close(fig)


def write_summary(rows, status, stats, ex, outdir):
    s1 = stats["quick_task_vs_full_task"]
    s2 = stats["quick_grasp_vs_full_task"]
    s3 = stats["quick_grasp_vs_full_grasp"]
    lines = [
        "# Preliminary PPT summary",
        "",
        f"Valid completed checkpoints used: {len(rows)}",
        "",
        "## Current statistics",
        "",
        "| Comparison | n | Pearson r | Spearman rho |",
        "|---|---:|---:|---:|",
        f"| Quick task success -> full task success | {s1['n']} | {fmt(s1['pearson'])} | {fmt(s1['spearman'])} |",
        f"| Quick grasp -> full task success | {s2['n']} | {fmt(s2['pearson'])} | {fmt(s2['spearman'])} |",
        f"| Quick grasp -> full grasp | {s3['n']} | {fmt(s3['pearson'])} | {fmt(s3['spearman'])} |",
        "",
        "## Slide 1 text",
        "",
        "Title: Preliminary Validation: 5-Episode Success Does Not Reliably Rank Checkpoints",
        "",
        f"Across the {len(rows)} currently completed checkpoints, 5-episode task success shows weak "
        f"rank agreement with 200-episode task success (Spearman rho = {fmt(s1['spearman'])}). "
        "The coarse 20% resolution and sensitivity to sampled initial states produce both false negatives "
        "and overestimation.",
        "",
        "Bottom line: Raw 5-episode task success is too sparse and unstable for checkpoint ranking.",
        "",
        "## Slide 2 text",
        "",
        "Title: Preliminary Direction: Intermediate Grasp Signals Are More Informative",
        "",
        f"Quick grasp shows stronger agreement with full grasp (Pearson r = {fmt(s3['pearson'])}, "
        f"Spearman rho = {fmt(s3['spearman'])}) than raw quick task success does with full task success. "
        "This motivates a stage-wise surrogate that first evaluates intermediate manipulation competence "
        "and reserves expensive full-task evaluation for retained checkpoints.",
        "",
        "Bottom line: Use stage-wise competence signals rather than raw few-episode task success.",
        "",
        "## Camera-specific quick-task Spearman",
        "",
    ]
    for c, st in stats["camera_quick_task_vs_full_task"].items():
        lines.append(f"- {c}: n={st['n']}, rho={fmt(st['spearman'])}")
    lines += ["", "## Current status counts", ""]
    for k, v in sorted(status.items()):
        lines.append(f"- {k}: {v}")
    fn, ov = ex["false_negative"], ex["overestimate"]
    lines += ["", "## Examples", ""]
    if fn:
        lines.append(f"- False negative: {fn['model_name']}: quick={fn['quick_success']:.0f}%, full={fn['full_success']:.1f}%.")
    if ov:
        lines.append(f"- Overestimation: {ov['model_name']}: quick={ov['quick_success']:.0f}%, full={ov['full_success']:.1f}%.")
    (outdir / "ppt_preliminary_summary.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="quick40_pickplace")
    ap.add_argument("--eval-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    root = Path(args.eval_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    run_dir = root / args.run_name
    outdir = Path(args.output_dir) if args.output_dir else run_dir / "ppt_figures"
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows, status = load_rows(run_dir)
    if len(rows) < 3:
        raise RuntimeError(f"Need >=3 valid rows, found {len(rows)}")

    stats = {
        "valid_n": len(rows),
        "quick_task_vs_full_task": corr(rows, "quick_success", "full_success"),
        "quick_grasp_vs_full_task": corr(rows, "quick_grasp_success", "full_success"),
        "quick_grasp_vs_full_grasp": corr(rows, "quick_grasp_success", "full_grasp_success"),
        "camera_quick_task_vs_full_task": {},
        "status_counts": status,
    }
    for c in sorted({r["camera_group"] for r in rows}):
        sub = [r for r in rows if r["camera_group"] == c]
        stats["camera_quick_task_vs_full_task"][c] = corr(sub, "quick_success", "full_success")

    ex = examples(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    plot_slide1(rows, stats, ex, outdir)
    plot_slide2(rows, stats, outdir)
    write_summary(rows, status, stats, ex, outdir)
    (outdir / "figure_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"VALID CHECKPOINTS: {len(rows)}")
    print(f"Quick task -> full task Spearman: {fmt(stats['quick_task_vs_full_task']['spearman'])}")
    print(f"Quick grasp -> full grasp Spearman: {fmt(stats['quick_grasp_vs_full_grasp']['spearman'])}")
    print(f"OUTPUT: {outdir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
