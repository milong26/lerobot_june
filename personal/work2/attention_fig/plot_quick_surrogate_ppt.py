#!/usr/bin/env python3
"""Create two simple PPT-ready scatter figures from current quick-eval results.

Important:
- Historical full evals used camera_name="corner,gripperPOV" for all checkpoints.
- Older partial quick-eval rows produced with corner2/corner3 cameras are excluded.
- Only eval_status=="ok" rows with matching camera configuration are plotted.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_ROOT = SCRIPT_DIR / "quick_surrogate_eval"
EXPECTED_CAMERA = "corner,gripperPOV"


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


def load_rows(run_dir: Path):
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
    excluded_camera = 0
    valid: list[dict[str, Any]] = []

    for r in by_path.values():
        s = r.get("eval_status", "")
        status[s] = status.get(s, 0) + 1
        if s != "ok":
            continue

        if r.get("camera_name", "") != EXPECTED_CAMERA:
            excluded_camera += 1
            continue

        vals = {}
        good = True
        for k in ("quick_success", "quick_grasp_success", "full_success", "full_grasp_success"):
            v = sf(r.get(k))
            if v is None:
                good = False
                break
            vals[k] = v
        if not good:
            continue

        item = dict(r)
        item.update(vals)
        valid.append(item)

    valid.sort(key=lambda r: (r.get("model", ""), int(r.get("checkpoint", "0"))))
    return valid, status, excluded_camera


def deterministic_jitter(n: int, scale: float = 1.0) -> np.ndarray:
    pattern = np.array([-2, -1, 0, 1, 2], dtype=float) * scale
    return np.array([pattern[i % len(pattern)] for i in range(n)])


def linear_fit_line(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or np.std(x) < 1e-12:
        return None
    coef = np.polyfit(x, y, deg=1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    yy = coef[0] * xx + coef[1]
    return xx, yy


def save(fig, outdir: Path, stem: str):
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


def plot_task_scatter(rows, stat, outdir):
    fig, ax = plt.subplots(figsize=(10.8, 6.3))

    x = np.asarray([r["quick_success"] for r in rows], dtype=float)
    y = np.asarray([r["full_success"] for r in rows], dtype=float)
    xj = x + deterministic_jitter(len(x), scale=0.8)

    ax.scatter(xj, y, s=78, alpha=0.82, edgecolors="white", linewidths=0.7)

    fit = linear_fit_line(x, y)
    if fit is not None:
        ax.plot(fit[0], fit[1], linewidth=2, label="Linear trend")

    ax.set_xlim(-7, 107)
    ymax = max(40, math.ceil(max(y) / 10) * 10 + 5)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("5-episode task success (%)")
    ax.set_ylabel("200-episode task success (%)")
    ax.set_title("Few-Episode Task Success Is Not a Reliable Checkpoint Ranking Signal",
                 fontsize=17, fontweight="bold")
    ax.grid(alpha=0.18)

    ax.text(
        0.98, 0.96,
        f"n = {stat['n']}\nPearson r = {fmt(stat['pearson'])}\nSpearman ρ = {fmt(stat['spearman'])}",
        transform=ax.transAxes, ha="right", va="top", fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8")
    )

    ax.text(
        0.02, 0.04,
        "Each point = one checkpoint\nOnly matched camera configuration is included",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5
    )

    save(fig, outdir, "slide1_task_success_scatter")
    plt.close(fig)


def plot_grasp_scatter(rows, task_stat, grasp_stat, outdir):
    fig, ax = plt.subplots(figsize=(10.8, 6.3))

    x = np.asarray([r["quick_grasp_success"] for r in rows], dtype=float)
    y = np.asarray([r["full_grasp_success"] for r in rows], dtype=float)
    xj = x + deterministic_jitter(len(x), scale=0.65)

    ax.scatter(xj, y, s=78, alpha=0.82, edgecolors="white", linewidths=0.7)

    fit = linear_fit_line(x, y)
    if fit is not None:
        ax.plot(fit[0], fit[1], linewidth=2, label="Linear trend")

    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("5-episode grasp success (%)")
    ax.set_ylabel("200-episode grasp success (%)")
    ax.set_title("Intermediate Grasp Success Provides a More Informative Early Signal",
                 fontsize=17, fontweight="bold")
    ax.grid(alpha=0.18)

    ax.text(
        0.98, 0.96,
        f"n = {grasp_stat['n']}\nPearson r = {fmt(grasp_stat['pearson'])}\nSpearman ρ = {fmt(grasp_stat['spearman'])}",
        transform=ax.transAxes, ha="right", va="top", fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8")
    )

    ax.text(
        0.02, 0.04,
        f"Task-success ranking: ρ = {fmt(task_stat['spearman'])}\n"
        f"Grasp-success ranking: ρ = {fmt(grasp_stat['spearman'])}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="0.96", edgecolor="0.82")
    )

    save(fig, outdir, "slide2_grasp_success_scatter")
    plt.close(fig)


def write_summary(rows, status, excluded_camera, task_stat, grasp_stat, outdir):
    lines = [
        "# Preliminary PPT summary",
        "",
        f"Valid matched-camera checkpoints used: **{len(rows)}**",
        f"Rows excluded because an older quick run used a mismatched camera: **{excluded_camera}**",
        "",
        "## Slide 1",
        "",
        "**Title:** Few-Episode Task Success Is Not a Reliable Checkpoint Ranking Signal",
        "",
        f"Across the currently valid {len(rows)} checkpoints, 5-episode task success shows "
        f"weak agreement with the historical 200-episode task success "
        f"(Pearson r = {fmt(task_stat['pearson'])}, Spearman rho = {fmt(task_stat['spearman'])}).",
        "",
        "The five-episode estimate changes in 20-percentage-point increments and is highly sensitive "
        "to the sampled initial states, so it can mis-rank checkpoints.",
        "",
        "**Takeaway:** Raw 5-episode task success is insufficient as a checkpoint filter.",
        "",
        "## Slide 2",
        "",
        "**Title:** Intermediate Grasp Success Provides a More Informative Early Signal",
        "",
        f"Using the same matched-camera checkpoints, quick grasp success shows stronger agreement "
        f"with full-evaluation grasp success "
        f"(Pearson r = {fmt(grasp_stat['pearson'])}, Spearman rho = {fmt(grasp_stat['spearman'])}).",
        "",
        f"For reference, task-success rank correlation is rho = {fmt(task_stat['spearman'])}, "
        f"whereas grasp-success rank correlation is rho = {fmt(grasp_stat['spearman'])}.",
        "",
        "**Takeaway:** A useful low-cost surrogate should use intermediate task competence "
        "rather than only final success from a few episodes.",
        "",
        "## Current status counts",
        "",
    ]
    for k, v in sorted(status.items()):
        lines.append(f"- {k}: {v}")
    (outdir / "ppt_preliminary_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    outdir = Path(args.output_dir) if args.output_dir else run_dir / "ppt_figures_simple"
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows, status, excluded_camera = load_rows(run_dir)
    if len(rows) < 3:
        raise RuntimeError(
            f"Only {len(rows)} matched-camera valid rows. "
            "Rerun quick eval after pulling the fixed camera code."
        )

    task_stat = corr(rows, "quick_success", "full_success")
    grasp_stat = corr(rows, "quick_grasp_success", "full_grasp_success")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plot_task_scatter(rows, task_stat, outdir)
    plot_grasp_scatter(rows, task_stat, grasp_stat, outdir)
    write_summary(rows, status, excluded_camera, task_stat, grasp_stat, outdir)

    stats = {
        "valid_matched_camera_n": len(rows),
        "excluded_camera_mismatch_n": excluded_camera,
        "expected_camera": EXPECTED_CAMERA,
        "quick_task_vs_full_task": task_stat,
        "quick_grasp_vs_full_grasp": grasp_stat,
        "status_counts": status,
    }
    (outdir / "figure_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"MATCHED-CAMERA VALID CHECKPOINTS: {len(rows)}")
    print(f"EXCLUDED CAMERA-MISMATCH ROWS: {excluded_camera}")
    print(f"Task success Spearman:  {fmt(task_stat['spearman'])}")
    print(f"Grasp success Spearman: {fmt(grasp_stat['spearman'])}")
    print(f"OUTPUT: {outdir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
