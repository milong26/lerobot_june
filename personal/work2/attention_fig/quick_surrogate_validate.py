#!/usr/bin/env python3
"""
Fast retrospective validation of cheap checkpoint screening.

Default:
  - 7 existing corner 12k checkpoints
  - 5 fixed-seed episodes per checkpoint
  - <= 5 minutes per checkpoint
  - <= 38 minutes total
  - reuse existing attention/inference-trace aggregate statistics
  - compare all cheap signals with historical 200-episode success

Run:
    python personal/work2/attention_fig/quick_surrogate_validate.py --gpu-id 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from model_configs_12k_corner import CORNER_12K_MODEL_CONFIGS

ANALYSIS_DIR = SCRIPT_DIR / "analysis_12k_corner"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "quick_surrogate_eval"
RENAME_MAP = (
    '{"observation.images.top": "observation.images.camera1", '
    '"observation.images.wrist": "observation.images.camera2"}'
)


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def as_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if -1e-9 <= value <= 1.0 + 1e-9 else value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    return pearson(rankdata(x).tolist(), rankdata(y).tolist())


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def find_metric(obj: Any, candidate_keys: tuple[str, ...]) -> float | None:
    if isinstance(obj, dict):
        for key in candidate_keys:
            if key in obj:
                value = safe_float(obj[key])
                if value is not None:
                    return value
        for value in obj.values():
            found = find_metric(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_metric(value, candidate_keys)
            if found is not None:
                return found
    return None


def find_episode_lists(obj: Any) -> list[list[dict[str, Any]]]:
    found: list[list[dict[str, Any]]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "episodes" and isinstance(value, list) and value and isinstance(value[0], dict):
                found.append(value)
            else:
                found.extend(find_episode_lists(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(find_episode_lists(value))
    return found


def episode_rate(episodes: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for ep in episodes:
        for key in keys:
            if key not in ep:
                continue
            raw = ep[key]
            if isinstance(raw, bool):
                values.append(1.0 if raw else 0.0)
            else:
                value = safe_float(raw)
                if value is not None:
                    values.append(value)
            break
    return None if not values else float(np.mean(values) * 100.0)


def parse_eval_outputs(output_dir: Path, console: str) -> tuple[float | None, float | None, str | None]:
    json_paths = sorted(
        output_dir.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for path in json_paths:
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            continue

        task = find_metric(
            data,
            ("pc_success", "task_success_rate", "success_rate", "mean_success"),
        )
        grasp = find_metric(
            data,
            ("pc_grasp_success", "grasp_success_rate", "mean_grasp_success"),
        )
        task = as_percent(task)
        grasp = as_percent(grasp)

        for episodes in find_episode_lists(data):
            if task is None:
                task = episode_rate(episodes, ("success", "task_success", "is_success"))
            if grasp is None:
                grasp = episode_rate(
                    episodes,
                    ("grasp_success", "grasp_reached", "is_grasp_success"),
                )

        if task is not None:
            return task, grasp, str(path)

    # Fallback for versions of lerobot-eval that only print metrics.
    task = None
    grasp = None
    for pattern in (
        r"pc_success[^0-9+\-.]*([0-9]*\.?[0-9]+)",
        r"success_rate[^0-9+\-.]*([0-9]*\.?[0-9]+)",
    ):
        match = re.search(pattern, console, flags=re.IGNORECASE)
        if match:
            task = as_percent(float(match.group(1)))
            break

    match = re.search(
        r"pc_grasp_success[^0-9+\-.]*([0-9]*\.?[0-9]+)",
        console,
        flags=re.IGNORECASE,
    )
    if match:
        grasp = as_percent(float(match.group(1)))

    return task, grasp, "stdout" if task is not None else None


def build_eval_command(
    model_path: Path,
    output_dir: Path,
    episodes: int,
    start_seed: int,
    device: str,
) -> list[str]:
    return [
        "lerobot-eval",
        f"--policy.path={model_path}",
        "--env.type=metaworld",
        "--env.task=pick-place-v3",
        "--env.camera_name=corner,gripperPOV",
        "--env.use_self_mw=true",
        f"--eval.batch_size={min(8, episodes)}",
        f"--eval.n_episodes={episodes}",
        f"--policy.device={device}",
        "--policy.use_amp=false",
        f"--seed={start_seed}",
        f"--output_dir={output_dir}",
        f"--rename_map={RENAME_MAP}",
    ]


def load_internal_features() -> dict[str, dict[str, float]]:
    """Reuse existing offline attention/inference-trace results; do not rerun them."""
    features: dict[str, dict[str, float]] = {}

    for row in read_csv(ANALYSIS_DIR / "attention_aggregate.csv"):
        if (
            row.get("phase") == "initial"
            and row.get("layer") == "11"
            and row.get("attention_source") == "expert_cross"
        ):
            value = safe_float(row.get("visual_total_mean"))
            if value is not None:
                features.setdefault(row["model_name"], {})["attention_l11_visual"] = value

    for row in read_csv(ANALYSIS_DIR / "velocity_statistics.csv"):
        value = safe_float(row.get("v_t_norm_mean_of_means"))
        if value is not None:
            features.setdefault(row["model_name"], {})["v_t_norm_mean"] = value

    for row in read_csv(ANALYSIS_DIR / "heatmap_statistics.csv"):
        if (
            row.get("phase") == "inference_trace"
            and row.get("layer") == "11"
            and row.get("attention_source") == "expert_cross"
        ):
            value = safe_float(row.get("entropy_mean"))
            if value is not None:
                features.setdefault(row["model_name"], {})["trace_entropy_l11"] = value

    return features


def correlate(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    pairs = []
    for row in rows:
        x = safe_float(row.get(feature))
        y = safe_float(row.get("full_success"))
        if x is not None and y is not None:
            pairs.append((x, y))
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    return {
        "feature": feature,
        "n": len(pairs),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
    }


def topk_recall(
    rows: list[dict[str, Any]],
    full_k: int = 2,
    screen_k: int = 3,
) -> tuple[float | None, list[str], list[str]]:
    valid = [
        row for row in rows
        if safe_float(row.get("quick_success")) is not None
        and safe_float(row.get("full_success")) is not None
    ]
    if len(valid) < max(full_k, screen_k):
        return None, [], []

    full_top = [
        row["model_name"]
        for row in sorted(valid, key=lambda r: float(r["full_success"]), reverse=True)[:full_k]
    ]
    quick_top = [
        row["model_name"]
        for row in sorted(valid, key=lambda r: float(r["quick_success"]), reverse=True)[:screen_k]
    ]
    recall = len(set(full_top) & set(quick_top)) / float(full_k)
    return recall, full_top, quick_top


def build_report(
    rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    runtime_min: float,
    args: argparse.Namespace,
) -> str:
    quick = correlations[0]
    rho = quick["spearman"]
    recall, full_top, quick_top = topk_recall(rows)

    if quick["n"] < 4:
        verdict = "INCONCLUSIVE"
        conclusion = "有效 quick-eval 模型少于 4 个，目前不能判断预筛选是否可靠。"
    elif rho is not None and rho >= 0.60 and (recall is None or recall >= 0.50):
        verdict = "SUPPORTED"
        conclusion = (
            "小规模 fixed-seed eval 与历史 200-episode 结果具有较好的排序一致性，"
            "可以作为 full eval 前的 checkpoint 预筛选，但不能替代最终 full eval。"
        )
    elif rho is not None and rho >= 0.30:
        verdict = "PARTIAL"
        conclusion = (
            "小规模 fixed-seed eval 有一定排序信息，可用于粗筛，但当前证据不足以做强筛选。"
        )
    else:
        verdict = "NOT_SUPPORTED"
        conclusion = (
            "当前 5-episode 设置与 200-episode 排序一致性不足，不建议据此减少 full eval；"
            "应增加 quick episodes 或引入更强的 action-quality surrogate。"
        )

    lines = [
        "# Quick Surrogate Validation",
        "",
        f"- Runtime: {runtime_min:.1f} min",
        f"- Quick episodes/model: {args.episodes}",
        f"- Start seed: {args.start_seed}",
        f"- Parsed models: {quick['n']}/{len(rows)}",
        f"- Verdict: **{verdict}**",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "本验证测试的是低成本 checkpoint screening 是否能减少完整 eval 次数；"
        "attention / flow-matching dynamics 仅作为辅助诊断，不把 attention 直接解释为 success predictor。",
        "",
        "## Correlation with historical 200-episode success",
        "",
        "| Signal | n | Pearson | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for item in correlations:
        lines.append(
            f"| {item['feature']} | {item['n']} | {fmt(item['pearson'])} | {fmt(item['spearman'])} |"
        )

    lines += [
        "",
        "## Per-model results",
        "",
        "| Model | 200ep success | Quick success | Quick grasp | L11 visual | v_t norm | Trace entropy | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: float(r["full_success"]), reverse=True):
        lines.append(
            f"| {row['model_name']} | {fmt(safe_float(row.get('full_success')), 1)} | "
            f"{fmt(safe_float(row.get('quick_success')), 1)} | "
            f"{fmt(safe_float(row.get('quick_grasp_success')), 1)} | "
            f"{fmt(safe_float(row.get('attention_l11_visual')))} | "
            f"{fmt(safe_float(row.get('v_t_norm_mean')))} | "
            f"{fmt(safe_float(row.get('trace_entropy_l11')))} | "
            f"{row.get('eval_status')} |"
        )

    lines += ["", "## Top-k screening", ""]
    if recall is None:
        lines.append("有效模型不足，未计算 top-k recall。")
    else:
        lines.append(f"- Historical 200ep top-2: {', '.join(full_top)}")
        lines.append(f"- Quick-eval top-3: {', '.join(quick_top)}")
        lines.append(f"- Top-2 recall: {recall * 100:.1f}%")

    lines += [
        "",
        "## Decision rule",
        "",
        "- Spearman >= 0.60：可作为有用的 preliminary ranking signal。",
        "- 0.30 <= Spearman < 0.60：仅适合粗筛。",
        "- Spearman < 0.30：当前 quick-eval 配置不适合作为 checkpoint filter。",
        "- 即使 SUPPORTED，也只减少需要做 200-episode full eval 的 checkpoint 数量，不取消最终 full eval。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate cheap checkpoint screening against existing 200-episode results."
    )
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=10042)
    parser.add_argument("--time-budget-min", type=float, default=38.0)
    parser.add_argument("--per-model-timeout-sec", type=int, default=300)
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated model names or method names.",
    )
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.time_budget_min <= 0 or args.per_model_timeout_sec <= 0:
        raise ValueError("time budgets must be > 0")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    selected = list(CORNER_12K_MODEL_CONFIGS)
    if args.models:
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        selected = [
            cfg for cfg in selected
            if cfg["name"] in wanted or cfg["method"] in wanted
        ]
    if not selected:
        print("ERROR: no models selected")
        return 2

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    internal = load_internal_features()
    rows: list[dict[str, Any]] = []
    started = time.monotonic()

    print("=" * 80)
    print("QUICK SURROGATE VALIDATION")
    print("=" * 80)
    print(f"Models: {[cfg['name'] for cfg in selected]}")
    print(f"Episodes/model: {args.episodes}")
    print(f"Time budget: {args.time_budget_min:.1f} min")
    print(f"Per-model timeout: {args.per_model_timeout_sec}s")
    print(f"Output: {run_dir}")

    for idx, cfg in enumerate(selected, 1):
        elapsed = time.monotonic() - started
        remaining = args.time_budget_min * 60.0 - elapsed
        if remaining <= 30:
            print("GLOBAL TIME BUDGET reached; stop launching new evals.")
            break

        name = cfg["name"]
        model_path = PROJECT_ROOT / cfg["path"]
        model_output = run_dir / name
        model_output.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = {
            "model_name": name,
            "method": cfg["method"],
            "model_path": cfg["path"],
            "full_success": float(cfg["eval_task_success"]),
            "full_grasp_success": float(cfg.get("eval_grasp_success", np.nan)),
            "quick_success": None,
            "quick_grasp_success": None,
            "eval_status": "not_run",
            "result_source": None,
        }
        row.update(internal.get(name, {}))

        cmd = build_eval_command(
            model_path,
            model_output,
            args.episodes,
            args.start_seed,
            args.device,
        )
        print(f"\n[{idx}/{len(selected)}] {name}")
        print("  " + " ".join(cmd))

        model_started = time.monotonic()
        if args.dry_run:
            row["eval_status"] = "dry_run"
            row["runtime_sec"] = 0.0
            rows.append(row)
            continue

        timeout_sec = min(
            args.per_model_timeout_sec,
            max(30, int(remaining - 15)),
        )

        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            console = (result.stdout or "") + "\n" + (result.stderr or "")
            (model_output / "eval_console.log").write_text(console)

            if result.returncode != 0:
                row["eval_status"] = f"failed_rc_{result.returncode}"
                print(f"  FAILED rc={result.returncode}")
                print("\n".join(console.splitlines()[-10:]))
            else:
                task, grasp, source = parse_eval_outputs(model_output, console)
                row["quick_success"] = task
                row["quick_grasp_success"] = grasp
                row["result_source"] = source
                row["eval_status"] = "ok" if task is not None else "ok_parse_missing"
                print(
                    f"  quick success={fmt(task, 1)}%, "
                    f"grasp={fmt(grasp, 1)}%, source={source}"
                )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            (model_output / "eval_console.log").write_text(out + "\n" + err)
            row["eval_status"] = "timeout"
            print(f"  TIMEOUT after {timeout_sec}s; continue to next model.")
        except FileNotFoundError:
            row["eval_status"] = "lerobot_eval_not_found"
            print("  ERROR: lerobot-eval not found. Activate the lb_server environment.")
            row["runtime_sec"] = round(time.monotonic() - model_started, 2)
            rows.append(row)
            break

        row["runtime_sec"] = round(time.monotonic() - model_started, 2)
        rows.append(row)

    completed = {row["model_name"] for row in rows}
    for cfg in selected:
        if cfg["name"] in completed:
            continue
        row = {
            "model_name": cfg["name"],
            "method": cfg["method"],
            "model_path": cfg["path"],
            "full_success": float(cfg["eval_task_success"]),
            "full_grasp_success": float(cfg.get("eval_grasp_success", np.nan)),
            "quick_success": None,
            "quick_grasp_success": None,
            "eval_status": "not_run_budget",
            "result_source": None,
            "runtime_sec": 0.0,
        }
        row.update(internal.get(cfg["name"], {}))
        rows.append(row)

    correlations = [
        correlate(rows, "quick_success"),
        correlate(rows, "attention_l11_visual"),
        correlate(rows, "v_t_norm_mean"),
        correlate(rows, "trace_entropy_l11"),
    ]
    runtime_min = (time.monotonic() - started) / 60.0

    write_csv(run_dir / "quick_surrogate_results.csv", rows)
    (run_dir / "correlations.json").write_text(json.dumps(correlations, indent=2))
    (run_dir / "CONCLUSION.md").write_text(
        build_report(rows, correlations, runtime_min, args)
    )

    quick = correlations[0]
    recall, full_top, quick_top = topk_recall(rows)

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    for item in correlations:
        print(
            f"{item['feature']:>24}: n={item['n']}, "
            f"Pearson={fmt(item['pearson'])}, Spearman={fmt(item['spearman'])}"
        )

    print()
    rho = quick["spearman"]
    if quick["n"] < 4:
        print("Conclusion: INCONCLUSIVE")
    elif rho is not None and rho >= 0.60 and (recall is None or recall >= 0.50):
        print("Conclusion: SUPPORTED")
        print("Cheap fixed-seed eval can be used to pre-screen checkpoints before 200-episode full eval.")
    elif rho is not None and rho >= 0.30:
        print("Conclusion: PARTIAL")
        print("Use quick eval only for coarse screening; keep full eval for final decisions.")
    else:
        print("Conclusion: NOT_SUPPORTED")
        print("This 5-episode quick-eval setup is not reliable enough for checkpoint filtering.")

    if recall is not None:
        print(f"Historical top-2: {full_top}")
        print(f"Quick top-3: {quick_top}")
        print(f"Top-2 recall: {recall * 100:.1f}%")

    print(f"Runtime: {runtime_min:.1f} min")
    print(f"CSV: {run_dir / 'quick_surrogate_results.csv'}")
    print(f"Report: {run_dir / 'CONCLUSION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
