#!/usr/bin/env python3
"""
Fast retrospective validation of cheap checkpoint screening.

Main behavior:
- Discover ALL checkpoints under personal/work2/eval_model that already have a
  non-empty historical eval_episode_results.json (default >= 100 episodes).
- Recover checkpoint paths from nearby .tasks.jsonl files.
- Split the discovered pool deterministically with --shard INDEX/TOTAL.
- Run a very small fixed-seed eval for the assigned shard.
- Stream lerobot-eval stdout/stderr LIVE to the terminal and also save it.
- Compare quick-eval success with historical full-eval success.
- When all shards of the same --run-name are complete, automatically write a
  combined report across all shards.

Recommended two-GPU run:
  python -u personal/work2/attention_fig/quick_surrogate_validate.py \
      --gpu-id 0 --shard 0/2 --run-name quick40_20260920

  python -u personal/work2/attention_fig/quick_surrogate_validate.py \
      --gpu-id 1 --shard 1/2 --run-name quick40_20260920
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
WORK2_ROOT = PROJECT_ROOT / "personal" / "work2"
EVAL_ROOT = WORK2_ROOT / "eval_model"
ANALYSIS_DIR = SCRIPT_DIR / "analysis_12k_corner"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "quick_surrogate_eval"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from model_configs_12k_corner import CORNER_12K_MODEL_CONFIGS

RENAME_MAP = (
    '{"observation.images.top": "observation.images.camera1", '
    '"observation.images.wrist": "observation.images.camera2"}'
)


def out(message: str = "") -> None:
    print(message, flush=True)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model)


def parse_shard(spec: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)/(\d+)", spec.strip())
    if not match:
        raise ValueError("--shard must look like 0/2 or 1/2")
    index, total = int(match.group(1)), int(match.group(2))
    if total <= 0 or index < 0 or index >= total:
        raise ValueError(f"invalid --shard {spec}: require 0 <= index < total")
    return index, total


def infer_camera(model: str, checkpoint_path: str) -> str:
    text = f"{model} {checkpoint_path}".lower()
    if "corner3" in text:
        return "corner3,gripperPOV"
    if "corner2" in text:
        return "corner2,gripperPOV"
    return "corner,gripperPOV"


def infer_task_from_result_path(path: Path) -> str:
    # Typical: .../eval_results/pick-place-v3_0/eval_episode_results.json
    parent = path.parent.name
    match = re.match(r"(.+)_\d+$", parent)
    if match and match.group(1) not in {"eval_results"}:
        return match.group(1)
    return "pick-place-v3"


def load_episode_result(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception:
        return None

    episodes = data.get("episodes", [])
    n = int(data.get("num_episodes", len(episodes)) or len(episodes))
    if n <= 0 or not episodes:
        return None

    success_count = data.get("success_count")
    if success_count is None:
        success_count = sum(1 for ep in episodes if ep.get("success", False))
    grasp_count = data.get("grasp_success_count")
    if grasp_count is None:
        grasp_count = sum(1 for ep in episodes if ep.get("grasp_success", False))

    success_rate = data.get("success_rate")
    if success_rate is None:
        success_rate = success_count / max(1, n)
    grasp_rate = data.get("grasp_success_rate")
    if grasp_rate is None:
        grasp_rate = grasp_count / max(1, n)

    return {
        "n_episodes": n,
        "pc_success": float(success_rate) * 100.0,
        "pc_grasp_success": float(grasp_rate) * 100.0,
        "success_count": int(success_count),
        "grasp_success_count": int(grasp_count),
    }


def discover_task_records(eval_root: Path) -> dict[str, dict[str, Any]]:
    """Read all local .tasks.jsonl files and index by the generated task key."""
    records: dict[str, dict[str, Any]] = {}
    task_files = sorted(eval_root.rglob(".tasks.jsonl")) if eval_root.exists() else []

    for task_file in task_files:
        try:
            lines = task_file.read_text().splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            model = str(item.get("model", "")).strip()
            checkpoint = str(item.get("checkpoint", "")).strip()
            checkpoint_path = str(item.get("checkpoint_path", "")).strip().replace("//", "/")
            if not model or not checkpoint or not checkpoint_path:
                continue
            key = f"{sanitize_model(model)}__{sanitize_model(checkpoint)}"
            records[key] = {
                "model": model,
                "checkpoint": checkpoint,
                "checkpoint_path": checkpoint_path,
                "task_file": str(task_file.relative_to(PROJECT_ROOT)),
            }
    return records


def discover_historical_pool(
    eval_root: Path,
    min_historical_episodes: int,
) -> list[dict[str, Any]]:
    """Discover checkpoints with real historical episode-level eval results.

    We deliberately ignore the old summary.csv if it says n_episodes=0. Some
    sweep scripts selected the empty top-level eval_episode_results.json while
    the real 200-episode file lives one directory deeper.
    """
    task_records = discover_task_records(eval_root)
    best_by_checkpoint: dict[str, dict[str, Any]] = {}

    result_files = sorted(eval_root.rglob("eval_episode_results.json")) if eval_root.exists() else []
    out(f"[DISCOVER] .tasks.jsonl entries: {len(task_records)}")
    out(f"[DISCOVER] eval_episode_results.json files: {len(result_files)}")

    for result_path in result_files:
        metrics = load_episode_result(result_path)
        if not metrics or metrics["n_episodes"] < min_historical_episodes:
            continue

        matched_key = None
        for parent in result_path.parents:
            if parent.name in task_records:
                matched_key = parent.name
                break
            if parent == eval_root:
                break
        if matched_key is None:
            continue

        task = task_records[matched_key]
        checkpoint_path = task["checkpoint_path"]
        record = {
            "model": task["model"],
            "checkpoint": task["checkpoint"],
            "model_name": f"{task['model']}@{task['checkpoint']}",
            "checkpoint_path": checkpoint_path,
            "full_success": metrics["pc_success"],
            "full_grasp_success": metrics["pc_grasp_success"],
            "historical_n_episodes": metrics["n_episodes"],
            "historical_result_file": str(result_path.relative_to(PROJECT_ROOT)),
            "task": infer_task_from_result_path(result_path),
            "camera_name": infer_camera(task["model"], checkpoint_path),
        }

        previous = best_by_checkpoint.get(checkpoint_path)
        if previous is None or record["historical_n_episodes"] > previous["historical_n_episodes"]:
            best_by_checkpoint[checkpoint_path] = record

    pool = sorted(
        best_by_checkpoint.values(),
        key=lambda r: (r["model"], int(r["checkpoint"]) if r["checkpoint"].isdigit() else r["checkpoint"]),
    )

    # Fallback keeps the tool usable if historical result files are not present
    # on a machine, but normal work2 usage should take the branch above.
    if not pool:
        out("[DISCOVER] WARNING: no episode-level historical pool found; falling back to static 12k configs.")
        for cfg in CORNER_12K_MODEL_CONFIGS:
            pool.append(
                {
                    "model": cfg["method"],
                    "checkpoint": "012000",
                    "model_name": cfg["name"],
                    "checkpoint_path": cfg["path"],
                    "full_success": float(cfg["eval_task_success"]),
                    "full_grasp_success": float(cfg.get("eval_grasp_success", np.nan)),
                    "historical_n_episodes": 200,
                    "historical_result_file": "static_model_configs_12k_corner",
                    "task": "pick-place-v3",
                    "camera_name": cfg["camera_name"],
                }
            )
    return pool


def pool_fingerprint(pool: list[dict[str, Any]]) -> str:
    payload = [
        (
            r["model_name"],
            r["checkpoint_path"],
            round(float(r["full_success"]), 6),
            int(r["historical_n_episodes"]),
        )
        for r in pool
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def split_pool(pool: list[dict[str, Any]], shard_index: int, shard_total: int) -> list[dict[str, Any]]:
    """Contiguous deterministic split; shard sizes differ by at most one."""
    n = len(pool)
    base = n // shard_total
    extra = n % shard_total
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    return pool[start : start + size]


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

        task = as_percent(find_metric(
            data,
            ("pc_success", "task_success_rate", "success_rate", "mean_success"),
        ))
        grasp = as_percent(find_metric(
            data,
            ("pc_grasp_success", "grasp_success_rate", "mean_grasp_success"),
        ))

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
    task: str,
    camera_name: str,
) -> list[str]:
    return [
        "lerobot-eval",
        f"--policy.path={model_path}",
        "--env.type=metaworld",
        f"--env.task={task}",
        f"--env.camera_name={camera_name}",
        "--env.use_self_mw=true",
        f"--eval.batch_size={min(8, episodes)}",
        f"--eval.n_episodes={episodes}",
        f"--policy.device={device}",
        "--policy.use_amp=false",
        f"--seed={start_seed}",
        f"--output_dir={output_dir}",
        f"--rename_map={RENAME_MAP}",
    ]


def stream_process(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_sec: int,
    heartbeat_sec: int,
    label: str,
) -> tuple[int | None, str, bool]:
    """Run child process with immediate terminal mirroring and heartbeat."""
    child_env = env.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )

    chunks: queue.Queue[bytes | None] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                chunks.put(chunk)
        finally:
            chunks.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    collected = bytearray()
    reader_done = False
    started = time.monotonic()
    last_heartbeat = started
    timed_out = False

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        while True:
            try:
                item = chunks.get(timeout=0.25)
                if item is None:
                    reader_done = True
                else:
                    collected.extend(item)
                    log_file.write(item)
                    log_file.flush()
                    # Direct bytes write avoids Python text buffering when piped to tee.
                    sys.stdout.buffer.write(item)
                    sys.stdout.buffer.flush()
            except queue.Empty:
                pass

            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_sec:
                out(
                    f"\n[HEARTBEAT] {label}: child still running, "
                    f"elapsed={now - started:.0f}s / timeout={timeout_sec}s"
                )
                last_heartbeat = now

            if now - started >= timeout_sec and proc.poll() is None:
                timed_out = True
                out(f"\n[TIMEOUT] {label}: terminating child after {timeout_sec}s")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        proc.kill()
                break

            if proc.poll() is not None and reader_done and chunks.empty():
                break

        # Drain any final queued output.
        while True:
            try:
                item = chunks.get_nowait()
            except queue.Empty:
                break
            if item:
                collected.extend(item)
                log_file.write(item)
                sys.stdout.buffer.write(item)
        log_file.flush()
        sys.stdout.buffer.flush()

    return proc.poll(), collected.decode(errors="replace"), timed_out


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


def load_internal_features_by_path() -> dict[str, dict[str, float]]:
    """Reuse existing 12k attention features where checkpoint paths match."""
    by_model: dict[str, dict[str, float]] = {}

    for row in read_csv(ANALYSIS_DIR / "attention_aggregate.csv"):
        if (
            row.get("phase") == "initial"
            and row.get("layer") == "11"
            and row.get("attention_source") == "expert_cross"
        ):
            value = safe_float(row.get("visual_total_mean"))
            if value is not None:
                by_model.setdefault(row["model_name"], {})["attention_l11_visual"] = value

    for row in read_csv(ANALYSIS_DIR / "velocity_statistics.csv"):
        value = safe_float(row.get("v_t_norm_mean_of_means"))
        if value is not None:
            by_model.setdefault(row["model_name"], {})["v_t_norm_mean"] = value

    for row in read_csv(ANALYSIS_DIR / "heatmap_statistics.csv"):
        if (
            row.get("phase") == "inference_trace"
            and row.get("layer") == "11"
            and row.get("attention_source") == "expert_cross"
        ):
            value = safe_float(row.get("entropy_mean"))
            if value is not None:
                by_model.setdefault(row["model_name"], {})["trace_entropy_l11"] = value

    by_path: dict[str, dict[str, float]] = {}
    for cfg in CORNER_12K_MODEL_CONFIGS:
        path = str(Path(cfg["path"]))
        if cfg["name"] in by_model:
            by_path[path] = by_model[cfg["name"]]
    return by_path


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


def screening_recall(rows: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    valid = [
        row for row in rows
        if safe_float(row.get("quick_success")) is not None
        and safe_float(row.get("full_success")) is not None
    ]
    n = len(valid)
    if n < 5:
        return None, 0, 0
    full_k = max(2, math.ceil(n * 0.20))
    screen_k = max(full_k, math.ceil(n * 0.30))
    full_top = {
        row["model_name"]
        for row in sorted(valid, key=lambda r: float(r["full_success"]), reverse=True)[:full_k]
    }
    quick_top = {
        row["model_name"]
        for row in sorted(valid, key=lambda r: float(r["quick_success"]), reverse=True)[:screen_k]
    }
    return len(full_top & quick_top) / float(full_k), full_k, screen_k


def build_report(
    rows: list[dict[str, Any]],
    title: str,
    runtime_min: float | None = None,
) -> str:
    correlations = [
        correlate(rows, "quick_success"),
        correlate(rows, "attention_l11_visual"),
        correlate(rows, "v_t_norm_mean"),
        correlate(rows, "trace_entropy_l11"),
    ]
    quick = correlations[0]
    rho = quick["spearman"]
    recall, full_k, screen_k = screening_recall(rows)

    if quick["n"] < 5:
        verdict = "INCONCLUSIVE"
        conclusion = "有效 quick-eval 模型不足 5 个，暂时不能判断预筛选可靠性。"
    elif rho is not None and rho >= 0.60 and (recall is None or recall >= 0.60):
        verdict = "SUPPORTED"
        conclusion = (
            "小规模 fixed-seed eval 与历史 full eval 具有较好的排序一致性，"
            "可以作为 full eval 前的 checkpoint 预筛选，从而减少昂贵的完整评估次数。"
        )
    elif rho is not None and rho >= 0.30:
        verdict = "PARTIAL"
        conclusion = (
            "小规模 fixed-seed eval 包含一定排序信息，可用于粗筛，但不能据此直接替代完整评估。"
        )
    else:
        verdict = "NOT_SUPPORTED"
        conclusion = (
            "当前 quick-eval 设置与历史 full-eval 排序一致性不足，不适合作为可靠 checkpoint filter。"
        )

    lines = [
        f"# {title}",
        "",
        f"- Models with quick result: {quick['n']}/{len(rows)}",
        f"- Verdict: **{verdict}**",
    ]
    if runtime_min is not None:
        lines.append(f"- Runtime: {runtime_min:.1f} min")

    lines += [
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Correlation with historical full-eval success",
        "",
        "| Signal | n | Pearson | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for item in correlations:
        lines.append(
            f"| {item['feature']} | {item['n']} | {fmt(item['pearson'])} | {fmt(item['spearman'])} |"
        )

    if recall is not None:
        lines += [
            "",
            "## Screening retention",
            "",
            f"- Historical top-{full_k} retained by quick-eval top-{screen_k}: {recall * 100:.1f}%",
        ]

    lines += [
        "",
        "## Per-checkpoint results",
        "",
        "| Model/checkpoint | Hist n | Full success | Quick success | Quick grasp | Task | Camera | Status |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: float(r["full_success"]), reverse=True):
        lines.append(
            f"| {row['model_name']} | {row.get('historical_n_episodes', '')} | "
            f"{fmt(safe_float(row.get('full_success')), 1)} | "
            f"{fmt(safe_float(row.get('quick_success')), 1)} | "
            f"{fmt(safe_float(row.get('quick_grasp_success')), 1)} | "
            f"{row.get('task', '')} | {row.get('camera_name', '')} | "
            f"{row.get('eval_status', '')} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Spearman >= 0.60：可作为有用的 preliminary ranking signal。",
        "- 0.30 <= Spearman < 0.60：只适合粗筛。",
        "- Spearman < 0.30：当前 quick-eval 配置不适合作为 checkpoint filter。",
        "- 即使结果为 SUPPORTED，仍然保留最终 full eval，只减少需要 full eval 的 checkpoint 数量。",
        "- Attention / inference-trace 指标只对已有对应分析的少数 checkpoint 附加展示，不作为单独成功率预测器。",
    ]
    return "\n".join(lines) + "\n"


def maybe_build_combined(run_dir: Path, shard_total: int, fingerprint: str) -> None:
    metas = []
    all_rows: list[dict[str, Any]] = []
    for index in range(shard_total):
        shard_dir = run_dir / f"shard_{index}_of_{shard_total}"
        meta_path = shard_dir / "meta.json"
        result_path = shard_dir / "quick_surrogate_results.csv"
        if not meta_path.exists() or not result_path.exists():
            out(
                f"[COMBINE] waiting for shard {index}/{shard_total}; "
                f"combined report not ready yet."
            )
            return
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            return
        if meta.get("pool_fingerprint") != fingerprint:
            out(f"[COMBINE] shard {index} belongs to a different discovered pool; skip combine.")
            return
        metas.append(meta)
        all_rows.extend(read_csv(result_path))

    # Deduplicate by checkpoint path in case the split configuration changes.
    dedup: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        dedup[row["checkpoint_path"]] = row
    combined = list(dedup.values())

    write_csv(run_dir / "COMBINED_results.csv", combined)
    (run_dir / "COMBINED_CONCLUSION.md").write_text(
        build_report(combined, "Combined Quick Surrogate Validation")
    )
    correlations = [
        correlate(combined, "quick_success"),
        correlate(combined, "attention_l11_visual"),
        correlate(combined, "v_t_norm_mean"),
        correlate(combined, "trace_entropy_l11"),
    ]
    (run_dir / "COMBINED_correlations.json").write_text(
        json.dumps(correlations, indent=2)
    )
    quick = correlations[0]
    out("")
    out("=" * 80)
    out("COMBINED RESULT READY")
    out("=" * 80)
    out(
        f"Combined models: {quick['n']}/{len(combined)} | "
        f"quick-vs-full Spearman={fmt(quick['spearman'])}"
    )
    out(f"Report: {run_dir / 'COMBINED_CONCLUSION.md'}")
    out(f"CSV:    {run_dir / 'COMBINED_results.csv'}")
    out("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast live-output validation of checkpoint screening against historical evals."
    )
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1000,
        help="Default 1000 matches the historical checkpoint sweep seed prefix.",
    )
    parser.add_argument("--time-budget-min", type=float, default=38.0)
    parser.add_argument("--per-model-timeout-sec", type=int, default=240)
    parser.add_argument("--heartbeat-sec", type=int, default=10)
    parser.add_argument(
        "--min-historical-episodes",
        type=int,
        default=100,
        help="Only use checkpoints with at least this many historical eval episodes.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default="0/1",
        help="Deterministic pool split, e.g. 0/2 and 1/2.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="quick40",
        help="Use the SAME run-name for all shards so they can auto-combine.",
    )
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional debug limit applied before sharding.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    shard_index, shard_total = parse_shard(args.shard)
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.time_budget_min <= 0 or args.per_model_timeout_sec <= 0:
        raise ValueError("time budgets must be > 0")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ["PYTHONUNBUFFERED"] = "1"

    out("=" * 80)
    out("QUICK SURROGATE VALIDATION - DISCOVERY")
    out("=" * 80)
    pool = discover_historical_pool(EVAL_ROOT, args.min_historical_episodes)
    if args.max_models is not None:
        pool = pool[: args.max_models]

    fingerprint = pool_fingerprint(pool)
    selected = split_pool(pool, shard_index, shard_total)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / args.run_name
    shard_dir = run_dir / f"shard_{shard_index}_of_{shard_total}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    out(f"[DISCOVER] historical checkpoint pool: {len(pool)}")
    out(
        f"[SHARD] {shard_index}/{shard_total}: assigned {len(selected)} models "
        f"(other shards receive the remainder)"
    )
    out(f"[SHARD] pool fingerprint: {fingerprint}")
    out(f"[GPU] physical id requested: {args.gpu_id}")
    out(f"[BUDGET] episodes/model={args.episodes}, total={args.time_budget_min:.1f} min, "
        f"per-model timeout={args.per_model_timeout_sec}s")
    out(f"[OUTPUT] {shard_dir}")
    out("")
    out("[ASSIGNED MODELS]")
    for i, item in enumerate(selected, 1):
        out(
            f"  {i:03d}/{len(selected):03d}  {item['model_name']}  "
            f"hist={item['full_success']:.1f}%/{item['historical_n_episodes']}ep  "
            f"camera={item['camera_name']}"
        )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "shard_index": shard_index,
        "shard_total": shard_total,
        "pool_size": len(pool),
        "assigned_size": len(selected),
        "pool_fingerprint": fingerprint,
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "min_historical_episodes": args.min_historical_episodes,
        "gpu_id": args.gpu_id,
    }
    (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    internal_by_path = load_internal_features_by_path()
    rows: list[dict[str, Any]] = []
    started = time.monotonic()

    for idx, item in enumerate(selected, 1):
        global_elapsed = time.monotonic() - started
        remaining = args.time_budget_min * 60.0 - global_elapsed
        if remaining <= 20:
            out("[BUDGET] global time budget reached; stop launching new models.")
            break

        model_name = item["model_name"]
        model_path = PROJECT_ROOT / item["checkpoint_path"]
        safe_name = sanitize_model(model_name)
        model_output = shard_dir / "eval_outputs" / safe_name
        log_path = shard_dir / "logs" / f"{safe_name}.log"
        model_output.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = dict(item)
        row.update(
            {
                "quick_success": None,
                "quick_grasp_success": None,
                "eval_status": "not_run",
                "result_source": None,
                "runtime_sec": 0.0,
            }
        )
        row.update(internal_by_path.get(str(Path(item["checkpoint_path"])), {}))

        cmd = build_eval_command(
            model_path=model_path,
            output_dir=model_output,
            episodes=args.episodes,
            start_seed=args.start_seed,
            device=args.device,
            task=item["task"],
            camera_name=item["camera_name"],
        )

        out("")
        out("=" * 80)
        out(
            f"[MODEL {idx}/{len(selected)}] {model_name} | "
            f"historical={item['full_success']:.1f}% "
            f"({item['historical_n_episodes']} episodes)"
        )
        out(f"[PATH] {model_path}")
        out(f"[CMD] {' '.join(cmd)}")
        out("=" * 80)

        if args.dry_run:
            row["eval_status"] = "dry_run"
            rows.append(row)
            continue

        if not model_path.exists():
            row["eval_status"] = "checkpoint_missing"
            out(f"[SKIP] checkpoint path does not exist locally: {model_path}")
            rows.append(row)
            continue

        timeout_sec = min(
            args.per_model_timeout_sec,
            max(20, int(remaining - 10)),
        )
        model_started = time.monotonic()

        try:
            rc, console, timed_out = stream_process(
                cmd=cmd,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                log_path=log_path,
                timeout_sec=timeout_sec,
                heartbeat_sec=args.heartbeat_sec,
                label=model_name,
            )
            if timed_out:
                row["eval_status"] = "timeout"
            elif rc != 0:
                row["eval_status"] = f"failed_rc_{rc}"
                out(f"[FAILED] {model_name}: return code={rc}")
            else:
                task_success, grasp_success, source = parse_eval_outputs(model_output, console)
                row["quick_success"] = task_success
                row["quick_grasp_success"] = grasp_success
                row["result_source"] = source
                row["eval_status"] = "ok" if task_success is not None else "ok_parse_missing"
                out(
                    f"[DONE] {model_name}: quick_success={fmt(task_success, 1)}% "
                    f"quick_grasp={fmt(grasp_success, 1)}%"
                )
        except FileNotFoundError:
            row["eval_status"] = "lerobot_eval_not_found"
            out("[FATAL] lerobot-eval not found. Activate lb_server and pull the latest branch.")
            row["runtime_sec"] = round(time.monotonic() - model_started, 2)
            rows.append(row)
            break
        except KeyboardInterrupt:
            row["eval_status"] = "interrupted"
            row["runtime_sec"] = round(time.monotonic() - model_started, 2)
            rows.append(row)
            write_csv(shard_dir / "quick_surrogate_results.csv", rows)
            out("[INTERRUPT] partial CSV saved.")
            raise

        row["runtime_sec"] = round(time.monotonic() - model_started, 2)
        rows.append(row)

        # Persist after every model so results survive interruption.
        write_csv(shard_dir / "quick_surrogate_results.csv", rows)
        done = sum(1 for r in rows if r.get("eval_status") == "ok")
        out(
            f"[PROGRESS] shard={shard_index}/{shard_total} processed={len(rows)}/{len(selected)} "
            f"successful_quick_results={done} "
            f"global_elapsed={(time.monotonic() - started) / 60.0:.1f} min"
        )

    completed_paths = {r["checkpoint_path"] for r in rows}
    for item in selected:
        if item["checkpoint_path"] in completed_paths:
            continue
        row = dict(item)
        row.update(
            {
                "quick_success": None,
                "quick_grasp_success": None,
                "eval_status": "not_run_budget",
                "result_source": None,
                "runtime_sec": 0.0,
            }
        )
        row.update(internal_by_path.get(str(Path(item["checkpoint_path"])), {}))
        rows.append(row)

    runtime_min = (time.monotonic() - started) / 60.0
    write_csv(shard_dir / "quick_surrogate_results.csv", rows)
    (shard_dir / "CONCLUSION.md").write_text(
        build_report(
            rows,
            title=f"Quick Surrogate Validation - shard {shard_index}/{shard_total}",
            runtime_min=runtime_min,
        )
    )
    correlations = [
        correlate(rows, "quick_success"),
        correlate(rows, "attention_l11_visual"),
        correlate(rows, "v_t_norm_mean"),
        correlate(rows, "trace_entropy_l11"),
    ]
    (shard_dir / "correlations.json").write_text(json.dumps(correlations, indent=2))

    quick = correlations[0]
    out("")
    out("=" * 80)
    out(f"SHARD {shard_index}/{shard_total} FINISHED")
    out("=" * 80)
    out(
        f"Quick results: {quick['n']}/{len(rows)} | "
        f"quick-vs-full Spearman={fmt(quick['spearman'])}"
    )
    out(f"Runtime: {runtime_min:.1f} min")
    out(f"Report: {shard_dir / 'CONCLUSION.md'}")
    out(f"CSV:    {shard_dir / 'quick_surrogate_results.csv'}")
    out("=" * 80)

    maybe_build_combined(run_dir, shard_total, fingerprint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
