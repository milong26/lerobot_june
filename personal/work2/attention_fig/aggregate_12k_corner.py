#!/usr/bin/env python
"""
Aggregate post-hoc analysis for 12k corner 7-model attention diagnosis.

This script reads already-generated data from:
  - personal/work2/attention_fig/result_12k_corner (probe/rollout)
  - personal/work2/attention_fig/result_inference_trace_12k_corner (inference_trace)

It does NOT run models, create environments, or re-evaluate.
All outputs go to personal/work2/attention_fig/analysis_12k_corner.

This is purely offline statistics aggregation.
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from model_configs_12k_corner import CORNER_12K_MODEL_CONFIGS

RESULT_DIR = PROJECT_ROOT / "personal/work2/attention_fig/result_12k_corner"
INFERENCE_TRACE_DIR = PROJECT_ROOT / "personal/work2/attention_fig/result_inference_trace_12k_corner"
OUTPUT_DIR = PROJECT_ROOT / "personal/work2/attention_fig/analysis_12k_corner"


def _build_model_lookup():
    """Build lookup dicts from CORNER_12K_MODEL_CONFIGS."""
    lookup = {}
    for cfg in CORNER_12K_MODEL_CONFIGS:
        lookup[cfg["name"]] = cfg
    return lookup


def ensure_output_dir():
    """Create the analysis output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")


def load_model_metadata():
    """Read CORNER_12K_MODEL_CONFIGS and return model info list.

    eval_task_success and eval_grasp_success are ONLY for metadata tables
    and final report. They MUST NOT participate in any sorting, filtering,
    or statistical inference.
    """
    lookup = _build_model_lookup()
    records = []
    for cfg in CORNER_12K_MODEL_CONFIGS:
        records.append({
            "name": cfg["name"],
            "method": cfg["method"],
            "path": cfg["path"],
            "eval_task_success": cfg["eval_task_success"],
            "eval_grasp_success": cfg["eval_grasp_success"],
        })
    print(f"  Loaded {len(records)} model metadata entries from model_configs_12k_corner.py")
    return records


def _discover_seed_dirs(model_dir):
    """Return sorted list of (seed_int, seed_dir) tuples."""
    seeds = []
    if not model_dir.exists():
        return seeds
    for entry in sorted(model_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("seed_"):
            try:
                seed_val = int(entry.name.split("_")[1])
                seeds.append((seed_val, entry))
            except (IndexError, ValueError):
                continue
    return seeds


def _read_csv_rows(csv_path):
    """Read a CSV file and return list of dicts."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(val):
    """Convert to float or return None."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    """Convert to int or return None."""
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _write_csv(output_path, rows):
    """Write list of dicts to CSV."""
    if not rows:
        print(f"  WARNING: No data to write to {output_path}")
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {output_path} with {len(rows)} rows")


def _agg_stats(values):
    """Return (mean, std, count) for a list of floats."""
    if not values:
        return None, None, 0
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), len(arr)


def aggregate_attention_metrics():
    """Read attention_summary.csv from result_12k_corner and aggregate.

    For each model/seed, plot_attention.py writes attention_summary.csv.
    We aggregate across all seeds by model_name/method/phase/layer/attention_source,
    computing mean/std/count for camera1_mass, camera2_mass, visual_total,
    language_mass, state_mass.
    """
    print("  Scanning attention_summary.csv files...")

    all_rows = []
    lookup = _build_model_lookup()

    for cfg in CORNER_12K_MODEL_CONFIGS:
        model_name = cfg["name"]
        model_dir = RESULT_DIR / model_name
        for seed_val, seed_dir in _discover_seed_dirs(model_dir):
            csv_path = seed_dir / "attention_summary.csv"
            if csv_path.exists():
                rows = _read_csv_rows(csv_path)
                for row in rows:
                    row["_seed"] = seed_val
                    all_rows.append(row)

    if not all_rows:
        print("  WARNING: No per-seed attention_summary.csv data found. Trying root-level CSV...")
        root_csv = RESULT_DIR / "attention_summary.csv"
        if root_csv.exists():
            all_rows = _read_csv_rows(root_csv)

    if not all_rows:
        print("  WARNING: No attention data found at all.")
        return []

    aggregated = {}
    for row in all_rows:
        model_name = row.get("model_name", "")
        method = row.get("method", "")
        phase = row.get("phase", "")
        layer = row.get("layer", "")
        attention_source = row.get("attention_source", "")
        key = (model_name, method, phase, layer, attention_source)

        if key not in aggregated:
            aggregated[key] = {
                "model_name": model_name,
                "method": method,
                "phase": phase,
                "layer": _safe_int(layer),
                "attention_source": attention_source,
                "camera1_mass": [],
                "camera2_mass": [],
                "language_mass": [],
                "state_mass": [],
                "seed_values": [],
            }

        agg = aggregated[key]
        for field in ["camera1_mass", "camera2_mass", "language_mass", "state_mass"]:
            val = _safe_float(row.get(field))
            if val is not None:
                agg[field].append(val)
        agg["seed_values"].append(row.get("_seed"))

    output_rows = []
    for key, agg in aggregated.items():
        mean1, std1, _ = _agg_stats(agg["camera1_mass"])
        mean2, std2, _ = _agg_stats(agg["camera2_mass"])
        mean_lang, std_lang, _ = _agg_stats(agg["language_mass"])
        mean_state, std_state, _ = _agg_stats(agg["state_mass"])

        cam1 = agg["camera1_mass"]
        cam2 = agg["camera2_mass"]
        visual_total = [a + b for a, b in zip(cam1, cam2) if a is not None and b is not None]
        mean_vis, std_vis, _ = _agg_stats(visual_total)

        output_rows.append({
            "model_name": agg["model_name"],
            "method": agg["method"],
            "phase": agg["phase"],
            "layer": agg["layer"],
            "attention_source": agg["attention_source"],
            "count": len(agg["seed_values"]),
            "camera1_mass_mean": mean1,
            "camera1_mass_std": std1,
            "camera2_mass_mean": mean2,
            "camera2_mass_std": std2,
            "visual_total_mean": mean_vis,
            "visual_total_std": std_vis,
            "language_mass_mean": mean_lang,
            "language_mass_std": std_lang,
            "state_mass_mean": mean_state,
            "state_mass_std": std_state,
        })

    _write_csv(OUTPUT_DIR / "attention_aggregate.csv", output_rows)
    return output_rows


def aggregate_phase_reach():
    """Read rollout metadata.json and compute phase reach ratios.

    This is ONLY behavioral diagnostics. No causal analysis with success metrics.
    """
    print("  Scanning rollout metadata for phase reach...")

    all_records = []
    for cfg in CORNER_12K_MODEL_CONFIGS:
        model_name = cfg["name"]
        method = cfg["method"]
        model_dir = RESULT_DIR / model_name
        for seed_val, seed_dir in _discover_seed_dirs(model_dir):
            meta_path = seed_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path, "r") as f:
                meta = json.load(f)

            phases = meta.get("phases", {})
            for phase_name in ["initial", "pre_grasp", "post_grasp", "pre_place"]:
                phase_data = phases.get(phase_name, {})
                reached = phase_data.get("reached", False)
                all_records.append({
                    "model_name": model_name,
                    "method": method,
                    "seed": seed_val,
                    "phase": phase_name,
                    "reached": 1 if reached else 0,
                })

    if not all_records:
        print("  WARNING: No rollout phase data found.")
        return []

    agg = defaultdict(lambda: {"reached": 0, "total": 0})
    for rec in all_records:
        key = (rec["model_name"], rec["method"], rec["phase"])
        agg[key]["total"] += 1
        agg[key]["reached"] += rec["reached"]

    output_rows = []
    for (model_name, method, phase), stats in sorted(agg.items()):
        output_rows.append({
            "model_name": model_name,
            "method": method,
            "phase": phase,
            "reached_count": stats["reached"],
            "total_count": stats["total"],
            "reached_ratio": stats["reached"] / stats["total"] if stats["total"] > 0 else 0.0,
        })

    _write_csv(OUTPUT_DIR / "phase_reach_aggregate.csv", output_rows)
    return output_rows


def aggregate_heatmap_statistics():
    """Read npz heatmap files and compute spatial statistics.

    For each model/seed/phase/layer/camera, compute:
      entropy, max_attention, center_x, center_y, mean_attention, std_attention
    Then aggregate across seeds.
    """
    print("  Scanning heatmap npz files...")

    all_records = []
    search_dirs = [RESULT_DIR, INFERENCE_TRACE_DIR]

    for cfg in CORNER_12K_MODEL_CONFIGS:
        model_name = cfg["name"]
        method = cfg["method"]

        for search_dir in search_dirs:
            model_dir = search_dir / model_name
            if not model_dir.exists():
                continue

            for seed_val, seed_dir in _discover_seed_dirs(model_dir):
                hm_root = seed_dir / "heatmaps"
                if not hm_root.exists():
                    continue

                for npz_file in sorted(hm_root.rglob("*.npz")):
                    try:
                        data = np.load(str(npz_file), allow_pickle=True)
                        attention_map = data.get("attention_map")
                        if attention_map is None:
                            continue

                        npz_model = str(data.get("model_name", model_name))
                        npz_method = str(data.get("method", method))
                        npz_seed = int(data.get("seed", seed_val))
                        npz_phase = str(data.get("phase", "unknown"))
                        npz_layer = int(data.get("layer", -1))
                        npz_source = str(data.get("attention_source", "unknown"))
                        npz_camera = str(data.get("camera", "unknown"))

                        max_attn = float(np.max(attention_map))
                        mean_attn = float(np.mean(attention_map))
                        std_attn = float(np.std(attention_map))

                        if attention_map.ndim == 2:
                            h, w = attention_map.shape
                            y_coords, x_coords = np.mgrid[0:h, 0:w]
                            total = float(np.sum(attention_map))
                            norm_attn = attention_map / (total + 1e-10)
                            center_x = float(np.sum(x_coords * norm_attn))
                            center_y = float(np.sum(y_coords * norm_attn))

                            flat_norm = norm_attn.flatten()
                            flat_norm = flat_norm[flat_norm > 0]
                            entropy = float(-np.sum(flat_norm * np.log(flat_norm + 1e-10))) if len(flat_norm) > 0 else 0.0
                        else:
                            center_x = None
                            center_y = None
                            entropy = None

                        all_records.append({
                            "model_name": npz_model,
                            "method": npz_method,
                            "seed": npz_seed,
                            "phase": npz_phase,
                            "layer": npz_layer,
                            "attention_source": npz_source,
                            "camera": npz_camera,
                            "max_attention": max_attn,
                            "mean_attention": mean_attn,
                            "std_attention": std_attn,
                            "center_x": center_x,
                            "center_y": center_y,
                            "entropy": entropy,
                        })
                    except Exception as e:
                        print(f"  WARNING: Failed to read {npz_file}: {e}")

    if not all_records:
        print("  WARNING: No heatmap npz files found.")
        return []

    agg = defaultdict(lambda: {
        "max_attention": [], "mean_attention": [], "std_attention": [],
        "center_x": [], "center_y": [], "entropy": [],
    })

    for rec in all_records:
        key = (rec["model_name"], rec["method"], rec["phase"], rec["layer"],
               rec["attention_source"], rec["camera"])
        a = agg[key]
        a["max_attention"].append(rec["max_attention"])
        a["mean_attention"].append(rec["mean_attention"])
        a["std_attention"].append(rec["std_attention"])
        if rec["center_x"] is not None:
            a["center_x"].append(rec["center_x"])
        if rec["center_y"] is not None:
            a["center_y"].append(rec["center_y"])
        if rec["entropy"] is not None:
            a["entropy"].append(rec["entropy"])

    output_rows = []
    for (model_name, method, phase, layer, source, camera), a in sorted(agg.items()):
        count = len(a["max_attention"])
        mean_max, std_max, _ = _agg_stats(a["max_attention"])
        mean_mean, std_mean, _ = _agg_stats(a["mean_attention"])
        mean_std, std_std, _ = _agg_stats(a["std_attention"])
        mean_cx, std_cx, _ = _agg_stats(a["center_x"])
        mean_cy, std_cy, _ = _agg_stats(a["center_y"])
        mean_ent, std_ent, _ = _agg_stats(a["entropy"])

        output_rows.append({
            "model_name": model_name,
            "method": method,
            "phase": phase,
            "layer": layer,
            "attention_source": source,
            "camera": camera,
            "count": count,
            "max_attention_mean": mean_max,
            "max_attention_std": std_max,
            "mean_attention_mean": mean_mean,
            "mean_attention_std": std_mean,
            "std_attention_mean": mean_std,
            "std_attention_std": std_std,
            "center_x_mean": mean_cx,
            "center_x_std": std_cx,
            "center_y_mean": mean_cy,
            "center_y_std": std_cy,
            "entropy_mean": mean_ent,
            "entropy_std": std_ent,
        })

    _write_csv(OUTPUT_DIR / "heatmap_statistics.csv", output_rows)
    return output_rows


def aggregate_inference_trace():
    """Read inference trace data from per-seed directories and aggregate x_t_l2, v_t_l2, v_t_cosine, hidden_cosine, etc.

    Reads from:
      - INFERENCE_TRACE_DIR/model_name/seed_xxxx/ (trace.pt + metadata.json)
      - INFERENCE_TRACE_DIR/model_name/seed_xxxx/inference_attention_trace.csv (if exists)
      - INFERENCE_TRACE_DIR/plots/seed_xxxx/*pairwise_action_divergence.csv (for pairwise data)

    Groups by model_name/denoise_index for x_t_l2, v_t_l2, v_t_cosine, hidden_cosine.
    """
    print("  Scanning inference trace data from per-seed directories...")

    all_rows = []

    for cfg in CORNER_12K_MODEL_CONFIGS:
        model_name = cfg["name"]
        model_dir = INFERENCE_TRACE_DIR / model_name
        if not model_dir.exists():
            continue

        for seed_val, seed_dir in _discover_seed_dirs(model_dir):
            csv_path = seed_dir / "inference_attention_trace.csv"
            if csv_path.exists():
                rows = _read_csv_rows(csv_path)
                for row in rows:
                    row["_seed"] = seed_val
                    all_rows.append(row)
                continue

            trace_pt_path = seed_dir / "trace.pt"
            meta_path = seed_dir / "metadata.json"
            if trace_pt_path.exists():
                try:
                    trace_data = torch.load(trace_pt_path, map_location="cpu")
                    x_t = trace_data.get("x_t")
                    v_t = trace_data.get("v_t")
                    suffix_hidden = trace_data.get("suffix_hidden")
                    timesteps = trace_data.get("timesteps")
                    if x_t is not None and v_t is not None:
                        num_steps = x_t.shape[0]
                        for step_idx in range(num_steps):
                            row = {
                                "model_name": model_name,
                                "seed": seed_val,
                                "denoise_index": step_idx,
                                "x_t_norm": float(torch.norm(x_t[step_idx]).item()),
                                "v_t_norm": float(torch.norm(v_t[step_idx]).item()),
                                "suffix_hidden_norm": float(torch.norm(suffix_hidden[step_idx]).item()) if suffix_hidden is not None and step_idx < suffix_hidden.shape[0] else 0.0,
                                "timestep": float(timesteps[step_idx].item()) if timesteps is not None and step_idx < timesteps.shape[0] else 0.0,
                            }
                            if meta_path.exists():
                                with open(meta_path, "r") as f:
                                    meta = json.load(f)
                                row["method"] = meta.get("method", "")
                                row["eval_task_success"] = meta.get("eval_task_success", None)
                                row["eval_grasp_success"] = meta.get("eval_grasp_success", None)
                            all_rows.append(row)
                except Exception as e:
                    print(f"  WARNING: Failed to read trace.pt for {model_name}/{seed_dir.name}: {e}")

    if not all_rows:
        print("  Trying root-level inference trace CSV...")
        csv_path = INFERENCE_TRACE_DIR / "inference_attention_trace.csv"
        if csv_path.exists():
            all_rows = _read_csv_rows(csv_path)

    if not all_rows:
        print("  WARNING: No inference trace data found.")
        return []

    agg = defaultdict(lambda: {"x_t_norm": [], "v_t_norm": [], "suffix_hidden_norm": [], "seed_values": []})

    for row in all_rows:
        model_name = row.get("model_name", "")
        layer = row.get("layer", "")
        denoise_index = row.get("denoise_index", "")
        key = (model_name, _safe_int(layer), _safe_int(denoise_index))

        a = agg[key]
        a["seed_values"].append(row.get("seed", ""))

        for field in ["x_t_norm", "v_t_norm", "suffix_hidden_norm"]:
            val = _safe_float(row.get(field))
            if val is not None:
                a[field].append(val)

    output_rows = []
    for (model_name, layer, denoise_idx), a in sorted(agg.items()):
        count = len(a["seed_values"])
        mean_xt, std_xt, _ = _agg_stats(a["x_t_norm"])
        mean_vt, std_vt, _ = _agg_stats(a["v_t_norm"])
        mean_sh, std_sh, _ = _agg_stats(a["suffix_hidden_norm"])

        output_rows.append({
            "model_name": model_name,
            "layer": layer,
            "denoise_index": denoise_idx,
            "count": count,
            "x_t_norm_mean": mean_xt,
            "x_t_norm_std": std_xt,
            "v_t_norm_mean": mean_vt,
            "v_t_norm_std": std_vt,
            "suffix_hidden_norm_mean": mean_sh,
            "suffix_hidden_norm_std": std_sh,
        })

    _write_csv(OUTPUT_DIR / "inference_trace_aggregate.csv", output_rows)
    return output_rows


def aggregate_pairwise_action_divergence():
    """Read per-seed pairwise_action_divergence.csv and aggregate.

    For each seed, there are 21 model pairs (7 choose 2).
    Aggregates v_t_l2, v_t_cosine, x_t_l2, hidden_cosine by model_a/model_b/denoise_index.
    """
    print("  Scanning pairwise divergence CSVs...")

    all_rows = []
    plots_dir = INFERENCE_TRACE_DIR / "plots"

    if plots_dir.exists():
        for seed_dir in sorted(plots_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            for csv_file in sorted(seed_dir.glob("*pairwise*divergence.csv")):
                rows = _read_csv_rows(csv_file)
                for row in rows:
                    row["_seed"] = seed_dir.name
                    all_rows.append(row)

    for cfg in CORNER_12K_MODEL_CONFIGS:
        model_name = cfg["name"]
        model_dir = INFERENCE_TRACE_DIR / model_name
        if not model_dir.exists():
            continue
        for seed_val, seed_dir in _discover_seed_dirs(model_dir):
            plots_sub = seed_dir / "plots"
            if plots_sub.exists():
                for csv_file in sorted(plots_sub.glob("*pairwise*divergence.csv")):
                    rows = _read_csv_rows(csv_file)
                    for row in rows:
                        row["_seed"] = seed_dir.name
                        all_rows.append(row)

    if not all_rows:
        print("  Trying per-model pairwise CSVs in result dirs...")
        for cfg in CORNER_12K_MODEL_CONFIGS:
            model_name = cfg["name"]
            model_dir = RESULT_DIR / model_name
            if not model_dir.exists():
                continue
            for seed_val, seed_dir in _discover_seed_dirs(model_dir):
                plots_sub = seed_dir / "plots"
                if plots_sub.exists():
                    for csv_file in sorted(plots_sub.glob("*pairwise*divergence.csv")):
                        rows = _read_csv_rows(csv_file)
                        for row in rows:
                            row["_seed"] = seed_dir.name
                            all_rows.append(row)

    if not all_rows:
        print("  WARNING: No pairwise divergence data found.")
        return []

    agg = defaultdict(lambda: {
        "v_t_l2": [], "v_t_cosine": [], "x_t_l2": [], "hidden_cosine": [],
        "seed_values": [],
    })

    for row in all_rows:
        model_a = row.get("model_a", "")
        model_b = row.get("model_b", "")
        denoise_idx = _safe_int(row.get("denoise_index"))
        key = (model_a, model_b, denoise_idx)

        a = agg[key]
        a["seed_values"].append(row.get("_seed", ""))

        for field in ["v_t_l2", "v_t_cosine", "x_t_l2", "hidden_cosine"]:
            val = _safe_float(row.get(field))
            if val is not None:
                a[field].append(val)

    output_rows = []
    for (model_a, model_b, denoise_idx), a in sorted(agg.items()):
        count = len(a["seed_values"])
        mean_vt_l2, std_vt_l2, _ = _agg_stats(a["v_t_l2"])
        mean_vt_cos, std_vt_cos, _ = _agg_stats(a["v_t_cosine"])
        mean_xt_l2, std_xt_l2, _ = _agg_stats(a["x_t_l2"])
        mean_h_cos, std_h_cos, _ = _agg_stats(a["hidden_cosine"])

        output_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "denoise_index": denoise_idx,
            "count": count,
            "v_t_l2_mean": mean_vt_l2,
            "v_t_l2_std": std_vt_l2,
            "v_t_cosine_mean": mean_vt_cos,
            "v_t_cosine_std": std_vt_cos,
            "x_t_l2_mean": mean_xt_l2,
            "x_t_l2_std": std_xt_l2,
            "hidden_cosine_mean": mean_h_cos,
            "hidden_cosine_std": std_h_cos,
        })

    _write_csv(OUTPUT_DIR / "pairwise_action_divergence_aggregate.csv", output_rows)
    return output_rows


def aggregate_velocity_statistics():
    """Aggregate velocity (v_t_norm) statistics from inference trace.

    For each model, compute:
      - Per-seed mean and std of v_t_norm across all denoise steps
      - Cross-seed mean of per-seed means, std of per-seed means
      - Cross-seed mean of per-seed stds, std of per-seed stds
    """
    print("  Computing velocity statistics from inference trace...")

    all_rows = []

    for cfg in CORNER_12K_MODEL_CONFIGS:
        model_name = cfg["name"]
        model_dir = INFERENCE_TRACE_DIR / model_name
        if not model_dir.exists():
            continue

        for seed_val, seed_dir in _discover_seed_dirs(model_dir):
            csv_path = seed_dir / "inference_attention_trace.csv"
            if csv_path.exists():
                rows = _read_csv_rows(csv_path)
                for row in rows:
                    row["_seed"] = seed_val
                    all_rows.append(row)
                continue

            trace_pt_path = seed_dir / "trace.pt"
            if trace_pt_path.exists():
                try:
                    trace_data = torch.load(trace_pt_path, map_location="cpu")
                    v_t = trace_data.get("v_t")
                    timesteps = trace_data.get("timesteps")
                    if v_t is not None:
                        num_steps = v_t.shape[0]
                        for step_idx in range(num_steps):
                            all_rows.append({
                                "model_name": model_name,
                                "seed": seed_val,
                                "denoise_index": step_idx,
                                "v_t_norm": float(torch.norm(v_t[step_idx]).item()),
                                "timestep": float(timesteps[step_idx].item()) if timesteps is not None and step_idx < timesteps.shape[0] else 0.0,
                            })
                except Exception as e:
                    print(f"  WARNING: Failed to read trace.pt for {model_name}/{seed_dir.name}: {e}")

    if not all_rows:
        csv_path = INFERENCE_TRACE_DIR / "inference_attention_trace.csv"
        if csv_path.exists():
            all_rows = _read_csv_rows(csv_path)

    if not all_rows:
        for cfg in CORNER_12K_MODEL_CONFIGS:
            model_name = cfg["name"]
            model_csv = INFERENCE_TRACE_DIR / model_name / "inference_attention_trace.csv"
            if model_csv.exists():
                all_rows.extend(_read_csv_rows(model_csv))

    if not all_rows:
        print("  WARNING: No inference trace data for velocity statistics.")
        return []

    per_model_seed = defaultdict(lambda: {"v_t_norm": []})

    for row in all_rows:
        model_name = row.get("model_name", "")
        seed = row.get("seed", row.get("_seed", ""))
        key = (model_name, seed)
        val = _safe_float(row.get("v_t_norm"))
        if val is not None:
            per_model_seed[key]["v_t_norm"].append(val)

    per_model_agg = defaultdict(lambda: {"seed_means": [], "seed_stds": []})

    for (model_name, seed), data in per_model_seed.items():
        values = data["v_t_norm"]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        per_model_agg[model_name]["seed_means"].append(float(np.mean(arr)))
        per_model_agg[model_name]["seed_stds"].append(float(np.std(arr)))

    output_rows = []
    for model_name, data in sorted(per_model_agg.items()):
        means = data["seed_means"]
        stds = data["seed_stds"]
        num_seeds = len(means)

        mean_of_means, std_of_means, _ = _agg_stats(means)
        mean_of_stds, std_of_stds, _ = _agg_stats(stds)

        output_rows.append({
            "model_name": model_name,
            "num_seeds": num_seeds,
            "v_t_norm_mean_of_means": mean_of_means,
            "v_t_norm_std_of_means": std_of_means,
            "v_t_norm_mean_of_stds": mean_of_stds,
            "v_t_norm_std_of_stds": std_of_stds,
        })

    _write_csv(OUTPUT_DIR / "velocity_statistics.csv", output_rows)
    return output_rows


def write_model_performance_metadata(model_records):
    """Write model_performance_metadata.csv.

    ONLY records name, method, path, eval_task_success, eval_grasp_success.
    These metrics are for post-hoc reference only and MUST NOT be used
    for any sorting, filtering, or statistical inference in this script.
    """
    output_rows = []
    for rec in model_records:
        output_rows.append({
            "model_name": rec["name"],
            "method": rec["method"],
            "checkpoint_path": rec["path"],
            "eval_task_success": rec["eval_task_success"],
            "eval_grasp_success": rec["eval_grasp_success"],
        })

    _write_csv(OUTPUT_DIR / "model_performance_metadata.csv", output_rows)
    return output_rows


def _load_csv_for_report(filename):
    """Helper to load a CSV from OUTPUT_DIR for report generation."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        return []
    return _read_csv_rows(path)


def _fmt(val, decimals=4):
    """Format a number for markdown table."""
    if val is None or val == "":
        return "N/A"
    try:
        numeric_val = float(val)
        return f"{numeric_val:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def write_summary_report(model_records):
    """Generate analysis_12k_corner_summary.md from aggregated CSVs.

    This function ONLY describes observable statistical facts.
    It does NOT make causal claims.
    Correlation != Causation is explicitly stated.
    Single-seed artifacts and cross-seed instability are flagged.
    """
    print("  Generating summary report...")

    lookup = _build_model_lookup()
    md_path = OUTPUT_DIR / "analysis_12k_corner_summary.md"

    lines = []
    lines.append("# 12k Corner 7-Model Attention Analysis Summary")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("This report summarizes post-hoc attention and action trajectory analysis")
    lines.append("of 7 models at the 12k checkpoint using corner,gripperPOV camera configuration.")
    lines.append("Analysis is based on 10 fixed seeds: 10042,10043,10044,10045,10046,10047,10048,10049,10050,10051.")
    lines.append("")
    lines.append("### Models Analyzed")
    lines.append("")
    lines.append("| Model | Method | Task Success | Grasp Success |")
    lines.append("| --- | --- | --- | --- |")
    for rec in model_records:
        lines.append(
            f"| {rec['name']} | {rec['method']} | "
            f"{rec['eval_task_success']}% | {rec['eval_grasp_success']}% |"
        )
    lines.append("")
    lines.append("**Important**: The success metrics above are 200-episode final eval results used")
    lines.append("ONLY for post-hoc metadata (tables, sorting, correlation observation).")
    lines.append("They did NOT participate in seed selection, attention computation, model inference,")
    lines.append("parameter adjustment, or any selection decision.")
    lines.append("")

    lines.append("## 2. Attention Metrics Aggregate")
    lines.append("")
    att_rows = _load_csv_for_report("attention_aggregate.csv")
    if att_rows:
        lines.append(f"Total aggregated records: {len(att_rows)}.")
        lines.append("")

        expert_cross = [r for r in att_rows if r.get("attention_source") == "expert_cross"]
        if expert_cross:
            layer11 = [r for r in expert_cross if r.get("layer") == "11"]
            if layer11:
                lines.append("### Layer 11 Expert-Cross Attention Mass (mean +/- std across seeds)")
                lines.append("")
                lines.append("| Model | Phase | Camera1 | Camera2 | Visual Total | Language | State | Count |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for r in sorted(layer11, key=lambda x: x.get("model_name", "")):
                    lines.append(
                        f"| {r['model_name']} | {r['phase']} | "
                        f"{_fmt(r.get('camera1_mass_mean'))} +/- {_fmt(r.get('camera1_mass_std'))} | "
                        f"{_fmt(r.get('camera2_mass_mean'))} +/- {_fmt(r.get('camera2_mass_std'))} | "
                        f"{_fmt(r.get('visual_total_mean'))} +/- {_fmt(r.get('visual_total_std'))} | "
                        f"{_fmt(r.get('language_mass_mean'))} +/- {_fmt(r.get('language_mass_std'))} | "
                        f"{_fmt(r.get('state_mass_mean'))} +/- {_fmt(r.get('state_mass_std'))} | "
                        f"{r.get('count', 'N/A')} |"
                    )
                lines.append("")

            layer3 = [r for r in expert_cross if r.get("layer") == "3"]
            if layer3:
                lines.append("### Layer 3 Expert-Cross Attention Mass (mean +/- std across seeds)")
                lines.append("")
                lines.append("| Model | Phase | Camera1 | Camera2 | Visual Total | Language | State | Count |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for r in sorted(layer3, key=lambda x: x.get("model_name", "")):
                    lines.append(
                        f"| {r['model_name']} | {r['phase']} | "
                        f"{_fmt(r.get('camera1_mass_mean'))} +/- {_fmt(r.get('camera1_mass_std'))} | "
                        f"{_fmt(r.get('camera2_mass_mean'))} +/- {_fmt(r.get('camera2_mass_std'))} | "
                        f"{_fmt(r.get('visual_total_mean'))} +/- {_fmt(r.get('visual_total_std'))} | "
                        f"{_fmt(r.get('language_mass_mean'))} +/- {_fmt(r.get('language_mass_std'))} | "
                        f"{_fmt(r.get('state_mass_mean'))} +/- {_fmt(r.get('state_mass_std'))} | "
                        f"{r.get('count', 'N/A')} |"
                    )
                lines.append("")

            layer7 = [r for r in expert_cross if r.get("layer") == "7"]
            if layer7:
                lines.append("### Layer 7 Expert-Cross Attention Mass (mean +/- std across seeds)")
                lines.append("")
                lines.append("| Model | Phase | Camera1 | Camera2 | Visual Total | Language | State | Count |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for r in sorted(layer7, key=lambda x: x.get("model_name", "")):
                    lines.append(
                        f"| {r['model_name']} | {r['phase']} | "
                        f"{_fmt(r.get('camera1_mass_mean'))} +/- {_fmt(r.get('camera1_mass_std'))} | "
                        f"{_fmt(r.get('camera2_mass_mean'))} +/- {_fmt(r.get('camera2_mass_std'))} | "
                        f"{_fmt(r.get('visual_total_mean'))} +/- {_fmt(r.get('visual_total_std'))} | "
                        f"{_fmt(r.get('language_mass_mean'))} +/- {_fmt(r.get('language_mass_std'))} | "
                        f"{_fmt(r.get('state_mass_mean'))} +/- {_fmt(r.get('state_mass_std'))} | "
                        f"{r.get('count', 'N/A')} |"
                    )
                lines.append("")
    else:
        lines.append("No attention aggregate data available.")
        lines.append("")

    lines.append("## 3. Phase Reach Statistics (Rollout Behavioral Diagnostics)")
    lines.append("")
    phase_rows = _load_csv_for_report("phase_reach_aggregate.csv")
    if phase_rows:
        lines.append(f"Total phase records: {len(phase_rows)}.")
        lines.append("")
        lines.append("| Model | Phase | Reached | Total | Ratio |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in sorted(phase_rows, key=lambda x: (x.get("model_name", ""), x.get("phase", ""))):
            lines.append(
                f"| {r['model_name']} | {r['phase']} | "
                f"{r.get('reached_count', 0)} | {r.get('total_count', 0)} | "
                f"{_fmt(r.get('reached_ratio'), 3)} |"
            )
        lines.append("")
        lines.append("**Note**: These are behavioral diagnostics only. Different models in post-grasp and")
        lines.append("pre-place phases use their own policy trajectories, so these are NOT strict")
        lines.append("same-observation causal comparisons. Strict same-observation comparison is based")
        lines.append("on inference_trace initial input.")
        lines.append("")
    else:
        lines.append("No phase reach data available.")
        lines.append("")

    lines.append("## 4. Heatmap Spatial Statistics")
    lines.append("")
    hm_rows = _load_csv_for_report("heatmap_statistics.csv")
    if hm_rows:
        lines.append(f"Total heatmap statistics records: {len(hm_rows)}.")
        lines.append("")
        lines.append("Key metrics: entropy (spatial concentration), center_x/center_y (attention centroid),")
        lines.append("max_attention (peak attention value). Lower entropy = more concentrated attention.")
        lines.append("")

        expert_cross_hm = [r for r in hm_rows if r.get("attention_source") == "expert_cross"]
        if expert_cross_hm:
            layer11_hm = [r for r in expert_cross_hm if r.get("layer") == "11"]
            if layer11_hm:
                lines.append("### Layer 11 Expert-Cross Heatmap Statistics")
                lines.append("")
                lines.append("| Model | Phase | Camera | Entropy | Max Attn | Center X | Center Y | Count |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for r in sorted(layer11_hm, key=lambda x: (x.get("model_name", ""), x.get("camera", ""))):
                    lines.append(
                        f"| {r['model_name']} | {r['phase']} | {r['camera']} | "
                        f"{_fmt(r.get('entropy_mean'))} +/- {_fmt(r.get('entropy_std'))} | "
                        f"{_fmt(r.get('max_attention_mean'))} +/- {_fmt(r.get('max_attention_std'))} | "
                        f"{_fmt(r.get('center_x_mean'), 2)} +/- {_fmt(r.get('center_x_std'), 2)} | "
                        f"{_fmt(r.get('center_y_mean'), 2)} +/- {_fmt(r.get('center_y_std'), 2)} | "
                        f"{r.get('count', 'N/A')} |"
                    )
                lines.append("")
    else:
        lines.append("No heatmap statistics data available.")
        lines.append("")

    lines.append("## 5. Inference Trace Aggregate")
    lines.append("")
    trace_rows = _load_csv_for_report("inference_trace_aggregate.csv")
    if trace_rows:
        lines.append(f"Total inference trace records: {len(trace_rows)}.")
        lines.append("")
        lines.append("| Model | Layer | Denoise Step | x_t_norm | v_t_norm | suffix_hidden_norm | Count |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in sorted(trace_rows, key=lambda x: (x.get("model_name", ""), x.get("layer", ""), x.get("denoise_index", ""))):
            lines.append(
                f"| {r['model_name']} | {r.get('layer', 'N/A')} | {r.get('denoise_index', 'N/A')} | "
                f"{_fmt(r.get('x_t_norm_mean'))} +/- {_fmt(r.get('x_t_norm_std'))} | "
                f"{_fmt(r.get('v_t_norm_mean'))} +/- {_fmt(r.get('v_t_norm_std'))} | "
                f"{_fmt(r.get('suffix_hidden_norm_mean'))} +/- {_fmt(r.get('suffix_hidden_norm_std'))} | "
                f"{r.get('count', 'N/A')} |"
            )
        lines.append("")
    else:
        lines.append("No inference trace data available.")
        lines.append("")

    lines.append("## 6. Pairwise Action Divergence")
    lines.append("")
    pw_rows = _load_csv_for_report("pairwise_action_divergence_aggregate.csv")
    if pw_rows:
        lines.append(f"Total pairwise records: {len(pw_rows)}.")
        lines.append("")
        lines.append("| Model A | Model B | Denoise | v_t_l2 | v_t_cosine | x_t_l2 | hidden_cosine | Count |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in sorted(pw_rows, key=lambda x: (x.get("model_a", ""), x.get("model_b", ""), x.get("denoise_index", ""))):
            lines.append(
                f"| {r['model_a']} | {r['model_b']} | {r.get('denoise_index', 'N/A')} | "
                f"{_fmt(r.get('v_t_l2_mean'))} +/- {_fmt(r.get('v_t_l2_std'))} | "
                f"{_fmt(r.get('v_t_cosine_mean'))} +/- {_fmt(r.get('v_t_cosine_std'))} | "
                f"{_fmt(r.get('x_t_l2_mean'))} +/- {_fmt(r.get('x_t_l2_std'))} | "
                f"{_fmt(r.get('hidden_cosine_mean'))} +/- {_fmt(r.get('hidden_cosine_std'))} | "
                f"{r.get('count', 'N/A')} |"
            )
        lines.append("")
    else:
        lines.append("No pairwise divergence data available.")
        lines.append("")

    lines.append("## 7. Velocity Statistics")
    lines.append("")
    vel_rows = _load_csv_for_report("velocity_statistics.csv")
    if vel_rows:
        lines.append(f"Total velocity records: {len(vel_rows)}.")
        lines.append("")
        lines.append("| Model | Seeds | v_t_norm Mean-of-Means | Std-of-Means | Mean-of-Stds | Std-of-Stds |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in sorted(vel_rows, key=lambda x: x.get("model_name", "")):
            lines.append(
                f"| {r['model_name']} | {r.get('num_seeds', 'N/A')} | "
                f"{_fmt(r.get('v_t_norm_mean_of_means'))} | "
                f"{_fmt(r.get('v_t_norm_std_of_means'))} | "
                f"{_fmt(r.get('v_t_norm_mean_of_stds'))} | "
                f"{_fmt(r.get('v_t_norm_std_of_stds'))} |"
            )
        lines.append("")
    else:
        lines.append("No velocity statistics data available.")
        lines.append("")

    lines.append("## 8. Key Research Questions (Descriptive Observations Only)")
    lines.append("")
    lines.append("**DISCLAIMER**: All observations below describe statistical correlations only.")
    lines.append("Correlation does NOT imply causation. If different seeds show inconsistent results,")
    lines.append("this is marked as unstable. If only a single seed shows a difference, it is marked")
    lines.append("as a single-seed artifact.")
    lines.append("")

    lines.append("### Q1: ours_v1 -> v2 -> v3 -> v4 attention pattern evolution")
    lines.append("")
    lines.append(f"Task success progression: 20.0% -> 25.5% -> 32.0% -> 34.5%.")
    lines.append("")
    lines.append("Observation framework:")
    lines.append("- Check attention_aggregate.csv for camera1_mass, camera2_mass, visual_total trends")
    lines.append("  across layers 3/7/11 in the initial phase (strict same-observation).")
    lines.append("- Check heatmap_statistics.csv for entropy, center_x, center_y stability across seeds.")
    lines.append("- Check inference_trace_aggregate.csv for x_t_norm, v_t_norm, suffix_hidden_norm trends.")
    lines.append("")
    lines.append("If attention mass or spatial concentration shows consistent directional change")
    lines.append("across all 10 seeds, this indicates a stable pattern. If only some seeds agree,")
    lines.append("mark as partially stable. If seeds disagree, mark as unstable.")
    lines.append("")

    lines.append("### Q2: ours_v3 vs random (both 32.0 task success)")
    lines.append("")
    lines.append("Both models achieve 32.0% task success but use different selection methods.")
    lines.append("")
    lines.append("Observation framework:")
    lines.append("- Compare attention_aggregate.csv entries for ours_v3_corner_12k vs random_corner_12k")
    lines.append("  at same layer/phase/attention_source.")
    lines.append("- Compare pairwise_action_divergence_aggregate.csv for the (ours_v3, random) pair.")
    lines.append("  High v_t_l2 or low v_t_cosine indicates different velocity trajectories despite")
    lines.append("  same success rate.")
    lines.append("- Compare heatmap_statistics.csv for spatial attention differences.")
    lines.append("")
    lines.append("If v_t_l2 between v3 and random is consistently high across seeds, this suggests")
    lines.append("different internal dynamics despite same external success. This is a correlation,")
    lines.append("not a causal claim about which method is better.")
    lines.append("")

    lines.append("### Q3: ours_v4 vs random (34.5% vs 32.0%)")
    lines.append("")
    lines.append("v4 achieves 2.5 percentage points higher task success than random.")
    lines.append("")
    lines.append("Observation framework:")
    lines.append("- Check phase_reach_aggregate.csv for post_grasp and pre_place reach ratio differences.")
    lines.append("- Check attention_aggregate.csv for post_grasp and pre_place attention differences")
    lines.append("  at layers 3/7/11.")
    lines.append("- Check pairwise_action_divergence_aggregate.csv for (ours_v4, random) pair metrics.")
    lines.append("")
    lines.append("If v4 shows consistently different attention in post-grasp or pre-place phases")
    lines.append("compared to random, this is a behavioral correlation. Note that rollout phase")
    lines.append("comparisons are NOT strict same-observation (each model generates its own trajectory).")
    lines.append("Only inference_trace initial phase provides strict same-observation comparison.")
    lines.append("")

    lines.append("### Q4: v4 action-related design consistency vs v3")
    lines.append("")
    lines.append("v4 adds action-related selection design compared to v3.")
    lines.append("")
    lines.append("Observation framework:")
    lines.append("- Check inference_trace_aggregate.csv: compare v_t_norm and suffix_hidden_norm")
    lines.append("  between v4 and v3 across denoise steps.")
    lines.append("- Check pairwise_action_divergence_aggregate.csv for (ours_v4, ours_v3) pair.")
    lines.append("- Check velocity_statistics.csv for v_t_norm differences.")
    lines.append("")
    lines.append("If v4 shows consistently different v_t_norm or hidden representation compared to v3")
    lines.append("across seeds and denoise steps, this indicates the action-related design changes")
    lines.append("internal dynamics. This is correlation, not causation.")
    lines.append("")

    lines.append("### Q5: Baseline trend support (uniform, zero, random)")
    lines.append("")
    lines.append("uniform (28.2%), zero (32.0%), random (32.0%) serve as baselines.")
    lines.append("")
    lines.append("Observation framework:")
    lines.append("- Check if baseline attention patterns follow similar or different trends compared")
    lines.append("  to the ours_v* progression.")
    lines.append("- Check pairwise divergence between baselines and ours models.")
    lines.append("")
    lines.append("If baselines show attention patterns that differ from the ours_v* trend,")
    lines.append("this may suggest the selection method matters. If baselines follow similar patterns,")
    lines.append("the trend may be method-agnostic. All observations are correlational.")
    lines.append("")

    lines.append("## 9. Stability Assessment")
    lines.append("")

    if hm_rows:
        lines.append("### Cross-seed Stability of Heatmap Statistics")
        lines.append("")
        unstable_records = []
        for r in hm_rows:
            std_val = _safe_float(r.get("entropy_std"))
            mean_val = _safe_float(r.get("entropy_mean"))
            if std_val is not None and mean_val is not None and mean_val > 0:
                cv = std_val / mean_val
                if cv > 0.3:
                    unstable_records.append(r)

        if unstable_records:
            lines.append(f"Found {len(unstable_records)} records with coefficient of variation > 0.3 (unstable across seeds):")
            lines.append("")
            for r in unstable_records[:10]:
                lines.append(
                    f"- {r['model_name']} / {r['phase']} / layer {r.get('layer', '?')} / "
                    f"{r['camera']}: entropy mean={_fmt(r.get('entropy_mean'))}, "
                    f"std={_fmt(r.get('entropy_std'))}"
                )
            lines.append("")
        else:
            lines.append("All heatmap statistics show low cross-seed variation (CV < 0.3).")
            lines.append("")

    if att_rows:
        lines.append("### Cross-seed Stability of Attention Metrics")
        lines.append("")
        unstable_att = []
        for r in att_rows:
            std_val = _safe_float(r.get("camera1_mass_std"))
            mean_val = _safe_float(r.get("camera1_mass_mean"))
            if std_val is not None and mean_val is not None and abs(mean_val) > 0.01:
                cv = std_val / abs(mean_val)
                if cv > 0.3:
                    unstable_att.append(r)

        if unstable_att:
            lines.append(f"Found {len(unstable_att)} attention records with CV > 0.3 (unstable across seeds):")
            lines.append("")
            for r in unstable_att[:10]:
                lines.append(
                    f"- {r['model_name']} / {r['phase']} / layer {r.get('layer', '?')} / "
                    f"{r['attention_source']}: camera1_mass mean={_fmt(r.get('camera1_mass_mean'))}, "
                    f"std={_fmt(r.get('camera1_mass_std'))}"
                )
            lines.append("")
        else:
            lines.append("All attention metrics show low cross-seed variation (CV < 0.3).")
            lines.append("")

    lines.append("---")
    lines.append("*Report generated by aggregate_12k_corner.py*")
    lines.append("*All observations are correlational. No causal claims are made.*")
    lines.append("*Single-seed artifacts and cross-seed instability are flagged where applicable.*")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {md_path}")


def main():
    print("=" * 80)
    print("12k Corner 7-Model Attention Analysis Aggregation")
    print("=" * 80)
    print()

    ensure_output_dir()

    print("\n[1/9] Loading model metadata from model_configs_12k_corner.py...")
    model_records = load_model_metadata()

    print("\n[2/9] Aggregating attention metrics...")
    aggregate_attention_metrics()

    print("\n[3/9] Aggregating phase reach statistics...")
    aggregate_phase_reach()

    print("\n[4/9] Aggregating heatmap statistics...")
    aggregate_heatmap_statistics()

    print("\n[5/9] Aggregating inference trace...")
    aggregate_inference_trace()

    print("\n[6/9] Aggregating pairwise action divergence...")
    aggregate_pairwise_action_divergence()

    print("\n[7/9] Aggregating velocity statistics...")
    aggregate_velocity_statistics()

    print("\n[8/9] Writing model performance metadata...")
    write_model_performance_metadata(model_records)

    print("\n[9/9] Generating summary report...")
    write_summary_report(model_records)

    print("\n" + "=" * 80)
    print("DONE. All outputs saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()