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

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULT_DIR = PROJECT_ROOT / "personal/work2/attention_fig/result_12k_corner"
INFERENCE_TRACE_DIR = PROJECT_ROOT / "personal/work2/attention_fig/result_inference_trace_12k_corner"
OUTPUT_DIR = PROJECT_ROOT / "personal/work2/attention_fig/analysis_12k_corner"

MODEL_NAMES = [
    "ours_v1_corner_12k",
    "ours_v2_corner_12k",
    "ours_v3_corner_12k",
    "ours_v4_corner_12k",
    "random_corner_12k",
    "uniform_corner_12k",
    "zero_corner_12k",
]

METHOD_MAP = {
    "ours_v1_corner_12k": "ours_v1",
    "ours_v2_corner_12k": "ours_v2",
    "ours_v3_corner_12k": "ours_v3",
    "ours_v4_corner_12k": "ours_v4",
    "random_corner_12k": "random",
    "uniform_corner_12k": "uniform",
    "zero_corner_12k": "zero",
}

PATH_MAP = {
    "ours_v1_corner_12k": "personal/work2/duibi/ours_112_seed42_corner/dynamicanchor_112_seed42/checkpoints/012000/pretrained_model",
    "ours_v2_corner_12k": "personal/work2/duibi/ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42/checkpoints/012000/pretrained_model",
    "ours_v3_corner_12k": "personal/work2/duibi/ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42/checkpoints/012000/pretrained_model",
    "ours_v4_corner_12k": "personal/work2/duibi/ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42/checkpoints/012000/pretrained_model",
    "random_corner_12k": "personal/work2/duibi/random_42_corner/random_112_seed42/checkpoints/012000/pretrained_model",
    "uniform_corner_12k": "personal/work2/duibi/uniform_42_corner/uniform_112_seed42/checkpoints/012000/pretrained_model",
    "zero_corner_12k": "personal/work2/duibi/subzerocore_112_seed42_corner/subzerocore_112_seed42/checkpoints/012000/pretrained_model",
}

PERFORMANCE_MAP = {
    "ours_v1_corner_12k": {"eval_task_success": 20.0, "eval_grasp_success": 90.0},
    "ours_v2_corner_12k": {"eval_task_success": 25.5, "eval_grasp_success": 95.5},
    "ours_v3_corner_12k": {"eval_task_success": 32.0, "eval_grasp_success": 94.5},
    "ours_v4_corner_12k": {"eval_task_success": 34.5, "eval_grasp_success": 95.0},
    "random_corner_12k": {"eval_task_success": 32.0, "eval_grasp_success": 94.5},
    "uniform_corner_12k": {"eval_task_success": 28.2, "eval_grasp_success": 92.5},
    "zero_corner_12k": {"eval_task_success": 32.0, "eval_grasp_success": 95.0},
}


def load_model_metadata():
    """Load metadata from each model/seed directory."""
    metadata_records = []
    for model_name in MODEL_NAMES:
        method = METHOD_MAP[model_name]
        model_dir = RESULT_DIR / model_name
        if not model_dir.exists():
            print(f"  WARNING: Model directory not found: {model_dir}")
            continue
        for seed_dir in sorted(model_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_")[1])
            meta_path = seed_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                meta["_model_name"] = model_name
                meta["_method"] = method
                meta["_seed"] = seed
                metadata_records.append(meta)
    print(f"  Loaded {len(metadata_records)} metadata records")
    return metadata_records


def aggregate_attention_metrics(metadata_records):
    """Aggregate attention metrics across seeds by model/phase/layer/source."""
    csv_path = RESULT_DIR / "attention_summary.csv"
    if not csv_path.exists():
        print(f"  WARNING: attention_summary.csv not found at {csv_path}")
        return []

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    aggregated = {}
    for row in rows:
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
                "layer": int(layer) if layer else None,
                "attention_source": attention_source,
                "camera1_mass_values": [],
                "camera2_mass_values": [],
                "visual_total_values": [],
                "language_mass_values": [],
                "state_mass_values": [],
            }

        for field, values_key in [
            ("camera1_mass", "camera1_mass_values"),
            ("camera2_mass", "camera2_mass_values"),
            ("visual_total", "visual_total_values"),
            ("language_mass", "language_mass_values"),
            ("state_mass", "state_mass_values"),
        ]:
            val = row.get(field)
            if val is not None and val != "":
                try:
                    aggregated[key][values_key].append(float(val))
                except ValueError:
                    pass

    output_rows = []
    for key, agg in aggregated.items():
        record = {
            "model_name": agg["model_name"],
            "method": agg["method"],
            "phase": agg["phase"],
            "layer": agg["layer"],
            "attention_source": agg["attention_source"],
            "count": len(agg["camera1_mass_values"]),
        }
        for metric, values_key in [
            ("camera1_mass", "camera1_mass_values"),
            ("camera2_mass", "camera2_mass_values"),
            ("visual_total", "visual_total_values"),
            ("language_mass", "language_mass_values"),
            ("state_mass", "state_mass_values"),
        ]:
            values = agg[values_key]
            if values:
                record[f"{metric}_mean"] = np.mean(values)
                record[f"{metric}_std"] = np.std(values)
            else:
                record[f"{metric}_mean"] = None
                record[f"{metric}_std"] = None
        output_rows.append(record)

    csv_out = OUTPUT_DIR / "attention_aggregate.csv"
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"  Saved {csv_out} with {len(output_rows)} rows")

    return output_rows


def aggregate_phase_reach(metadata_records):
    """Aggregate phase reach statistics from rollout metadata."""
    output_rows = []
    for meta in metadata_records:
        model_name = meta.get("_model_name", "")
        method = meta.get("_method", "")
        seed = meta.get("_seed", "")
        phases = meta.get("phases", {})
        for phase_name in ["initial", "pre_grasp", "post_grasp", "pre_place"]:
            phase_data = phases.get(phase_name, {})
            reached = phase_data.get("reached", False)
            output_rows.append({
                "model_name": model_name,
                "method": method,
                "seed": seed,
                "phase": phase_name,
                "reached": reached,
            })

    aggregated = {}
    for row in output_rows:
        key = (row["model_name"], row["method"], row["phase"])
        if key not in aggregated:
            aggregated[key] = {
                "model_name": row["model_name"],
                "method": row["method"],
                "phase": row["phase"],
                "reached_count": 0,
                "total_count": 0,
            }
        aggregated[key]["total_count"] += 1
        if row["reached"]:
            aggregated[key]["reached_count"] += 1

    final_rows = []
    for key, agg in aggregated.items():
        final_rows.append({
            "model_name": agg["model_name"],
            "method": agg["method"],
            "phase": agg["phase"],
            "reached_count": agg["reached_count"],
            "total_count": agg["total_count"],
            "reached_ratio": agg["reached_count"] / agg["total_count"] if agg["total_count"] > 0 else 0.0,
        })

    csv_out = OUTPUT_DIR / "phase_reach_aggregate.csv"
    if final_rows:
        fieldnames = list(final_rows[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)
        print(f"  Saved {csv_out} with {len(final_rows)} rows")

    return final_rows


def aggregate_heatmap_statistics():
    """Read npz heatmap files and compute spatial statistics."""
    output_rows = []

    for model_name in MODEL_NAMES:
        method = METHOD_MAP[model_name]
        heatmaps_dir = INFERENCE_TRACE_DIR / model_name
        if not heatmaps_dir.exists():
            heatmaps_dir = RESULT_DIR / model_name
        if not heatmaps_dir.exists():
            continue

        for seed_dir in sorted(heatmaps_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_")[1])

            hm_root = seed_dir / "heatmaps"
            if not hm_root.exists():
                continue

            for npz_file in sorted(hm_root.rglob("*.npz")):
                try:
                    data = np.load(str(npz_file), allow_pickle=True)
                    attention_map = data.get("attention_map")
                    if attention_map is None:
                        continue

                    model_name_npz = str(data.get("model_name", model_name))
                    method_npz = str(data.get("method", method))
                    seed_npz = int(data.get("seed", seed))
                    phase_npz = str(data.get("phase", "unknown"))
                    layer_npz = int(data.get("layer", -1))
                    attention_source_npz = str(data.get("attention_source", "unknown"))
                    camera_npz = str(data.get("camera", "unknown"))

                    max_attn = float(np.max(attention_map))
                    total_attn = float(np.sum(attention_map))

                    if attention_map.ndim == 2:
                        h, w = attention_map.shape
                        y_coords, x_coords = np.mgrid[0:h, 0:w]
                        norm_attn = attention_map / (total_attn + 1e-10)
                        center_x = float(np.sum(x_coords * norm_attn))
                        center_y = float(np.sum(y_coords * norm_attn))

                        entropy = 0.0
                        flat_norm = norm_attn.flatten()
                        flat_norm = flat_norm[flat_norm > 0]
                        if len(flat_norm) > 0:
                            entropy = float(-np.sum(flat_norm * np.log(flat_norm + 1e-10)))
                    else:
                        center_x = None
                        center_y = None
                        entropy = None

                    output_rows.append({
                        "model_name": model_name_npz,
                        "method": method_npz,
                        "seed": seed_npz,
                        "phase": phase_npz,
                        "layer": layer_npz,
                        "attention_source": attention_source_npz,
                        "camera": camera_npz,
                        "max_attention": max_attn,
                        "attention_center_x": center_x,
                        "attention_center_y": center_y,
                        "attention_entropy": entropy,
                    })
                except Exception as e:
                    print(f"  WARNING: Failed to read {npz_file}: {e}")

    aggregated = {}
    for row in output_rows:
        key = (row["model_name"], row["method"], row["phase"], row["layer"],
               row["attention_source"], row["camera"])
        if key not in aggregated:
            aggregated[key] = {
                "model_name": row["model_name"],
                "method": row["method"],
                "phase": row["phase"],
                "layer": row["layer"],
                "attention_source": row["attention_source"],
                "camera": row["camera"],
                "max_attention_values": [],
                "center_x_values": [],
                "center_y_values": [],
                "entropy_values": [],
                "count": 0,
            }
        aggregated[key]["count"] += 1
        aggregated[key]["max_attention_values"].append(row["max_attention"])
        if row["attention_center_x"] is not None:
            aggregated[key]["center_x_values"].append(row["attention_center_x"])
        if row["attention_center_y"] is not None:
            aggregated[key]["center_y_values"].append(row["attention_center_y"])
        if row["attention_entropy"] is not None:
            aggregated[key]["entropy_values"].append(row["attention_entropy"])

    final_rows = []
    for key, agg in aggregated.items():
        record = {
            "model_name": agg["model_name"],
            "method": agg["method"],
            "phase": agg["phase"],
            "layer": agg["layer"],
            "attention_source": agg["attention_source"],
            "camera": agg["camera"],
            "count": agg["count"],
            "max_attention_mean": np.mean(agg["max_attention_values"]) if agg["max_attention_values"] else None,
            "max_attention_std": np.std(agg["max_attention_values"]) if agg["max_attention_values"] else None,
        }
        if agg["center_x_values"]:
            record["attention_center_x_mean"] = np.mean(agg["center_x_values"])
            record["attention_center_x_std"] = np.std(agg["center_x_values"])
        else:
            record["attention_center_x_mean"] = None
            record["attention_center_x_std"] = None
        if agg["center_y_values"]:
            record["attention_center_y_mean"] = np.mean(agg["center_y_values"])
            record["attention_center_y_std"] = np.std(agg["center_y_values"])
        else:
            record["attention_center_y_mean"] = None
            record["attention_center_y_std"] = None
        if agg["entropy_values"]:
            record["attention_entropy_mean"] = np.mean(agg["entropy_values"])
            record["attention_entropy_std"] = np.std(agg["entropy_values"])
        else:
            record["attention_entropy_mean"] = None
            record["attention_entropy_std"] = None
        final_rows.append(record)

    csv_out = OUTPUT_DIR / "heatmap_statistics.csv"
    if final_rows:
        fieldnames = list(final_rows[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)
        print(f"  Saved {csv_out} with {len(final_rows)} rows")

    return final_rows


def aggregate_inference_trace():
    """Aggregate inference trace metrics from CSV."""
    csv_path = INFERENCE_TRACE_DIR / "inference_attention_trace.csv"
    if not csv_path.exists():
        print(f"  WARNING: inference_attention_trace.csv not found at {csv_path}")
        return []

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    aggregated = {}
    for row in rows:
        model_name = row.get("model_name", "")
        layer = row.get("layer", "")
        denoise_index = row.get("denoise_index", "")
        key = (model_name, layer, denoise_index)

        if key not in aggregated:
            aggregated[key] = {
                "model_name": model_name,
                "layer": int(layer) if layer else None,
                "denoise_index": int(denoise_index) if denoise_index else None,
                "x_t_norm_values": [],
                "v_t_norm_values": [],
                "suffix_hidden_norm_values": [],
                "count": 0,
            }

        aggregated[key]["count"] += 1
        for field, values_key in [
            ("x_t_norm", "x_t_norm_values"),
            ("v_t_norm", "v_t_norm_values"),
            ("suffix_hidden_norm", "suffix_hidden_norm_values"),
        ]:
            val = row.get(field)
            if val is not None and val != "":
                try:
                    aggregated[key][values_key].append(float(val))
                except ValueError:
                    pass

    output_rows = []
    for key, agg in aggregated.items():
        record = {
            "model_name": agg["model_name"],
            "layer": agg["layer"],
            "denoise_index": agg["denoise_index"],
            "count": agg["count"],
        }
        for metric, values_key in [
            ("x_t_norm", "x_t_norm_values"),
            ("v_t_norm", "v_t_norm_values"),
            ("suffix_hidden_norm", "suffix_hidden_norm_values"),
        ]:
            values = agg[values_key]
            if values:
                record[f"{metric}_mean"] = np.mean(values)
                record[f"{metric}_std"] = np.std(values)
            else:
                record[f"{metric}_mean"] = None
                record[f"{metric}_std"] = None
        output_rows.append(record)

    csv_out = OUTPUT_DIR / "inference_trace_aggregate.csv"
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"  Saved {csv_out} with {len(output_rows)} rows")

    return output_rows


def aggregate_pairwise_divergence():
    """Read per-seed pairwise CSVs and aggregate."""
    all_pairwise_rows = []
    plots_dir = INFERENCE_TRACE_DIR / "plots"
    if not plots_dir.exists():
        print(f"  WARNING: plots directory not found at {plots_dir}")
        return []

    for seed_dir in sorted(plots_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed = int(seed_dir.name.split("_")[1])

        for csv_file in sorted(seed_dir.glob("*pairwise_action_divergence.csv")):
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["_seed"] = seed
                    all_pairwise_rows.append(row)

    aggregated = {}
    for row in all_pairwise_rows:
        model_a = row.get("model_a", "")
        model_b = row.get("model_b", "")
        denoise_index = row.get("denoise_index", "")
        key = (model_a, model_b, denoise_index)

        if key not in aggregated:
            aggregated[key] = {
                "model_a": model_a,
                "model_b": model_b,
                "denoise_index": int(denoise_index) if denoise_index else None,
                "v_t_l2_values": [],
                "v_t_cosine_values": [],
                "x_t_l2_values": [],
                "hidden_cosine_values": [],
                "count": 0,
            }

        aggregated[key]["count"] += 1
        for field, values_key in [
            ("v_t_l2", "v_t_l2_values"),
            ("v_t_cosine", "v_t_cosine_values"),
            ("x_t_l2", "x_t_l2_values"),
            ("hidden_cosine", "hidden_cosine_values"),
        ]:
            val = row.get(field)
            if val is not None and val != "":
                try:
                    aggregated[key][values_key].append(float(val))
                except ValueError:
                    pass

    output_rows = []
    for key, agg in aggregated.items():
        record = {
            "model_a": agg["model_a"],
            "model_b": agg["model_b"],
            "denoise_index": agg["denoise_index"],
            "count": agg["count"],
        }
        for metric, values_key in [
            ("v_t_l2", "v_t_l2_values"),
            ("v_t_cosine", "v_t_cosine_values"),
            ("x_t_l2", "x_t_l2_values"),
            ("hidden_cosine", "hidden_cosine_values"),
        ]:
            values = agg[values_key]
            if values:
                record[f"{metric}_mean"] = np.mean(values)
                record[f"{metric}_std"] = np.std(values)
            else:
                record[f"{metric}_mean"] = None
                record[f"{metric}_std"] = None
        output_rows.append(record)

    csv_out = OUTPUT_DIR / "pairwise_action_divergence_aggregate.csv"
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"  Saved {csv_out} with {len(output_rows)} rows")

    final_step_rows = []
    for row in output_rows:
        if row.get("denoise_index") is not None:
            pass
    max_denoise = max((r["denoise_index"] for r in output_rows if r["denoise_index"] is not None), default=None)
    if max_denoise is not None:
        for row in output_rows:
            if row["denoise_index"] == max_denoise:
                final_step_rows.append(row)

    return output_rows


def aggregate_velocity_statistics():
    """Aggregate velocity statistics from inference trace."""
    csv_path = INFERENCE_TRACE_DIR / "inference_attention_trace.csv"
    if not csv_path.exists():
        return []

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    per_model_seed = {}
    for row in rows:
        model_name = row.get("model_name", "")
        seed = row.get("seed", "")
        key = (model_name, seed)
        if key not in per_model_seed:
            per_model_seed[key] = {
                "model_name": model_name,
                "seed": seed,
                "v_t_norm_values": [],
                "denoise_indices": [],
            }
        val = row.get("v_t_norm")
        if val is not None and val != "":
            try:
                per_model_seed[key]["v_t_norm_values"].append(float(val))
                per_model_seed[key]["denoise_indices"].append(int(row.get("denoise_index", 0)))
            except ValueError:
                pass

    per_model_stats = {}
    for key, data in per_model_seed.items():
        model_name, seed = key
        values = data["v_t_norm_values"]
        if not values:
            continue
        if model_name not in per_model_stats:
            per_model_stats[model_name] = {
                "model_name": model_name,
                "per_seed_means": [],
                "per_seed_stds": [],
            }
        per_model_stats[model_name]["per_seed_means"].append(np.mean(values))
        per_model_stats[model_name]["per_seed_stds"].append(np.std(values))

    output_rows = []
    for model_name, stats in per_model_stats.items():
        means = stats["per_seed_means"]
        stds = stats["per_seed_stds"]
        output_rows.append({
            "model_name": model_name,
            "num_seeds": len(means),
            "v_t_norm_mean_of_means": np.mean(means) if means else None,
            "v_t_norm_std_of_means": np.std(means) if means else None,
            "v_t_norm_mean_of_stds": np.mean(stds) if stds else None,
            "v_t_norm_std_of_stds": np.std(stds) if stds else None,
        })

    csv_out = OUTPUT_DIR / "velocity_statistics.csv"
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"  Saved {csv_out} with {len(output_rows)} rows")

    return output_rows


def write_model_performance_metadata():
    """Write model performance metadata CSV."""
    output_rows = []
    for model_name in MODEL_NAMES:
        perf = PERFORMANCE_MAP[model_name]
        output_rows.append({
            "model_name": model_name,
            "method": METHOD_MAP[model_name],
            "checkpoint_path": PATH_MAP[model_name],
            "eval_task_success": perf["eval_task_success"],
            "eval_grasp_success": perf["eval_grasp_success"],
        })

    csv_out = OUTPUT_DIR / "model_performance_metadata.csv"
    fieldnames = list(output_rows[0].keys())
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"  Saved {csv_out}")

    return output_rows


def write_summary_report():
    """Generate analysis_12k_corner_summary.md from aggregated CSVs."""
    md_path = OUTPUT_DIR / "analysis_12k_corner_summary.md"

    lines = []
    lines.append("# 12k Corner 7-Model Attention Analysis Summary")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("This report summarizes post-hoc attention and action trajectory analysis")
    lines.append("of 7 models at the 12k checkpoint using corner,gripperPOV camera configuration.")
    lines.append("")
    lines.append("### Models Analyzed")
    lines.append("")
    lines.append("| Model | Method | Task Success | Grasp Success |")
    lines.append("| --- | --- | --- | --- |")
    for model_name in MODEL_NAMES:
        perf = PERFORMANCE_MAP[model_name]
        method = METHOD_MAP[model_name]
        lines.append(f"| {model_name} | {method} | {perf['eval_task_success']}% | {perf['eval_grasp_success']}% |")
    lines.append("")
    lines.append("**Important**: The success metrics above are 200-episode final eval results used")
    lines.append("ONLY for post-hoc metadata (tables, sorting, correlation observation).")
    lines.append("They did NOT participate in seed selection, attention computation, model inference,")
    lines.append("parameter adjustment, or any selection decision.")
    lines.append("")

    attention_csv = OUTPUT_DIR / "attention_aggregate.csv"
    if attention_csv.exists():
        lines.append("## 2. Attention Metrics Aggregate")
        lines.append("")
        lines.append(f"See `attention_aggregate.csv` for full data.")
        lines.append("")

        rows = []
        with open(attention_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        expert_cross_rows = [r for r in rows if r.get("attention_source") == "expert_cross"]
        if expert_cross_rows:
            layer11_rows = [r for r in expert_cross_rows if r.get("layer") == "11"]
            if layer11_rows:
                lines.append("### Layer 11 Expert-Cross Attention Mass")
                lines.append("")
                lines.append("| Model | Phase | Camera1 Mean | Camera2 Mean | Visual Total Mean |")
                lines.append("| --- | --- | --- | --- | --- |")
                for r in sorted(layer11_rows, key=lambda x: x.get("model_name", "")):
                    lines.append(
                        f"| {r['model_name']} | {r['phase']} | "
                        f"{r.get('camera1_mass_mean', 'N/A')} | "
                        f"{r.get('camera2_mass_mean', 'N/A')} | "
                        f"{r.get('visual_total_mean', 'N/A')} |"
                    )
                lines.append("")

    heatmap_csv = OUTPUT_DIR / "heatmap_statistics.csv"
    if heatmap_csv.exists():
        lines.append("## 3. Heatmap Spatial Statistics")
        lines.append("")
        lines.append(f"See `heatmap_statistics.csv` for full data.")
        lines.append("")

    inference_csv = OUTPUT_DIR / "inference_trace_aggregate.csv"
    if inference_csv.exists():
        lines.append("## 4. Inference Trace Aggregate")
        lines.append("")
        lines.append(f"See `inference_trace_aggregate.csv` for full data.")
        lines.append("")

    pairwise_csv = OUTPUT_DIR / "pairwise_action_divergence_aggregate.csv"
    if pairwise_csv.exists():
        lines.append("## 5. Pairwise Action Divergence")
        lines.append("")
        lines.append(f"See `pairwise_action_divergence_aggregate.csv` for full data.")
        lines.append("")

    velocity_csv = OUTPUT_DIR / "velocity_statistics.csv"
    if velocity_csv.exists():
        lines.append("## 6. Velocity Statistics")
        lines.append("")
        lines.append(f"See `velocity_statistics.csv` for full data.")
        lines.append("")

    lines.append("## 7. Key Research Questions")
    lines.append("")
    lines.append("### Q1: ours_v1 -> v2 -> v3 -> v4 attention pattern evolution")
    lines.append("")
    lines.append("As pc_success increases from 20.0 -> 25.5 -> 32.0 -> 34.5,")
    lines.append("does a stable attention spatial pattern change exist across multiple seeds?")
    lines.append("")
    lines.append("### Q2: v3 vs random (both 32.0) internal differences")
    lines.append("")
    lines.append("When v3 and random share the same 32.0 task success,")
    lines.append("do their internal attention and action trajectories still show clear differences?")
    lines.append("")
    lines.append("### Q3: v4 vs random (34.5 vs 32.0) post-grasp/pre-place changes")
    lines.append("")
    lines.append("Does v4's advantage over random accompany stable post-grasp or pre-place attention changes?")
    lines.append("")
    lines.append("### Q4: v4 action-related design consistency vs v3")
    lines.append("")
    lines.append("After v4 adds action-related design, does it show more consistent changes")
    lines.append("in v_t, hidden representation, or final action trajectory compared to v3?")
    lines.append("")
    lines.append("### Q5: Baseline trend support")
    lines.append("")
    lines.append("Do uniform, zero, and random baselines support the same trends?")
    lines.append("")
    lines.append("## 8. Correlation vs Causality Disclaimer")
    lines.append("")
    lines.append("**This report describes observable correlations only.**")
    lines.append("If an attention metric ordering matches the success ordering,")
    lines.append("this does NOT imply causation. If different seeds show inconsistent results,")
    lines.append("this must be explicitly marked as unstable. If only a single seed shows a difference,")
    lines.append("it must be marked as a single-seed artifact.")
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by aggregate_12k_corner.py*")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {md_path}")


def main():
    print("=" * 80)
    print("12k Corner 7-Model Attention Analysis Aggregation")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/8] Loading model metadata...")
    metadata_records = load_model_metadata()

    print("\n[2/8] Aggregating attention metrics...")
    aggregate_attention_metrics(metadata_records)

    print("\n[3/8] Aggregating phase reach statistics...")
    aggregate_phase_reach(metadata_records)

    print("\n[4/8] Aggregating heatmap statistics...")
    aggregate_heatmap_statistics()

    print("\n[5/8] Aggregating inference trace...")
    aggregate_inference_trace()

    print("\n[6/8] Aggregating pairwise divergence...")
    aggregate_pairwise_divergence()

    print("\n[7/8] Aggregating velocity statistics...")
    aggregate_velocity_statistics()

    print("\n[8/8] Writing model performance metadata and summary report...")
    write_model_performance_metadata()
    write_summary_report()

    print("\n" + "=" * 80)
    print("DONE. All outputs saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()