#!/usr/bin/env python
"""Rebuild unified summary.csv from all per-checkpoint JSON files."""
import csv
import fcntl
import json
import os
import sys
from pathlib import Path

def build_summary(summary_dir, output_csv):
    summary_dir = Path(summary_dir)
    output_csv = Path(output_csv)

    rows = []
    for json_file in sorted(summary_dir.rglob("*.json")):
        if json_file.name.startswith("."):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            rows.append(data)
        except Exception:
            continue

    if not rows:
        return

    fieldnames = [
        "model", "checkpoint", "checkpoint_path", "gpu",
        "n_episodes", "success_count", "grasp_success_count",
        "success_rate", "grasp_success_rate",
        "pc_success", "pc_grasp_success",
        "status", "exitcode", "log_path", "log_file", "result_file"
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    tmp.rename(output_csv)

if __name__ == "__main__":
    summary_dir = sys.argv[1]
    output_csv = sys.argv[2]
    build_summary(summary_dir, output_csv)