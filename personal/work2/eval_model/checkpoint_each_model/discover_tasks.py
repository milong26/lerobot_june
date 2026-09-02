#!/usr/bin/env python
"""Discover all valid model-checkpoint pairs under personal/work2/duibi."""
import json
import os
import re
import sys
from pathlib import Path

SCAN_ROOT = Path(os.environ.get("DISCOVER_SCAN_ROOT", "personal/work2/duibi"))

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)

def discover():
    if not SCAN_ROOT.is_dir():
        print(f"[ERROR] Scan root does not exist: {SCAN_ROOT}", file=sys.stderr)
        sys.exit(1)

    tasks = []
    for cp_root in sorted(SCAN_ROOT.rglob("checkpoints")):
        run_dir = cp_root.parent
        rel = run_dir.relative_to(SCAN_ROOT)
        model_name = sanitize(str(rel))

        for step_dir in sorted(cp_root.iterdir()):
            if not step_dir.is_dir():
                continue
            step_name = step_dir.name
            if not re.fullmatch(r'[0-9]+', step_name):
                continue
            pretrained = step_dir / "pretrained_model"
            if not pretrained.is_dir():
                continue
            step_num = int(step_name)
            tasks.append({
                "model": model_name,
                "checkpoint": step_name,
                "step_num": step_num,
                "checkpoint_path": str(pretrained),
            })

    tasks.sort(key=lambda t: (t["model"], t["step_num"]))

    for t in tasks:
        print(json.dumps(t))

if __name__ == "__main__":
    discover()