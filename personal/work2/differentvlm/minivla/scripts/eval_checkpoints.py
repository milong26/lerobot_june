"""
MiniVLA Checkpoint Evaluation CLI

Auto-discovers all checkpoints in a training directory and evaluates each one.
Output format matches differentvlm eval results.
"""

import sys
import os
import argparse
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
WORK2_ROOT = Path(__file__).resolve().parents[3]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from differentvlm.minivla.configs.minivla_config import get_minivla_config
from differentvlm.minivla.eval.eval_minivla import run_eval_all_checkpoints


def main():
    parser = argparse.ArgumentParser(description="MiniVLA Checkpoint Evaluation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--dataset", type=str, default="disassemble-v3_corner",
                       help="Dataset name")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Path to training checkpoints directory")
    args = parser.parse_args()

    cfg = get_minivla_config(
        gpu_id=args.gpu,
        dataset_name=args.dataset,
    )
    cfg.ensure_dirs()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_eval_all_checkpoints(cfg, args.checkpoint_dir)


if __name__ == "__main__":
    main()