#!/usr/bin/env python
"""
MiniVLA Checkpoint Load Verification Runner

Usage:
    python run_verify_minivla.py [--step N] [--checkpoint-path PATH]

Examples:
    # Verify step 8000 checkpoint
    python run_verify_minivla.py --step 8000

    # Use specific checkpoint path
    python run_verify_minivla.py --checkpoint-path outputs/minivla/checkpoints/008000/pretrained_model
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from differentvlm.eval.eval_minivla_wrapper import find_all_checkpoints


def main():
    parser = argparse.ArgumentParser(description="Verify MiniVLA checkpoint loading")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step number")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Full checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Default paths
    base_dir = PROJECT_ROOT / "outputs" / "minivla_pick-place-v3_corner"
    checkpoint_dir = args.checkpoint_dir or str(base_dir / "checkpoints")

    print(f"=" * 80)
    print(f"MiniVLA Checkpoint Load Verification")
    print(f"=" * 80)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Device:         {args.device}")
    print(f"Seed:           {args.seed}")
    print(f"=" * 80)

    # Determine checkpoint path
    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
    elif args.step:
        ckpt_path = str(Path(checkpoint_dir) / f"{args.step:06d}" / "pretrained_model")
    else:
        # Use latest checkpoint
        checkpoints = find_all_checkpoints(checkpoint_dir)
        if not checkpoints:
            print(f"ERROR: No valid checkpoint found in {checkpoint_dir}")
            sys.exit(1)
        step_num, ckpt_path = checkpoints[-1]
        print(f"Using latest checkpoint: step {step_num:06d}")

    if not Path(ckpt_path).exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\nUsing checkpoint: {ckpt_path}\n")

    # Run verification
    sys.argv = [
        "verify_minivla_checkpoint_load.py",
        "--checkpoint_path", ckpt_path,
        "--device", args.device,
        "--seed", str(args.seed),
    ]

    # Import and run the verification module
    exec(open(PROJECT_ROOT / "personal" / "work2" / "differentvlm" / "debug" / "verify_minivla_checkpoint_load.py").read())


if __name__ == "__main__":
    main()