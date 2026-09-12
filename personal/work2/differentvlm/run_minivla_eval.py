#!/usr/bin/env python
"""
MiniVLA Evaluation Runner

Usage:
    cd /data/zhonglinye/jun/lerobot
    python personal/work2/differentvlm/run_minivla_eval.py [--episodes N] [--step N]

Examples:
    # 20k step smoke test: 20 episodes
    python personal/work2/differentvlm/run_minivla_eval.py --episodes 20 --step 20000

    # 20k step full eval: 200 episodes
    python personal/work2/differentvlm/run_minivla_eval.py --episodes 200 --step 20000

    # Use custom checkpoint path
    python personal/work2/differentvlm/run_minivla_eval.py \
        --checkpoint-path personal/work2/differentvlm/minivla/experiments/random_200_seed42_disassemblev3corner_minivla_random/checkpoints/checkpoints/020000/pretrained_model \
        --episodes 200
"""

import sys
import os
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
# File: personal/work2/differentvlm/run_minivla_eval.py
# parents[0] = differentvlm
# parents[1] = work2
# parents[2] = personal
# parents[3] = lerobot (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def run_minivla_eval_cli(
    checkpoint_path: str,
    env_type: str = "metaworld",
    env_task: str = "disassemble-v3",
    env_camera_name: str = "corner,gripperPOV",
    env_use_self_mw: bool = True,
    eval_batch_size: int = 8,
    eval_n_episodes: int = 200,
    policy_device: str = "cuda",
    policy_use_amp: bool = True,
    rename_map: str = '{"observation.images.camera1": "observation.images.top", "observation.images.camera2": "observation.images.wrist"}',
    log_file: str = None,
) -> dict:
    """
    Run MiniVLA evaluation using lerobot-eval CLI.
    
    Returns:
        dict with status, returncode, log_file, and parsed results
    """
    # Create independent output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("personal/work2/eval_model") / f"minivla_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_file or str(output_dir / "eval.log")
    
    # Build lerobot-eval command
    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        f"--env.type={env_type}",
        f"--env.task={env_task}",
        f"--env.camera_name={env_camera_name}",
        f"--env.use_self_mw={'true' if env_use_self_mw else 'false'}",
        f"--eval.batch_size={eval_batch_size}",
        f"--eval.n_episodes={eval_n_episodes}",
        f"--policy.device={policy_device}",
        f"--policy.use_amp={'true' if policy_use_amp else 'false'}",
        f"--rename_map={rename_map}",
    ]
    
    print(f"=" * 80)
    print(f"MiniVLA Evaluation")
    print(f"=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Env task:   {env_task}")
    print(f"Episodes:   {eval_n_episodes}")
    print(f"Batch size: {eval_batch_size}")
    print(f"Device:     {policy_device}")
    print(f"Log file:   {log_file}")
    print(f"=" * 80)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run evaluation
    with open(log_file, "w") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        
        process.wait()
    
    returncode = process.returncode
    
    # Parse results from output directory
    result = {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "checkpoint_path": checkpoint_path,
        "episodes": eval_n_episodes,
        "log_file": log_file,
        "output_dir": str(output_dir),
        "timestamp": timestamp,
        "pc_success": None,
        "pc_grasp_success": None,
    }
    
    # Try to find eval results JSON
    for json_file in output_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            aggregated = data.get("aggregated", {})
            if "pc_success" in aggregated:
                result["pc_success"] = aggregated["pc_success"]
                result["pc_grasp_success"] = aggregated.get("pc_grasp_success")
                break
        except (json.JSONDecodeError, IOError):
            continue
    
    # Also check parent directory
    if result["pc_success"] is None:
        for json_file in output_dir.parent.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                aggregated = data.get("aggregated", {})
                if "pc_success" in aggregated:
                    result["pc_success"] = aggregated["pc_success"]
                    result["pc_grasp_success"] = aggregated.get("pc_grasp_success")
                    break
            except (json.JSONDecodeError, IOError):
                continue
    
    if returncode != 0:
        result["status"] = "failed"
        result["error"] = f"Process exited with code {returncode}"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run MiniVLA evaluation")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--step", type=int, default=20000, help="Checkpoint step number (default: 20000)")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Full checkpoint path")
    parser.add_argument("--env-task", type=str, default="disassemble-v3", help="Environment task")
    parser.add_argument("--env-camera", type=str, default="corner,gripperPOV", help="Camera names")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    args = parser.parse_args()

    # Default checkpoint path for 20k step
    default_checkpoint_path = (
        "personal/work2/differentvlm/minivla/experiments/"
        "random_200_seed42_disassemblev3corner_minivla_random_official/"
        "checkpoints/checkpoints/020000/pretrained_model"
    )
    
    checkpoint_path = args.checkpoint_path or default_checkpoint_path
    
    # If step is specified but no checkpoint path, build path from step
    if args.checkpoint_path is None and args.step != 20000:
        checkpoint_path = (
            f"personal/work2/differentvlm/minivla/experiments/"
            f"random_200_seed42_disassemblev3corner_minivla_random_official/"
            f"checkpoints/checkpoints/{args.step:06d}/pretrained_model"
        )
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Run evaluation
    result = run_minivla_eval_cli(
        checkpoint_path=checkpoint_path,
        env_task=args.env_task,
        env_camera_name=args.env_camera,
        eval_batch_size=args.batch_size,
        eval_n_episodes=args.episodes,
        policy_device=args.device,
        log_file=args.log_file,
    )

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Evaluation Results")
    print(f"{'=' * 80}")
    print(f"Status:          {result['status']}")
    print(f"Checkpoint:      {result['checkpoint_path']}")
    print(f"Episodes:        {result['episodes']}")
    print(f"Return code:     {result['returncode']}")
    print(f"PC Success:      {result['pc_success']}")
    print(f"PC Grasp Success:{result['pc_grasp_success']}")
    print(f"Log file:        {result['log_file']}")
    print(f"{'=' * 80}")

    if result["status"] == "failed":
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()