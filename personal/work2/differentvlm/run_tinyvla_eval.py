#!/usr/bin/env python
"""
TinyVLA Evaluation Runner

Usage:
    cd /data/zhonglinye/jun/lerobot
    python personal/work2/differentvlm/run_tinyvla_eval.py [--episodes N] [--n-action-steps N]

Examples:
    # Smoke test: 20 episodes, n_action_steps=1
    python personal/work2/differentvlm/run_tinyvla_eval.py --episodes 20 --n-action-steps 1

    # Full eval: 200 episodes
    python personal/work2/differentvlm/run_tinyvla_eval.py --episodes 200 --n-action-steps 1
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
# File: personal/work2/differentvlm/run_tinyvla_eval.py
# parents[0] = differentvlm
# parents[1] = work2
# parents[2] = personal
# parents[3] = lerobot (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORK2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WORK2_ROOT))

from differentvlm.configs.vlm_config import VLMExperimentConfig
from differentvlm.eval.eval_tinyvla_wrapper import run_tinyvla_eval, find_best_checkpoint


# ============================================================================
# Configuration (matches lerobot-eval command)
# ============================================================================
CHECKPOINT_PATH = str(
    PROJECT_ROOT / "personal" / "work2" / "differentvlm" / "experiments" / 
    "tinyvla_b_disassemble-v3_corner" / "checkpoints" / 
    "tinyvla_tinyvla_b_disassemble-v3_corner" / "checkpoints" / "012000" / "pretrained_model"
)

ENV_TASK = "disassemble-v3"
CAMERA_NAMES = "corner,gripperPOV"
RENAME_MAP = '{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

EVAL_EPISODES = 200
EVAL_BATCH_SIZE = 8
EVAL_SEED = 42
N_ACTION_STEPS = 1
GPU_ID = 1

RESULTS_DIR = str(PROJECT_ROOT / "personal" / "work2" / "eval_model" / "results")
LOGS_DIR = str(PROJECT_ROOT / "personal" / "work2" / "eval_model" / "logs")
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run TinyVLA evaluation")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to evaluate (overrides default)")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Number of action steps (overrides default)")
    args = parser.parse_args()

    # Use command line args if provided, otherwise use defaults
    n_episodes = args.episodes if args.episodes is not None else EVAL_EPISODES
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else N_ACTION_STEPS

    print(f"=" * 80)
    print(f"TinyVLA Evaluation")
    print(f"=" * 80)
    print(f"Checkpoint:     {CHECKPOINT_PATH}")
    print(f"Env task:       {ENV_TASK}")
    print(f"Camera names:   {CAMERA_NAMES}")
    print(f"Episodes:       {n_episodes}")
    print(f"N action steps: {n_action_steps}")
    print(f"Batch size:     {EVAL_BATCH_SIZE}")
    print(f"Seed:           {EVAL_SEED}")
    print(f"GPU ID:         {GPU_ID}")
    print(f"Results dir:    {RESULTS_DIR}")
    print(f"Logs dir:       {LOGS_DIR}")
    print(f"=" * 80)

    # Create config
    cfg = VLMExperimentConfig(
        vlm_name="tinyvla_b",
        dataset_name=f"metaworld_{ENV_TASK.replace('-', '_')}",
        selection_vlm_model_id="lesjie/Llava-Pythia-400M",
        selection_vlm_description="TinyVLA-B (LLaVA-Pythia-400M + Diffusion Action Head)",
        env_task=ENV_TASK,
        camera_names=CAMERA_NAMES,
        eval_n_episodes=n_episodes,
        eval_batch_size=EVAL_BATCH_SIZE,
        eval_seed=EVAL_SEED,
        gpu_id=GPU_ID,
        results_dir=RESULTS_DIR,
        logs_dir=LOGS_DIR,
        rename_map=RENAME_MAP,
    )

    # Get checkpoint directory (parent of pretrained_model)
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint path does not exist: {CHECKPOINT_PATH}")
        sys.exit(1)
    
    print(f"\nUsing checkpoint: {checkpoint_path}\n")

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Run evaluation
    # checkpoint_path points to .../012000/pretrained_model
    # find_best_checkpoint expects the parent directory containing step subdirectories
    # So we need to go up 2 levels: pretrained_model -> 012000 -> checkpoints/
    result = run_tinyvla_eval(cfg, str(checkpoint_path.parent.parent), n_action_steps=n_action_steps)

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Evaluation Results")
    print(f"{'=' * 80}")
    print(f"Status:          {result['status']}")
    print(f"Checkpoint:      {result.get('checkpoint_path', 'N/A')}")
    print(f"Checkpoint step: {result.get('checkpoint_step', 'N/A')}")
    print(f"N action steps:  {result.get('n_action_steps', 'N/A')}")
    print(f"Episodes:        {result.get('n_episodes', 'N/A')}")
    print(f"Return code:     {result.get('returncode', 'N/A')}")
    
    agg = result.get("aggregated", {})
    print(f"PC Success:      {agg.get('pc_success', 'N/A')}")
    print(f"PC Grasp Success:{agg.get('pc_grasp_success', 'N/A')}")
    print(f"{'=' * 80}")

    if result["status"] == "failed":
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    else:
        print(f"\nResults saved to: {result.get('output_dir', 'N/A')}")
        print(f"Log file:         {result.get('log_file', 'N/A')}")


if __name__ == "__main__":
    main()