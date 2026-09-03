"""
DifferentVLM Experiment Main Entry Point

Orchestrates the full experiment pipeline:
1. VLM embedding extraction
2. V4 episode selection
3. SmolVLA training
4. Evaluation
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from differentvlm.configs.vlm_config import get_config
from differentvlm.vlm_extract.auto_extract_embedding import auto_extract_embedding
from differentvlm.selection.select_v4_wrapper import run_v4_selection
from differentvlm.train.train_smolvla_wrapper import run_smolvla_training
from differentvlm.eval.eval_wrapper import run_eval


def run_experiment(vlm_name: str, gpu_id: int, camera: str = "corner"):
    """Run the complete differentvlm experiment pipeline."""
    print(f"\n{'#'*60}")
    print(f"# DifferentVLM Experiment: {vlm_name}")
    print(f"# Camera: {camera}")
    print(f"# GPU: {gpu_id}")
    print(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    overall_start = time.time()

    cfg = get_config(vlm_name=vlm_name, gpu_id=gpu_id, camera=camera)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    results = {
        "vlm_name": vlm_name,
        "camera": camera,
        "gpu_id": gpu_id,
        "config": {
            "hf_model_id": cfg.hf_model_id,
            "pca_dim": cfg.pca_dim,
            "selection_num_episodes": cfg.selection_num_episodes,
            "selection_seed": cfg.selection_seed,
            "train_steps": cfg.train_steps,
            "eval_n_episodes": cfg.eval_n_episodes,
        },
        "stages": {},
        "final_metrics": {},
    }

    try:
        print(f"\n{'='*60}")
        print(f"Stage 1: VLM Embedding Extraction")
        print(f"{'='*60}")
        stage_start = time.time()
        embedding_dir = auto_extract_embedding(cfg)
        stage_time = time.time() - stage_start
        results["stages"]["embedding"] = {
            "status": "success",
            "embedding_dir": embedding_dir,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 1 complete: {stage_time:.1f}s")

        print(f"\n{'='*60}")
        print(f"Stage 2: V4 Episode Selection")
        print(f"{'='*60}")
        stage_start = time.time()
        subset_file = run_v4_selection(cfg)
        stage_time = time.time() - stage_start
        results["stages"]["selection"] = {
            "status": "success",
            "subset_file": subset_file,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 2 complete: {stage_time:.1f}s")

        print(f"\n{'='*60}")
        print(f"Stage 3: SmolVLA Training")
        print(f"{'='*60}")
        stage_start = time.time()
        checkpoint_dir = run_smolvla_training(cfg, subset_file)
        stage_time = time.time() - stage_start
        results["stages"]["training"] = {
            "status": "success",
            "checkpoint_dir": checkpoint_dir,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 3 complete: {stage_time:.1f}s")

        print(f"\n{'='*60}")
        print(f"Stage 4: Evaluation")
        print(f"{'='*60}")
        stage_start = time.time()
        eval_metrics = run_eval(cfg, checkpoint_dir)
        stage_time = time.time() - stage_start
        results["stages"]["evaluation"] = {
            "status": "success",
            "time_seconds": round(stage_time, 1),
        }
        results["final_metrics"] = eval_metrics
        print(f"Stage 4 complete: {stage_time:.1f}s")

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        results["error"] = str(e)
        results["stages"]["failed_at"] = "unknown"

    overall_time = time.time() - overall_start
    results["total_time_seconds"] = round(overall_time, 1)

    summary_file = Path(cfg.results_dir) / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# Experiment Complete")
    print(f"# Total time: {overall_time/3600:.2f} hours ({overall_time/60:.1f} minutes)")
    print(f"# Summary: {summary_file}")
    if "final_metrics" in results and results["final_metrics"]:
        print(f"# pc_success: {results['final_metrics'].get('pc_success', 'N/A')}")
        print(f"# pc_grasp_success: {results['final_metrics'].get('pc_grasp_success', 'N/A')}")
    print(f"{'#'*60}")

    sys.stdout.flush()
    return results


def main():
    parser = argparse.ArgumentParser(description="DifferentVLM Experiment Runner")
    parser.add_argument("--vlm", type=str, required=True,
                       choices=["llava_pythia400m", "prismatic_qwen25_05b"],
                       help="VLM backbone name")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID")
    parser.add_argument("--camera", type=str, default="corner",
                       help="Camera configuration (default: corner)")
    args = parser.parse_args()

    run_experiment(vlm_name=args.vlm, gpu_id=args.gpu, camera=args.camera)


if __name__ == "__main__":
    main()