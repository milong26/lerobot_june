#!/usr/bin/env python3
"""批量检查三个模型（TinyVLA-S, TinyVLA-B, MiniVLA）在12k step的action pipeline一致性。"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies import PreTrainedPolicy

ACTION = "action"

# 三个模型的checkpoint路径
DEFAULT_CHECKPOINTS = {
    "tinyvla_s": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_s_disassemble-v3_corner/checkpoints/012000/pretrained_model",
    "tinyvla_b": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_b_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_b_disassemble-v3_corner/checkpoints/012000/pretrained_model",
    "minivla": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_112_seed42_disassemblev3corner_minivla_random/checkpoints/checkpoints/012000/pretrained_model",
}


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("batch_check")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def load_policy_and_processors(policy_type: str, checkpoint_path: str, device: str, dataset_stats: dict | None = None, logger: logging.Logger | None = None):
    log = logger or logging.getLogger("batch_check")
    log.info(f"Loading {policy_type} from {checkpoint_path}")

    from lerobot.policies.pretrained import PreTrainedConfig, PreTrainedPolicy
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies import make_pre_post_processors

    # Load config from checkpoint
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    config.pretrained_path = Path(checkpoint_path)
    if not hasattr(config, "type"):
        config.type = policy_type

    # Load pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        dataset_stats=dataset_stats,
    )

    # Load policy
    policy_cls = get_policy_class(policy_type)
    policy = PreTrainedPolicy.from_pretrained.__func__(policy_cls, checkpoint_path)
    policy.to(device)
    policy.eval()

    log.info(f"Policy loaded: {type(policy).__name__}")
    return policy, preprocessor, postprocessor, config


def run_single_policy_check(policy_type: str, checkpoint_path: str, sample: dict, config: dict, device: str, dataset_stats: dict | None = None, logger: logging.Logger | None = None):
    log = logger or logging.getLogger("batch_check")
    log.info(f"--- Checking: {policy_type} ---")

    try:
        policy, preprocessor, postprocessor, policy_config = load_policy_and_processors(
            policy_type, checkpoint_path, device, dataset_stats, logger
        )
    except Exception as e:
        log.error(f"Failed to load {policy_type}: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}

    # 1. Raw action stats
    raw_act = sample[ACTION]
    if isinstance(raw_act, np.ndarray):
        raw_act_t = torch.from_numpy(raw_act).float().to(device)
    else:
        raw_act_t = raw_act.float().to(device)

    log.info(f"Raw action: shape={raw_act_t.shape}, min={raw_act_t.min():.4f}, max={raw_act_t.max():.4f}")

    # 2. Preprocess
    preprocessed = preprocessor(sample)
    if isinstance(preprocessed, dict) and ACTION in preprocessed:
        preprocessed_action = preprocessed[ACTION]
        if isinstance(preprocessed_action, torch.Tensor):
            log.info(f"Preprocessed action: shape={preprocessed_action.shape}, min={preprocessed_action.min():.4f}, max={preprocessed_action.max():.4f}")

    # 3. Model forward
    with torch.no_grad():
        if hasattr(policy, "predict_action_chunk"):
            pred_actions = policy.predict_action_chunk(preprocessed)
        else:
            pred_actions = policy(preprocessed)

    if isinstance(pred_actions, torch.Tensor):
        log.info(f"Model prediction: shape={pred_actions.shape}, min={pred_actions.min():.4f}, max={pred_actions.max():.4f}")
    else:
        log.info(f"Model prediction type: {type(pred_actions)}")

    # 4. Postprocess
    if isinstance(pred_actions, torch.Tensor):
        final_actions = postprocessor(pred_actions)
        if isinstance(final_actions, torch.Tensor):
            log.info(f"Final action: shape={final_actions.shape}, min={final_actions.min():.4f}, max={final_actions.max():.4f}")

            # 5. Compare with real action
            if final_actions.ndim == 3:
                pred_step0 = final_actions[0, 0, :]
            elif final_actions.ndim == 2:
                pred_step0 = final_actions[0, :]
            else:
                pred_step0 = final_actions

            abs_error = (raw_act_t - pred_step0.float()).abs()
            max_err = float(abs_error.max())
            mean_err = float(abs_error.mean())

            log.info(f"\nPrediction vs Real:")
            log.info(f"  Real:     {raw_act_t.tolist()}")
            log.info(f"  Predicted: {pred_step0.float().tolist()}")
            log.info(f"  Max error:  {max_err:.4f}")
            log.info(f"  Mean error: {mean_err:.4f}")

            return {
                "raw_action": {"shape": list(raw_act_t.shape), "min": float(raw_act_t.min()), "max": float(raw_act_t.max())},
                "prediction": {"shape": list(pred_actions.shape), "min": float(pred_actions.min()), "max": float(pred_actions.max())},
                "final_action": {"shape": list(final_actions.shape), "min": float(final_actions.min()), "max": float(final_actions.max())},
                "comparison": {
                    "real_action": raw_act_t.tolist(),
                    "predicted_step0": pred_step0.float().tolist(),
                    "max_absolute_error": max_err,
                    "mean_absolute_error": mean_err,
                },
            }

    return {"error": "Failed to get final actions"}


def main():
    parser = argparse.ArgumentParser(description="Batch check 12k step models")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None, help="Custom checkpoint paths")
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_file = str(Path(__file__).parent / "batch_check_12k.log")
    logger = setup_logging(log_file)

    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Episode: {args.episode_index}")
    logger.info(f"GPU ID: {args.gpu_id}")
    logger.info(f"Device: {device}")

    # Load dataset
    dataset = LeRobotDataset(args.dataset_root)
    logger.info(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")

    # Get sample - find frames belonging to the episode
    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    mask = episode_indices == args.episode_index
    indices = np.where(mask)[0]
    if len(indices) == 0:
        logger.error(f"Episode {args.episode_index} not found")
        return
    frame_idx = int(indices[len(indices) // 2])
    sample = dataset[frame_idx]
    logger.info(f"Loaded frame {frame_idx} from episode {args.episode_index}")

    # Dataset stats
    ds_stats = dataset.meta.stats if hasattr(dataset.meta, "stats") else None

    # Check all three models
    results = {}
    checkpoints = args.checkpoints if args.checkpoints else DEFAULT_CHECKPOINTS

    for policy_type, ckpt_path in checkpoints.items():
        result = run_single_policy_check(
            policy_type=policy_type,
            checkpoint_path=ckpt_path,
            sample=sample,
            config={},
            device=device,
            dataset_stats=ds_stats,
            logger=logger,
        )
        results[policy_type] = result

    # Summary
    logger.info("--- SUMMARY ---")
    for policy_type, result in results.items():
        if "comparison" in result:
            comp = result["comparison"]
            logger.info(f"{policy_type:12s}: max_err={comp['max_absolute_error']:.4f}, mean_err={comp['mean_absolute_error']:.4f}")
        else:
            logger.info(f"{policy_type:12s}: ERROR - {result.get('error', 'unknown')}")

    # Save results
    output_file = args.output_file or str(Path(__file__).parent / "batch_check_12k_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()