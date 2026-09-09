#!/usr/bin/env python3
"""对完整 episode 进行推理，对比真实 ground truth 和模型预测结果。"""

import argparse
import json
import logging
import os
import traceback
from pathlib import Path


import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedConfig, PreTrainedPolicy
from lerobot.policies.factory import get_policy_class
from lerobot.policies import make_pre_post_processors

ACTION = "action"

DEFAULT_CHECKPOINTS = {
    "tinyvla_s": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_s_disassemble-v3_corner/checkpoints/012000/pretrained_model",
    "tinyvla_b": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_b_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_b_disassemble-v3_corner/checkpoints/012000/pretrained_model",
    "minivla": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_112_seed42_disassemblev3corner_minivla_random/checkpoints/checkpoints/012000/pretrained_model",
}


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("episode_eval")
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


def load_policy(policy_type: str, checkpoint_path: str, device: str, dataset_stats: dict | None = None, logger: logging.Logger | None = None):
    log = logger or logging.getLogger("episode_eval")
    log.info(f"Loading {policy_type} from {checkpoint_path}")

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    config.pretrained_path = Path(checkpoint_path)
    if not hasattr(config, "type"):
        config.type = policy_type

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        dataset_stats=dataset_stats,
    )

    policy_cls = get_policy_class(policy_type)
    policy = PreTrainedPolicy.from_pretrained.__func__(policy_cls, checkpoint_path)

    model_dtype = getattr(config, "dtype", "float32")
    if model_dtype == "bfloat16":
        policy = policy.to(dtype=torch.bfloat16)
    elif model_dtype == "float16":
        policy = policy.to(dtype=torch.float16)

    policy.to(device)
    policy.eval()

    log.info(f"Policy loaded: {type(policy).__name__}, dtype={model_dtype}")
    return policy, preprocessor, postprocessor, config


def eval_episode(policy_type: str, checkpoint_path: str, dataset_root: str, episode_index: int, device: str, output_dir: str, logger: logging.Logger):
    log = logger
    log.info(f"=== Evaluating {policy_type} on episode {episode_index} ===")

    # Load policy
    policy, preprocessor, postprocessor, config = load_policy(
        policy_type, checkpoint_path, device, dataset_stats=None, logger=logger
    )

    # Load dataset
    dataset = LeRobotDataset(dataset_root)
    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    mask = episode_indices == episode_index
    frame_indices = np.where(mask)[0]

    if len(frame_indices) == 0:
        log.error(f"Episode {episode_index} not found")
        return

    log.info(f"Episode has {len(frame_indices)} frames")

    # Patch MiniVLA's VQ action tokenizer to handle bfloat16 -> float32 conversion
    if policy_type == "minivla":
        try:
            core = policy.model
            if hasattr(core, "action_tokenizer"):
                action_tokenizer = core.action_tokenizer
                if hasattr(action_tokenizer, "vq_vae"):
                    original_get_action = action_tokenizer.vq_vae.get_action_from_latent

                    def patched_get_action(latent):
                        result = original_get_action(latent)
                        if result.dtype == torch.bfloat16:
                            return result.float()
                        return result

                    action_tokenizer.vq_vae.get_action_from_latent = patched_get_action
                    log.info("Patched MiniVLA vq_vae.get_action_from_latent for bfloat16 compatibility")
        except Exception as e:
            log.warning(f"Failed to patch MiniVLA action tokenizer: {e}")

    # Get model dtype
    model_dtype = getattr(config, "dtype", "float32")
    if model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Evaluate each frame
    results = {
        "model": policy_type,
        "checkpoint": checkpoint_path,
        "dataset": dataset_root,
        "episode_index": episode_index,
        "num_frames": len(frame_indices),
        "per_frame": [],
        "summary": {},
    }

    all_real_actions = []
    all_pred_actions = []
    all_errors = []

    for i, frame_idx in enumerate(frame_indices):
        sample = dataset[frame_idx]
        if ACTION not in sample:
            continue

        raw_act = sample[ACTION]
        if isinstance(raw_act, np.ndarray):
            raw_act_t = torch.from_numpy(raw_act).float().to(device)
        else:
            raw_act_t = raw_act.float().to(device)

        # Preprocess
        preprocessed = preprocessor(sample)
        if isinstance(preprocessed, dict):
            for k, v in preprocessed.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    preprocessed[k] = v.to(dtype=torch_dtype)

        # Forward
        with torch.no_grad():
            if hasattr(policy, "predict_action_chunk"):
                pred_actions = policy.predict_action_chunk(preprocessed)
            else:
                pred_actions = policy(preprocessed)

        # Postprocess
        if isinstance(pred_actions, torch.Tensor):
            final_actions = postprocessor(pred_actions)
            if isinstance(final_actions, torch.Tensor):
                if final_actions.ndim == 3:
                    pred_step0 = final_actions[0, 0, :]
                elif final_actions.ndim == 2:
                    pred_step0 = final_actions[0, :]
                else:
                    pred_step0 = final_actions

                pred_step0 = pred_step0.float().to(raw_act_t.device)
                abs_error = (raw_act_t - pred_step0).abs()
                max_err = float(abs_error.max())
                mean_err = float(abs_error.mean())

                all_real_actions.append(raw_act_t.cpu().numpy().tolist())
                all_pred_actions.append(pred_step0.cpu().numpy().tolist())
                all_errors.append(max_err)

                frame_result = {
                    "frame_index": int(frame_idx),
                    "real_action": raw_act_t.cpu().numpy().tolist(),
                    "predicted_action": pred_step0.cpu().numpy().tolist(),
                    "abs_error_per_dim": abs_error.cpu().numpy().tolist(),
                    "max_absolute_error": max_err,
                    "mean_absolute_error": mean_err,
                }
                results["per_frame"].append(frame_result)

                if (i + 1) % 10 == 0:
                    log.info(f"  Processed {i + 1}/{len(frame_indices)} frames")

    # Summary statistics
    if all_errors:
        all_errors = np.array(all_errors)
        all_real = np.array(all_real_actions)
        all_pred = np.array(all_pred_actions)

        results["summary"] = {
            "max_error_over_episode": float(np.max(all_errors)),
            "mean_error_over_episode": float(np.mean(all_errors)),
            "median_error": float(np.median(all_errors)),
            "min_error": float(np.min(all_errors)),
            "real_action_mean": all_real.mean(axis=0).tolist(),
            "real_action_std": all_real.std(axis=0).tolist(),
            "predicted_action_mean": all_pred.mean(axis=0).tolist(),
            "predicted_action_std": all_pred.std(axis=0).tolist(),
        }

        log.info(f"\n--- Summary ---")
        log.info(f"  Max error:  {results['summary']['max_error_over_episode']:.4f}")
        log.info(f"  Mean error: {results['summary']['mean_error_over_episode']:.4f}")
        log.info(f"  Median:     {results['summary']['median_error']:.4f}")
        log.info(f"  Min error:  {results['summary']['min_error']:.4f}")

    # Save to file
    output_file = Path(output_dir) / f"{policy_type}_episode_{episode_index}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nResults saved to: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate full episode")
    parser.add_argument("--policy_type", type=str, required=True, choices=list(DEFAULT_CHECKPOINTS.keys()))
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = args.output_dir or str(Path(__file__).parent / "episode_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = str(Path(output_dir) / f"{args.policy_type}_eval.log")
    logger = setup_logging(log_file)

    checkpoint_path = args.checkpoint_path or DEFAULT_CHECKPOINTS[args.policy_type]

    logger.info(f"Model: {args.policy_type}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Episode: {args.episode_index}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    eval_episode(
        policy_type=args.policy_type,
        checkpoint_path=checkpoint_path,
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()