#!/usr/bin/env python3
"""
验证 MiniVLA 训练进度和收敛性

目标：
1. Action→tokenizer encode→decode→action oracle测试：表征误差上限
2. 对多个checkpoint计算token和action指标
3. 区分"token预测欠拟合"、"VQ重建误差过大"、"checkpoint未加载"、"视觉动作映射未学会"

用法：
python verify_minivla_training_progress.py \
    --checkpoint_dir personal/work2/differentvlm/minivla/experiments/xxx/checkpoints \
    --dataset_root personal/work2/differentvlm/datasets/xxx \
    --device cuda \
    --eval_steps 12000 20000 30000 40000 50000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.utils.constants import ACTION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_checkpoints(checkpoint_dir: str, eval_steps: list[int] | None = None) -> list[dict]:
    """Find all available checkpoints in the checkpoint directory."""
    checkpoints_dir = Path(checkpoint_dir)
    if not checkpoints_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    available = []
    for d in sorted(checkpoints_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            step = int(d.name)
            pretrained_dir = d / "pretrained_model"
            if (pretrained_dir / "model.safetensors").exists() or (pretrained_dir / "config.json").exists():
                available.append({"step": step, "path": str(pretrained_dir), "dir_name": d.name})

    if eval_steps:
        # Filter to only requested steps
        available = [c for c in available if c["step"] in eval_steps]

    logger.info(f"Found {len(available)} checkpoints: {[c['step'] for c in available]}")
    return available


def run_action_tokenizer_oracle(dataset_root: str, dataset_repo_id: str, policy_config, device: str) -> dict:
    """
    Oracle test: action → tokenizer encode → decode → action
    This measures the irreducible representation error of the tokenizer itself.
    Even if the policy predicts tokens perfectly, it cannot beat this error.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ACTION TOKENIZER ORACLE TEST")
    logger.info("=" * 80)

    # Load a small sample of actions from the dataset
    ds = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
    )

    # Sample actions from multiple episodes
    num_samples = min(100, ds.num_frames)
    indices = np.linspace(0, ds.num_frames - 1, num_samples, dtype=int)

    actions = []
    for idx in indices:
        sample = ds[idx]
        if ACTION in sample:
            act = sample[ACTION]
            if isinstance(act, torch.Tensor):
                act = act.cpu().numpy()
            actions.append(act)

    if not actions:
        logger.error("No actions found in dataset!")
        return {}

    actions = np.array(actions)  # [N, action_dim]
    action_dim = actions.shape[-1]
    logger.info(f"Sampled {len(actions)} actions, action_dim={action_dim}")
    logger.info(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")

    # For non-VQ mode, use ActionTokenizer
    is_vq = "vq" in policy_config.action_tokenizer_type.lower()

    if is_vq:
        # VQ mode: test encode→decode roundtrip
        from lerobot.policies.minivla.vq_action import VQActionTokenizer
        from lerobot.policies.minivla.tokenizer import VLATokenizerWrapper

        tokenizer_wrapper = VLATokenizerWrapper(
            base_vlm_checkpoint=policy_config.base_vlm_checkpoint,
            num_extra_tokens=policy_config.num_extra_tokens,
        )

        vq_path = policy_config.resolve_vq_model_path()
        if not vq_path:
            logger.error("VQ model path not found!")
            return {}

        action_tokenizer = VQActionTokenizer(
            tokenizer=tokenizer_wrapper.tokenizer,
            vq_vae_path=vq_path,
            device=device,
            use_extra=True,
        )

        # Test with actions matching VQ expected shape
        vq_action_dim = action_tokenizer.vq_vae.input_dim_w
        vq_input_dim_h = action_tokenizer.vq_vae.input_dim_h

        logger.info(f"VQ config: input_dim_w={vq_action_dim}, input_dim_h={vq_input_dim_h}")

        if action_dim != vq_action_dim:
            logger.warning(
                f"Dataset action_dim ({action_dim}) != VQ input_dim_w ({vq_action_dim}). "
                f"Oracle test cannot run with mismatched dimensions."
            )
            return {"error": f"Dimension mismatch: dataset={action_dim}, VQ={vq_action_dim}"}

        # Reshape for VQ: [N, A] -> [N, 1, A] if input_dim_h=1, or [N, T, A] if input_dim_h=T
        if vq_input_dim_h == 1:
            actions_for_vq = actions[:, np.newaxis, :]  # [N, 1, A]
        else:
            # Need chunk_size actions per sample - use first chunk_size from dataset
            logger.warning(f"VQ requires input_dim_h={vq_input_dim_h}, using single-frame approximation")
            actions_for_vq = actions[:, np.newaxis, :]

        # Encode → decode
        per_dim_errors = []
        max_errors = []
        direction_accuracies = []
        gripper_accuracies = []

        batch_size = min(32, len(actions_for_vq))
        decoded_actions = []

        for start in range(0, len(actions_for_vq), batch_size):
            batch = actions_for_vq[start:start + batch_size]
            try:
                # Encode
                encoded = action_tokenizer.encode_token_ids(batch)
                # Decode
                decoded = action_tokenizer.decode_token_ids_to_actions(encoded)
                if decoded.ndim == 3:
                    decoded = decoded[:, 0, :]  # Take first timestep
                elif decoded.ndim == 2:
                    pass
                decoded_actions.append(decoded)
            except Exception as e:
                logger.warning(f"Encode/decode failed for batch at {start}: {e}")

        if decoded_actions:
            decoded_actions = np.concatenate(decoded_actions, axis=0)
            original_actions = actions_for_vq[:, 0, :] if actions_for_vq.ndim == 3 else actions_for_vq[:len(decoded_actions)]

            # Per-dimension MAE
            abs_errors = np.abs(decoded_actions - original_actions)
            per_dim_mae = abs_errors.mean(axis=0)
            per_dim_max = abs_errors.max(axis=0)

            # Direction/sign accuracy
            direction_correct = (np.sign(decoded_actions) == np.sign(original_actions)).mean(axis=0)

            # Gripper accuracy (typically last dimension, threshold=0.5)
            if action_dim > 0:
                gripper_pred = (decoded_actions[:, -1] > 0.5).astype(float)
                gripper_true = (original_actions[:, -1] > 0.5).astype(float)
                gripper_acc = (gripper_pred == gripper_true).mean()
            else:
                gripper_acc = 0.0

            oracle_results = {
                "per_dim_mae": per_dim_mae.tolist(),
                "per_dim_max_error": per_dim_max.tolist(),
                "mean_mae": float(per_dim_mae.mean()),
                "max_mae": float(per_dim_mae.max()),
                "direction_accuracy": direction_accuracies if direction_accuracies else direction_correct.tolist(),
                "gripper_accuracy": float(gripper_acc),
                "num_samples": len(decoded_actions),
            }

            logger.info(f"\nOracle Results ({len(decoded_actions)} samples):")
            logger.info(f"  Per-dim MAE: {per_dim_mae}")
            logger.info(f"  Per-dim Max Error: {per_dim_max}")
            logger.info(f"  Mean MAE: {per_dim_mae.mean():.6f}")
            logger.info(f"  Direction Accuracy: {direction_correct}")
            logger.info(f"  Gripper Accuracy: {gripper_acc:.4f}")

            return oracle_results

    else:
        # Non-VQ mode: use ActionTokenizer
        from lerobot.policies.minivla.vq_action import ActionTokenizer
        from lerobot.policies.minivla.tokenizer import VLATokenizerWrapper

        tokenizer_wrapper = VLATokenizerWrapper(
            base_vlm_checkpoint=policy_config.base_vlm_checkpoint,
            num_extra_tokens=policy_config.num_extra_tokens,
        )

        action_tokenizer = ActionTokenizer(
            tokenizer=tokenizer_wrapper.tokenizer,
            bins=256,
            min_action=-1,
            max_action=1,
            use_extra=True,
        )

        # Encode → decode
        decoded_actions = []
        for act in actions:
            try:
                encoded_text = action_tokenizer(act)
                # Get token IDs from text
                token_ids = tokenizer_wrapper.tokenizer(encoded_text)["input_ids"]
                decoded = action_tokenizer.decode_token_ids_to_actions(np.array(token_ids))
                decoded_actions.append(decoded)
            except Exception as e:
                logger.warning(f"Encode/decode failed: {e}")

        if decoded_actions:
            decoded_actions = np.array(decoded_actions)
            abs_errors = np.abs(decoded_actions - actions)
            per_dim_mae = abs_errors.mean(axis=0)
            per_dim_max = abs_errors.max(axis=0)
            direction_correct = (np.sign(decoded_actions) == np.sign(actions)).mean(axis=0)

            if action_dim > 0:
                gripper_pred = (decoded_actions[:, -1] > 0.5).astype(float)
                gripper_true = (actions[:, -1] > 0.5).astype(float)
                gripper_acc = (gripper_pred == gripper_true).mean()
            else:
                gripper_acc = 0.0

            oracle_results = {
                "per_dim_mae": per_dim_mae.tolist(),
                "per_dim_max_error": per_dim_max.tolist(),
                "mean_mae": float(per_dim_mae.mean()),
                "max_mae": float(per_dim_mae.max()),
                "direction_accuracy": direction_correct.tolist(),
                "gripper_accuracy": float(gripper_acc),
                "num_samples": len(decoded_actions),
            }

            logger.info(f"\nOracle Results ({len(decoded_actions)} samples):")
            logger.info(f"  Per-dim MAE: {per_dim_mae}")
            logger.info(f"  Mean MAE: {per_dim_mae.mean():.6f}")
            logger.info(f"  Gripper Accuracy: {gripper_acc:.4f}")

            return oracle_results

    return {"error": "Oracle test failed"}


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_root: str,
    dataset_repo_id: str,
    device: str,
    num_eval_samples: int = 50,
) -> dict:
    """Evaluate a single checkpoint on training data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"{'='*80}")

    # Load config
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device

    # Load policy
    policy_cls = get_policy_class("minivla")
    try:
        policy = policy_cls.from_pretrained(checkpoint_path)
        policy = policy.float()
        policy.to(device)
        policy.eval()
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {"error": str(e)}

    # Load pre/post processors
    config.pretrained_path = Path(checkpoint_path)
    if not hasattr(config, "type"):
        config.type = "minivla"

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        dataset_stats=None,
    )

    # Load dataset
    ds = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
    )

    num_samples = min(num_eval_samples, ds.num_frames)
    indices = np.linspace(0, ds.num_frames - 1, num_samples, dtype=int)

    # Collect metrics
    token_cross_entropies = []
    token_top1_accuracies = []
    exact_sequence_accuracies = []
    action_maes = []
    per_dim_action_maes = []
    gripper_accuracies = []

    with torch.no_grad():
        for idx in indices:
            sample = ds[idx]
            if ACTION not in sample:
                continue

            # Build batch
            batch = {}
            for key, value in sample.items():
                if key in ["observation.images.top", "observation.images.wrist", "observation.state"]:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.unsqueeze(0).float().to(device)
                    else:
                        batch[key] = torch.from_numpy(value).unsqueeze(0).float().to(device)

            batch["task"] = [sample.get("task", "")]
            real_action = sample[ACTION]
            if isinstance(real_action, torch.Tensor):
                real_action = real_action.cpu().numpy()

            # Preprocess
            try:
                preprocessed = preprocessor(batch)
            except Exception as e:
                logger.warning(f"Preprocessing failed for sample {idx}: {e}")
                continue

            # Forward pass
            try:
                loss, metrics = policy.forward(preprocessed)
                token_cross_entropies.append(loss.item())

                if metrics:
                    if "action_token_top1_accuracy" in metrics:
                        token_top1_accuracies.append(metrics["action_token_top1_accuracy"])
                    if "exact_action_token_sequence_accuracy" in metrics:
                        exact_sequence_accuracies.append(metrics["exact_action_token_sequence_accuracy"])
            except Exception as e:
                logger.warning(f"Forward pass failed for sample {idx}: {e}")
                continue

            # Predict action
            try:
                policy.reset()
                pred_action = policy.predict_action_chunk(preprocessed)

                # Postprocess
                final_action_obj = postprocessor(pred_action)
                if isinstance(final_action_obj, dict) and "action" in final_action_obj:
                    final_pred = final_action_obj["action"]
                else:
                    final_pred = final_action_obj

                if isinstance(final_pred, torch.Tensor):
                    final_pred = final_pred.cpu().numpy()

                # Compare with real action
                if final_pred.ndim == 3:
                    final_pred = final_pred[0, 0, :]
                elif final_pred.ndim == 2:
                    final_pred = final_pred[0, :]

                if real_action.ndim == 2:
                    real_action = real_action[0]

                abs_error = np.abs(final_pred - real_action)
                action_maes.append(float(abs_error.mean()))
                per_dim_action_maes.append(abs_error.tolist())

                # Gripper accuracy
                if len(final_pred) > 0 and len(real_action) > 0:
                    gripper_pred = 1.0 if final_pred[-1] > 0.5 else 0.0
                    gripper_true = 1.0 if real_action[-1] > 0.5 else 0.0
                    gripper_accuracies.append(1.0 if gripper_pred == gripper_true else 0.0)

            except Exception as e:
                logger.warning(f"Action prediction failed for sample {idx}: {e}")
                continue

    # Aggregate results
    results = {
        "checkpoint_path": checkpoint_path,
        "num_evaluated": len(token_cross_entropies),
    }

    if token_cross_entropies:
        results["token_cross_entropy"] = {
            "mean": float(np.mean(token_cross_entropies)),
            "std": float(np.std(token_cross_entropies)),
            "min": float(np.min(token_cross_entropies)),
            "max": float(np.max(token_cross_entropies)),
        }

    if token_top1_accuracies:
        results["action_token_top1_accuracy"] = {
            "mean": float(np.mean(token_top1_accuracies)),
            "std": float(np.std(token_top1_accuracies)),
        }

    if exact_sequence_accuracies:
        results["exact_action_token_sequence_accuracy"] = {
            "mean": float(np.mean(exact_sequence_accuracies)),
        }

    if action_maes:
        results["normalized_action_mae"] = {
            "mean": float(np.mean(action_maes)),
            "std": float(np.std(action_maes)),
        }

    if per_dim_action_maes:
        per_dim_array = np.array(per_dim_action_maes)
        results["per_dim_action_mae"] = per_dim_array.mean(axis=0).tolist()

    if gripper_accuracies:
        results["gripper_accuracy"] = float(np.mean(gripper_accuracies))

    # Print summary
    logger.info(f"\nCheckpoint Evaluation Summary:")
    logger.info(f"  Samples evaluated: {results.get('num_evaluated', 0)}")
    if "token_cross_entropy" in results:
        logger.info(f"  Token Cross Entropy: {results['token_cross_entropy']['mean']:.4f} ± {results['token_cross_entropy']['std']:.4f}")
    if "action_token_top1_accuracy" in results:
        logger.info(f"  Token Top-1 Accuracy: {results['action_token_top1_accuracy']['mean']:.4f}")
    if "exact_action_token_sequence_accuracy" in results:
        logger.info(f"  Exact Sequence Accuracy: {results['exact_action_token_sequence_accuracy']['mean']:.4f}")
    if "normalized_action_mae" in results:
        logger.info(f"  Normalized Action MAE: {results['normalized_action_mae']['mean']:.6f}")
    if "per_dim_action_mae" in results:
        logger.info(f"  Per-dim Action MAE: {results['per_dim_action_mae']}")
    if "gripper_accuracy" in results:
        logger.info(f"  Gripper Accuracy: {results['gripper_accuracy']:.4f}")

    return results


def check_checkpoint_weights(checkpoint_path: str, policy_config) -> dict:
    """Verify that checkpoint weights are properly loaded."""
    from safetensors.torch import load_file

    model_file = Path(checkpoint_path) / "model.safetensors"
    if not model_file.exists():
        return {"error": f"model.safetensors not found at {model_file}"}

    state_dict = load_file(str(model_file), device="cpu")

    # Check key weights
    key_layers = [
        "vlm.projector.linear_1.weight",
        "vlm.llm.model.layers.0.self_attn.q_proj.weight",
        "vlm.llm.lm_head.weight",
    ]

    weight_info = {}
    for layer_key in key_layers:
        if layer_key in state_dict:
            w = state_dict[layer_key]
            weight_info[layer_key] = {
                "shape": tuple(w.shape),
                "mean": float(w.float().mean()),
                "std": float(w.float().std()),
            }
            logger.info(f"  {layer_key}: shape={tuple(w.shape)}, mean={weight_info[layer_key]['mean']:.6f}, std={weight_info[layer_key]['std']:.6f}")
        else:
            weight_info[layer_key] = {"error": "missing"}
            logger.warning(f"  {layer_key}: MISSING")

    # Check action token embedding if applicable
    is_vq = "vq" in policy_config.action_tokenizer_type.lower()
    if is_vq:
        # VQ mode: check VQ-related weights
        vq_related = [k for k in state_dict.keys() if "vq" in k.lower() or "action" in k.lower()]
        if vq_related:
            logger.info(f"  VQ/action-related keys found: {len(vq_related)}")
        else:
            logger.warning("  No VQ/action-related keys found in checkpoint")
    else:
        # Non-VQ: check lm_head (action tokens use LLM vocab)
        if "vlm.llm.lm_head.weight" in state_dict:
            lm_head = state_dict["vlm.llm.lm_head.weight"]
            logger.info(f"  lm_head: shape={tuple(lm_head.shape)}")

    return weight_info


def diagnose_convergence(oracle_results: dict, checkpoint_results: list[dict]) -> dict:
    """Diagnose training convergence based on oracle and checkpoint results."""
    diagnosis = {
        "oracle_error": oracle_results.get("mean_mae", None),
        "checkpoints_evaluated": len(checkpoint_results),
        "diagnoses": [],
    }

    if not checkpoint_results:
        diagnosis["diagnoses"].append("No checkpoints evaluated")
        return diagnosis

    oracle_mae = oracle_results.get("mean_mae", 0.0)

    for ckpt in checkpoint_results:
        step = ckpt.get("step", "unknown")
        ckpt_diagnosis = {"step": step}

        # Check token cross entropy
        token_ce = ckpt.get("token_cross_entropy", {}).get("mean", None)
        if token_ce is not None:
            if token_ce > 5.0:
                ckpt_diagnosis["token_prediction"] = "severely_underfit"
            elif token_ce > 2.0:
                ckpt_diagnosis["token_prediction"] = "underfit"
            elif token_ce > 0.5:
                ckpt_diagnosis["token_prediction"] = "learning"
            else:
                ckpt_diagnosis["token_prediction"] = "well_trained"

        # Check token accuracy
        token_acc = ckpt.get("action_token_top1_accuracy", {}).get("mean", None)
        if token_acc is not None:
            if token_acc < 0.1:
                ckpt_diagnosis["token_accuracy"] = "very_low"
            elif token_acc < 0.5:
                ckpt_diagnosis["token_accuracy"] = "low"
            elif token_acc < 0.8:
                ckpt_diagnosis["token_accuracy"] = "moderate"
            else:
                ckpt_diagnosis["token_accuracy"] = "high"

        # Check action MAE vs oracle
        action_mae = ckpt.get("normalized_action_mae", {}).get("mean", None)
        if action_mae is not None and oracle_mae:
            ratio = action_mae / oracle_mae if oracle_mae > 0 else float("inf")
            if ratio > 10:
                ckpt_diagnosis["action_vs_oracle"] = "much_worse_than_oracle"
                ckpt_diagnosis["likely_issue"] = "visual_action_mapping_not_learned"
            elif ratio > 3:
                ckpt_diagnosis["action_vs_oracle"] = "worse_than_oracle"
                ckpt_diagnosis["likely_issue"] = "token_prediction_underfit"
            elif ratio > 1.5:
                ckpt_diagnosis["action_vs_oracle"] = "slightly_worse_than_oracle"
                ckpt_diagnosis["likely_issue"] = "approaching_oracle_limit"
            else:
                ckpt_diagnosis["action_vs_oracle"] = "near_oracle"
                ckpt_diagnosis["likely_issue"] = "converged"

        diagnosis["diagnoses"].append(ckpt_diagnosis)

    return diagnosis


def main():
    parser = argparse.ArgumentParser(description="Verify MiniVLA training progress")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset_repo_id", type=str, default="work2/disassemble-v3_corner")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_steps", type=int, nargs="*", default=None)
    parser.add_argument("--num_eval_samples", type=int, default=50)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MiniVLA Training Progress Verification")
    logger.info("=" * 80)

    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args.eval_steps)
    if not checkpoints:
        logger.error("No checkpoints found!")
        return 1

    # Load config from latest checkpoint
    latest_ckpt = checkpoints[-1]
    config = PreTrainedConfig.from_pretrained(latest_ckpt["path"])

    # Step 1: Action tokenizer oracle test
    oracle_results = run_action_tokenizer_oracle(
        args.dataset_root,
        args.dataset_repo_id,
        config,
        args.device,
    )

    # Step 2: Evaluate each checkpoint
    checkpoint_results = []
    for ckpt in checkpoints:
        results = evaluate_checkpoint(
            ckpt["path"],
            args.dataset_root,
            args.dataset_repo_id,
            args.device,
            num_eval_samples=args.num_eval_samples,
        )
        results["step"] = ckpt["step"]
        checkpoint_results.append(results)

        # Step 3: Verify checkpoint weights
        weight_info = check_checkpoint_weights(ckpt["path"], config)
        results["weight_info"] = weight_info

    # Step 4: Diagnose convergence
    diagnosis = diagnose_convergence(oracle_results, checkpoint_results)

    # Compile final results
    final_results = {
        "oracle_test": oracle_results,
        "checkpoints": checkpoint_results,
        "diagnosis": diagnosis,
    }

    # Save results
    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(args.checkpoint_dir).parent / "verification_results" / "training_progress.json")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")

    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL DIAGNOSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Oracle MAE (irreducible error): {oracle_results.get('mean_mae', 'N/A')}")
    logger.info(f"Checkpoints evaluated: {len(checkpoint_results)}")

    for d in diagnosis.get("diagnoses", []):
        step = d.get("step", "?")
        issue = d.get("likely_issue", "unknown")
        logger.info(f"  Step {step}: {issue}")

    return 0


if __name__ == "__main__":
    sys.exit(main())