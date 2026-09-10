"""
Verify TinyVLA Training Progress Across Checkpoints

Uses a fixed set of frames, fixed random seed, fixed diffusion noise, and fixed timesteps
to evaluate multiple checkpoints (12k, 20k, 30k, 40k, 50k) and determine whether
continued training produces meaningful improvement.

Metrics computed per checkpoint:
- Valid-position noise MSE (diffusion training loss proxy)
- Full denoising normalized action MAE
- Per-action-dimension MAE after postprocessor
- Direction sign accuracy
- Gripper accuracy
- Action head weight consistency (model.safetensors vs loaded weights)

Output: JSON results file + concise console log.
"""

import sys
import json
import os
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lerobot.policies import make_policy
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE
from lerobot.policies.tinyvla.configuration_tinyvla import TinyVLAConfig, TinyVLABConfig


FIXED_SEED = 42
NUM_FRAMES = 32
NUM_INFERENCE_TIMESTEPS = 10


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_checkpoints(checkpoints_dir: str, steps: list[int]) -> dict[int, Path]:
    ckpt_base = Path(checkpoints_dir)
    found = {}
    for step in steps:
        for pad in [6, 5, 4]:
            ckpt_dir = ckpt_base / f"{step:0{pad}d}"
            if ckpt_dir.is_dir() and (ckpt_dir / "pretrained_model").is_dir():
                found[step] = ckpt_dir
                break
        if step not in found:
            last_link = ckpt_base / "last"
            if last_link.is_symlink():
                target = last_link.resolve()
                if target.is_dir() and (target / "pretrained_model").is_dir():
                    step_file = target / "training_state" / "training_step.json"
                    if step_file.exists():
                        with open(step_file) as f:
                            data = json.load(f)
                        if data.get("step", 0) == step:
                            found[step] = target
    return found


def load_checkpoint_weights(ckpt_dir: Path) -> dict:
    model_file = ckpt_dir / "pretrained_model" / "model.safetensors"
    if model_file.exists():
        return load_file(str(model_file))
    return {}


def extract_action_head_weights(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if "embed_out" in k}


def compare_action_head_weights(ckpt_weights: dict, model_weights: dict) -> dict:
    ckpt_action = extract_action_head_weights(ckpt_weights)
    model_action = extract_action_head_weights(model_weights)

    results = {
        "ckpt_action_head_keys": len(ckpt_action),
        "model_action_head_keys": len(model_action),
        "matching_keys": 0,
        "mismatched_keys": 0,
        "missing_in_model": [],
        "max_abs_diff": 0.0,
    }

    common_keys = set(ckpt_action.keys()) & set(model_weights.keys())
    results["matching_keys"] = len(common_keys)

    for k in common_keys:
        diff = (ckpt_action[k].cpu() - model_weights[k].cpu()).abs().max().item()
        results["max_abs_diff"] = max(results["max_abs_diff"], diff)
        if diff > 1e-6:
            results["mismatched_keys"] += 1

    results["missing_in_model"] = list(set(ckpt_action.keys()) - set(model_weights.keys()))
    results["weights_consistent"] = (
        results["mismatched_keys"] == 0 and len(results["missing_in_model"]) == 0
    )
    return results


def get_fixed_frames(dataset, num_frames: int, seed: int = FIXED_SEED):
    set_seed(seed)
    total_frames = len(dataset)
    indices = np.random.choice(total_frames, size=min(num_frames, total_frames), replace=False)
    indices = sorted(indices.tolist())
    return [dataset[idx] for idx in indices]


def compute_noise_mse_with_fixed_noise(policy, batch, fixed_noise: torch.Tensor, fixed_timesteps: torch.Tensor):
    """Compute valid-position noise MSE using fixed noise and timesteps.

    Replicates the forward_diffusion_head forward pass but with deterministic noise.
    """
    policy.eval()
    device = policy.config.device
    base_model = _get_base_model(policy)

    actions = batch["action"].to(device)
    states = batch.get(OBS_STATE, None)
    if states is not None:
        states = states.to(device)
    is_pad = batch.get("is_pad", torch.zeros(actions.shape[0], actions.shape[1], dtype=torch.bool, device=device))

    B = actions.size(0)
    chunk_size = policy.config.chunk_size
    actions = actions[:, :chunk_size]
    is_pad = is_pad[:, :chunk_size]

    base_model._init_diffusion_head()

    noise = fixed_noise[:1].to(device, dtype=actions.dtype)
    timesteps = fixed_timesteps[:B].to(device)

    noisy_actions = base_model.noise_scheduler.add_noise(actions, noise[0], timesteps)
    noisy_actions = noisy_actions.to(dtype=actions.dtype)

    # Build hidden states through multimodal forward
    with torch.no_grad():
        images = []
        images_r = None
        images_top = None
        for k in sorted(base_model.config.image_features if hasattr(base_model.config, 'image_features') else []):
            if k in batch:
                img = batch[k].to(device)
                images.append(img)

        # Use the model's prepare_inputs_labels_for_multimodal to get hidden states
        input_ids = batch.get("input_ids", None)
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)

        if input_ids is not None:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if labels is not None:
                labels = labels.to(device)

            (
                input_ids, attention_mask, past_key_values, inputs_embeds, labels,
            ) = base_model.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values=None, labels=labels,
                images=torch.stack(images) if len(images) == 1 else None,
                images_r=images_r, images_top=images_top,
                visual_concat=getattr(base_model, 'visual_concat', 'token_cat'),
                states=states,
            )

            outputs = base_model.get_model()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            hidden_states = outputs[0]
        else:
            # Fallback: construct hidden states from images + states
            if len(images) == 1:
                image_features = base_model.encode_images(images[0])
            else:
                image_features = torch.zeros(B, 1, base_model.config.hidden_size, device=device)

            # Create dummy input_embeds from image features
            inputs_embeds = image_features
            attention_mask = torch.ones(B, inputs_embeds.shape[1], dtype=torch.long, device=device)

            outputs = base_model.get_model()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden_states = outputs[0]

    # Repeat for noise samples
    num_noise_samples = 1
    hidden_states = hidden_states.repeat(num_noise_samples, 1, 1)
    timesteps = timesteps.repeat(num_noise_samples)
    is_pad = is_pad.repeat(num_noise_samples, 1)

    if states is not None and states.ndim == 3:
        states = states[:, -1, :]
    if states is not None and states.ndim == 2:
        states = states.repeat(num_noise_samples, 1)

    noise_pred = base_model.embed_out(noisy_actions, timesteps, global_cond=hidden_states, states=states)
    noise_flat = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])

    loss = torch.nn.functional.mse_loss(noise_pred, noise_flat, reduction='none')
    valid_mask = ~is_pad.unsqueeze(-1)
    action_dim = loss.shape[-1]
    valid_count = valid_mask.sum()
    total_valid_elements = valid_count * action_dim
    denom = total_valid_elements.clamp_min(1)
    noise_mse = (loss * valid_mask).sum() / denom

    total_elements = valid_mask.numel() * action_dim if valid_mask.numel() > 0 else 1
    padding_ratio = 1.0 - (total_valid_elements.item() / max(total_elements, 1))

    return {
        "noise_mse": noise_mse.item(),
        "valid_count": total_valid_elements.item(),
        "padding_ratio": padding_ratio,
    }


def _get_base_model(policy):
    """Get the underlying LlavaPythiaForCausalLM regardless of LoRA wrapping."""
    try:
        from peft import PeftModel
        if isinstance(policy.model, PeftModel):
            return policy.model.base_model.model
    except ImportError:
        pass
    return policy.model


def compute_action_mae(policy, frames, preprocessor):
    """Compute action MAE after full denoising in normalized space."""
    policy.eval()
    device = policy.config.device

    all_mae = []
    all_dim_mae = []
    correct_sign = 0
    total_sign = 0
    correct_gripper = 0
    total_gripper = 0
    gripper_dim = -1

    with torch.no_grad():
        for frame in frames:
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in frame.items()}
            batch = preprocessor(batch)

            gt_action = batch["action"].to(device)
            predicted = policy.predict_action_chunk(batch)
            pred_action = predicted[0]

            mae_per_dim = torch.abs(pred_action - gt_action[0])
            mae = mae_per_dim.mean().item()
            all_mae.append(mae)
            all_dim_mae.append(mae_per_dim.cpu().numpy())

            gt_sign = torch.sign(gt_action[0])
            pred_sign = torch.sign(pred_action)
            sign_match = (gt_sign == pred_sign).float().mean().item()
            correct_sign += sign_match
            total_sign += 1

            if gt_action.shape[-1] > 0:
                gt_g = gt_action[0, gripper_dim].item()
                pred_g = pred_action[gripper_dim].item()
                gt_bin = 1.0 if gt_g > 0.5 else 0.0
                pred_bin = 1.0 if pred_g > 0.5 else 0.0
                if gt_bin == pred_bin:
                    correct_gripper += 1
                total_gripper += 1

    dim_mae_arr = np.mean(all_dim_mae, axis=0) if all_dim_mae else np.array([])
    return {
        "mean_mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "std_mae": float(np.std(all_mae)) if all_mae else 0.0,
        "per_dim_mae": dim_mae_arr.tolist(),
        "sign_accuracy": correct_sign / max(total_sign, 1),
        "gripper_accuracy": correct_gripper / max(total_gripper, 1),
        "num_frames": len(frames),
    }


def verify_optimizer_params(policy):
    """Verify that embed_out, proj_to_action, and LoRA params are in optimizer with requires_grad=True."""
    optim_params = policy.get_optim_params()
    optim_param_ids = set()
    for group in optim_params:
        for p in group["params"]:
            optim_param_ids.add(id(p))

    issues = []
    embed_out_found = False
    lora_found = False

    for name, param in policy.named_parameters():
        is_embed_out = "embed_out" in name
        is_lora = "lora" in name.lower()
        is_proj = "proj_to_action" in name

        if is_embed_out or is_lora or is_proj:
            if not param.requires_grad:
                issues.append(f"  FAIL: {name} requires_grad=False")
            if id(param) not in optim_param_ids:
                issues.append(f"  FAIL: {name} not in optimizer param groups")

            if is_embed_out:
                embed_out_found = True
            if is_lora:
                lora_found = True

    if not embed_out_found:
        issues.append("  FAIL: No embed_out parameters found in model")
    if not lora_found:
        issues.append("  WARNING: No LoRA parameters found (LoRA may be disabled)")

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())

    return {
        "issues": issues,
        "trainable_params": trainable,
        "total_params": total,
        "embed_out_in_optim": embed_out_found,
        "lora_in_optim": lora_found,
    }


def evaluate_checkpoint(ckpt_dir: Path, dataset, frames, step: int, policy_type: str):
    print(f"\n{'='*40}")
    print(f"Evaluating checkpoint at step {step}")
    print(f"Checkpoint dir: {ckpt_dir}")

    ckpt_weights = load_checkpoint_weights(ckpt_dir)

    config_cls = TinyVLABConfig if policy_type == "tinyvla_b" else TinyVLAConfig
    config = config_cls.from_pretrained(ckpt_dir / "pretrained_model")
    config.validate_features()

    policy = make_policy(
        cfg=config,
        ds_meta=dataset.meta,
        pretrained_path=str(ckpt_dir / "pretrained_model"),
    )
    policy.eval()
    device = policy.config.device

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(ckpt_dir / "pretrained_model"),
        dataset_stats=dataset.meta.stats,
    )

    model_weights = {k: v.cpu() for k, v in policy.state_dict().items()}
    weight_comparison = compare_action_head_weights(ckpt_weights, model_weights)

    opt_verification = verify_optimizer_params(policy)

    action_results = compute_action_mae(policy, frames, preprocessor)

    # Compute noise MSE with fixed noise
    set_seed(FIXED_SEED)
    B = min(len(frames), 4)
    action_dim = config.action_dim
    chunk_size = config.chunk_size
    fixed_noise = torch.randn(1, B, chunk_size, action_dim, device=device, dtype=torch.float32)
    fixed_timesteps = torch.randint(0, 100, (B,), device=device, dtype=torch.long)

    # Build a mini-batch from fixed frames
    mini_batch = {}
    for i in range(B):
        frame = frames[i % len(frames)]
        for k, v in frame.items():
            if isinstance(v, torch.Tensor):
                if k not in mini_batch:
                    mini_batch[k] = []
                mini_batch[k].append(v)
    for k in mini_batch:
        if isinstance(mini_batch[k], list):
            mini_batch[k] = torch.stack(mini_batch[k])

    mini_batch = preprocessor(mini_batch)
    noise_results = compute_noise_mse_with_fixed_noise(policy, mini_batch, fixed_noise, fixed_timesteps)

    results = {
        "step": step,
        "checkpoint_dir": str(ckpt_dir),
        "trainable_params": opt_verification["trainable_params"],
        "total_params": opt_verification["total_params"],
        "weight_comparison": weight_comparison,
        "optimizer_verification": {
            "issues": opt_verification["issues"],
            "embed_out_in_optim": opt_verification["embed_out_in_optim"],
            "lora_in_optim": opt_verification["lora_in_optim"],
        },
        "noise_mse": noise_results["noise_mse"],
        "valid_action_count": noise_results["valid_count"],
        "padding_ratio": noise_results["padding_ratio"],
        "action_metrics": action_results,
    }

    print(f"  Trainable params: {opt_verification['trainable_params']:,} / {opt_verification['total_params']:,}")
    print(f"  Optimizer issues: {len(opt_verification['issues'])}")
    for issue in opt_verification["issues"]:
        print(f"    {issue}")
    print(f"  Weight consistency: {weight_comparison['weights_consistent']}")
    print(f"  Noise MSE (valid): {noise_results['noise_mse']:.6f}")
    print(f"  Padding ratio: {noise_results['padding_ratio']:.4f}")
    print(f"  Mean MAE: {action_results['mean_mae']:.4f} +/- {action_results['std_mae']:.4f}")
    print(f"  Sign accuracy: {action_results['sign_accuracy']:.4f}")
    print(f"  Gripper accuracy: {action_results['gripper_accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify TinyVLA training progress across checkpoints")
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="1/2")
    parser.add_argument("--steps", type=int, nargs="+", default=[12000, 20000, 30000, 40000, 50000])
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--policy_type", type=str, default="tinyvla_s")
    args = parser.parse_args()

    set_seed(args.seed)

    checkpoints = find_checkpoints(args.checkpoints_dir, args.steps)
    print(f"Found {len(checkpoints)} / {len(args.steps)} checkpoints: {list(checkpoints.keys())}")
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    print(f"\nLoading dataset from {args.dataset_root}")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    print(f"\nSelecting {args.num_frames} fixed frames with seed {args.seed}")
    frames = get_fixed_frames(dataset, args.num_frames, args.seed)

    all_results = []
    for step in sorted(checkpoints.keys()):
        ckpt_dir = checkpoints[step]
        result = evaluate_checkpoint(ckpt_dir, dataset, frames, step, args.policy_type)
        all_results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Step':<8} {'NoiseMSE':<12} {'MAE':<10} {'SignAcc':<10} {'GripAcc':<10} {'PadRatio':<10} {'WeightsOK'}")
    for r in all_results:
        print(
            f"{r['step']:<8} {r['noise_mse']:<12.6f} "
            f"{r['action_metrics']['mean_mae']:<10.4f} "
            f"{r['action_metrics']['sign_accuracy']:<10.4f} "
            f"{r['action_metrics']['gripper_accuracy']:<10.4f} "
            f"{r['padding_ratio']:<10.4f} "
            f"{r['weight_comparison']['weights_consistent']}"
        )

    output_path = args.output
    if output_path is None:
        output_path = Path(args.checkpoints_dir).parent / "tinyvla_training_progress.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()