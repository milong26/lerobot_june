#!/usr/bin/env python3
"""One-command RoboMME three-task selection + joint VLA training pipeline.

The three fixed benchmark datasets are:
  MoveCube_easy, PatternLock_medium, RouteStick_hard.

Each task is selected independently with the same requested budget/method, then
all selected subsets are physically merged into one LeRobot dataset. The model
therefore trains on exactly N x 3 episodes rather than mixing episode indices
from unrelated dataset roots.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK2_ROOT = REPO_ROOT / "personal" / "work2"
PIPELINE_ROOT = WORK2_ROOT / "robomme_pipeline"
SELECTOR = PIPELINE_ROOT / "select_robomme_episodes.py"
MERGER = WORK2_ROOT / "duibi" / "train_and_eval_scripts" / "merge_selected_episodes.py"
DEMINF = WORK2_ROOT / "deminf" / "run_deminf.py"

TASKS = [
    {"dataset": "MoveCube_easy", "task": "MoveCube", "difficulty": "easy"},
    {"dataset": "PatternLock_medium", "task": "PatternLock", "difficulty": "medium"},
    {"dataset": "RouteStick_hard", "task": "RouteStick", "difficulty": "hard"},
]
METHOD_ALIASES = {
    "grid-uniform": "grid_uniform",
    "grid": "grid_uniform",
    "ours": "our_v6",
    "ours_v6": "our_v6",
    "v6": "our_v6",
}
MODEL_ALIASES = {
    "smovla": "smolvla",  # user-facing compatibility spelling
    "smolvla": "smolvla",
    "minivla": "minivla",
}


def run(cmd: list[str], env: dict | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def normalize_method(name: str) -> str:
    key = name.strip().lower()
    return METHOD_ALIASES.get(key, key)


def normalize_model(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError("model must be one of: smovla, smolvla, minivla")
    return MODEL_ALIASES[key]


def select_one(
    method: str,
    dataset_root: Path,
    dataset_name: str,
    output_file: Path,
    method_output_dir: Path,
    n: int,
    seed: int,
    device: str,
    visual_variant: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        data = json.loads(output_file.read_text(encoding="utf-8"))
        selected = data.get("selected_episode_indices", [])
        if len(selected) == n and len(selected) == len(set(selected)):
            print(f"[selection cache] {dataset_name}: {output_file}")
            return
        print(f"[selection cache invalid] regenerating {output_file}")
        output_file.unlink()

    if method == "deminf":
        method_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(DEMINF),
            "--dataset-path", str(dataset_root),
            "--output-dir", str(method_output_dir),
            "--target-episodes", str(n),
            "--seed", str(seed),
            "--device", device,
            "--vae-steps", "50000",
            "--vae-lr", "1e-4",
            "--vae-batch-size", "256",
            "--state-latent-dim", "12",
            "--action-latent-dim", "6",
            "--ks", "5", "6", "7",
            "--state-source", "observation.state",
            "--quality-batch-size", "1024",
            "--quality-repeat", "4",
        ]
        run(cmd)
        generated = method_output_dir / "subsets" / f"deminf_{n}_seed{seed}.json"
        if not generated.exists():
            matches = list(method_output_dir.rglob("*.json"))
            valid = []
            for path in matches:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if len(payload.get("selected_episode_indices", [])) == n:
                    valid.append(path)
            if len(valid) != 1:
                raise FileNotFoundError(
                    f"Cannot uniquely locate DemInf subset under {method_output_dir}; candidates={valid}"
                )
            generated = valid[0]
        shutil.copy2(generated, output_file)
        return

    supported = {"random", "grid_uniform", "fps", "our_v6"}
    if method not in supported:
        raise ValueError(
            f"Unsupported RoboMME method '{method}'. Supported: "
            + ", ".join(sorted(supported | {"deminf"}))
        )

    cmd = [
        sys.executable,
        str(SELECTOR),
        "--dataset-root", str(dataset_root),
        "--dataset-name", dataset_name,
        "--method", method,
        "--num-episodes", str(n),
        "--seed", str(seed),
        "--device", device,
        "--visual-variant", visual_variant,
        "--output-file", str(output_file),
    ]
    run(cmd)


def build_manifest(
    selections: dict[str, list[int]],
    method: str,
    n: int,
    seed: int,
    output_file: Path,
) -> Path:
    datasets = [x["dataset"] for x in TASKS]
    flattened: list[int] = []
    for ds in datasets:
        ids = selections[ds]
        if len(ids) != n:
            raise RuntimeError(f"{ds}: expected {n} selected episodes, got {len(ids)}")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{ds}: duplicate selected episode indices")
        flattened.extend(ids)

    payload = {
        "method": method,
        "datasets": datasets,
        "episodes_per_dataset": n,
        "seed": seed,
        "selected_episode_indices": flattened,
        "selected_by_dataset": selections,
        "total_selected": len(flattened),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Selection manifest: {output_file}")
    print(f"Total selected: {len(flattened)} ({n} x {len(datasets)})")
    return output_file


def merge_selected(manifest: Path, dataset_base: Path, merged_root: Path, force: bool) -> None:
    if force and merged_root.exists():
        shutil.rmtree(merged_root)
    if merged_root.exists() and (merged_root / "meta" / "info.json").exists():
        print(f"[merge cache] using existing merged dataset: {merged_root}")
        return
    run([
        sys.executable,
        str(MERGER),
        "--subset-file", str(manifest),
        "--output-dir", str(merged_root),
        "--dataset-base-dir", str(dataset_base),
    ])
    if not (merged_root / "meta" / "info.json").exists():
        raise RuntimeError(f"Merged LeRobot dataset was not created correctly: {merged_root}")


def common_train_args(merged_root: Path, output_dir: Path, seed: int, steps: int, batch_size: int) -> list[str]:
    return [
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=work2/robomme_three_tasks_selected",
        f"--dataset.root={merged_root}",
        "--dataset.eval_split=0.0",
        "--env.type=robomme",
        "--env.task=MoveCube",
        "--env.action_space=joint_angle",
        "--env.dataset_split=test",
        "--env_eval_freq=0",
        f"--steps={steps}",
        "--save_freq=2000",
        f"--batch_size={batch_size}",
        "--num_workers=16",
        "--eval.n_episodes=10",
        "--eval.batch_size=4",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
        "--wandb.enable=true",
    ]


def train_smolvla(
    merged_root: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    experiment_name: str,
) -> None:
    cmd = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=false",
        "--policy.optimizer_lr=1e-4",
        *common_train_args(merged_root, output_dir, seed, steps, batch_size),
        f"--job_name={experiment_name}",
    ]
    run(cmd)


def train_minivla(
    merged_root: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    experiment_name: str,
    init_mode: str,
    pretrained_checkpoint: str,
) -> None:
    cmd = [
        "lerobot-train",
        "--policy.type=minivla_wrist",
        "--policy.use_amp=true",
        "--policy.primary_image_key=observation.images.image",
        "--policy.wrist_image_key=observation.images.wrist_image",
        "--policy.action_tokenizer_type=extra_action_tokenizer",
        "--policy.vq_model_path=Stanford-ILIAD/pretrain_vq",
        "--policy.optimizer_lr=2e-5",
        "--policy.optimizer_weight_decay=0.0",
        "--policy.optimizer_grad_clip_norm=1.0",
        "--policy.scheduler_type=constant",
        "--policy.scheduler_warmup_ratio=0.0",
    ]

    if init_mode == "backbone_only":
        resolved = pretrained_checkpoint
        if not resolved.endswith(".pt"):
            from differentvlm.minivla.train.train_minivla import _resolve_hf_checkpoint_path

            maybe = _resolve_hf_checkpoint_path(resolved)
            if not maybe:
                raise FileNotFoundError(
                    f"MiniVLA pretrained checkpoint '{pretrained_checkpoint}' is not available in the HF cache. "
                    "Download it first or pass --minivla-init none."
                )
            resolved = maybe
        cmd.extend([
            "--policy.official_init_mode=backbone_only",
            f"--policy.official_pretrained_checkpoint={resolved}",
        ])

    cmd.extend(common_train_args(merged_root, output_dir, seed, steps, batch_size))
    cmd.append(f"--job_name={experiment_name}")
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select 100/task from all three RoboMME datasets and jointly train a VLA"
    )
    parser.add_argument("--method", required=True,
                        help="random | grid_uniform | fps | deminf | our_v6 (aliases: grid-uniform, ours, v6)")
    parser.add_argument("--model", required=True, help="smovla | smolvla | minivla")
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-variant", choices=["l2", "online_pca"], default="l2",
                        help="Used by our_v6 only")
    parser.add_argument("--dataset-base", default=str(WORK2_ROOT / "dataset_view_robomme"))
    parser.add_argument("--output-root", default=str(WORK2_ROOT / "robomme_runs"))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0,
                        help="0 uses model default: SmolVLA=12000, MiniVLA=50000")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 uses model default: SmolVLA=64, MiniVLA=2")
    parser.add_argument("--minivla-init", choices=["backbone_only", "none"], default="backbone_only")
    parser.add_argument("--minivla-pretrained", default="Stanford-ILIAD/minivla-libero90-prismatic")
    args = parser.parse_args()

    method = normalize_method(args.method)
    model = normalize_model(args.model)
    dataset_base = Path(args.dataset_base).resolve()
    output_root = Path(args.output_root).resolve()
    n = int(args.episodes_per_task)
    device = "cuda"

    if n < 1:
        raise ValueError("--episodes-per-task must be >= 1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("SAPIEN_VULKAN_DEVICE_INDEX", str(args.gpu_id))

    exp_name = f"robomme_{method}_{model}_{n}x3_seed{args.seed}"
    if method == "our_v6":
        exp_name += f"_{args.visual_variant}"
    exp_root = output_root / exp_name
    selection_root = exp_root / "selection"
    exp_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RoboMME three-task pipeline")
    print(f"method:            {method}")
    print(f"model:             {model} (input={args.model})")
    print(f"episodes/task:     {n}")
    print(f"total train eps:   {n * len(TASKS)}")
    print(f"gpu:               {args.gpu_id}")
    print(f"dataset base:      {dataset_base}")
    print(f"output:            {exp_root}")
    print("=" * 72)

    selections: dict[str, list[int]] = {}
    for spec in TASKS:
        ds = spec["dataset"]
        root = dataset_base / ds
        if not (root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"RoboMME LeRobot dataset not found: {root}")
        output_file = selection_root / ds / f"{method}_{n}_seed{args.seed}.json"
        select_one(
            method=method,
            dataset_root=root,
            dataset_name=ds,
            output_file=output_file,
            method_output_dir=selection_root / ds / "deminf_work",
            n=n,
            seed=args.seed,
            device=device,
            visual_variant=args.visual_variant,
        )
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        selected = [int(x) for x in payload["selected_episode_indices"]]
        if len(selected) != n:
            raise RuntimeError(f"{ds}: selected {len(selected)}, requested {n}")
        selections[ds] = selected
        print(f"{ds}: selected {len(selected)} episodes")

    manifest = build_manifest(
        selections,
        method,
        n,
        args.seed,
        exp_root / "subsets" / f"{method}_{n}x3_seed{args.seed}.json",
    )

    if args.selection_only:
        print("Selection-only complete.")
        return

    merged_root = exp_root / "merged_dataset"
    merge_selected(manifest, dataset_base, merged_root, args.force_merge)

    if model == "smolvla":
        steps = args.train_steps or 12000
        batch = args.batch_size or 64
        train_smolvla(
            merged_root,
            exp_root / "train_smolvla",
            args.seed,
            steps,
            batch,
            exp_name,
        )
    else:
        steps = args.train_steps or 50000
        batch = args.batch_size or 2
        train_minivla(
            merged_root,
            exp_root / "train_minivla",
            args.seed,
            steps,
            batch,
            exp_name,
            args.minivla_init,
            args.minivla_pretrained,
        )

    summary = {
        "experiment": exp_name,
        "method": method,
        "model": model,
        "input_model_name": args.model,
        "episodes_per_task": n,
        "total_episodes": n * len(TASKS),
        "seed": args.seed,
        "datasets": [x["dataset"] for x in TASKS],
        "manifest": str(manifest),
        "merged_dataset": str(merged_root),
        "training_output": str(exp_root / ("train_smolvla" if model == "smolvla" else "train_minivla")),
    }
    (exp_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRoboMME training pipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
