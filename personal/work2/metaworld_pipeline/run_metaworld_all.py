#!/usr/bin/env python3
"""One-command MetaWorld three-task selection + joint VLA training pipeline.

The three fixed datasets are:
  coffee-button-v3_corner
  disassemble-v3_corner
  pick_place-v3_corner

For each dataset, the requested selection method independently selects N episodes.
The three subsets are then physically merged into one LeRobot dataset and a single
VLA is trained jointly on all 3N demonstrations.

Supported selectors:
  our_v6, random, grid_uniform, deminf, fps
Supported models:
  smovla/smolvla, minivla, tinyvla_s, tinyvla_b
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
DATASET_BASE_DEFAULT = WORK2_ROOT / "dataset_view"
MERGER = WORK2_ROOT / "duibi" / "train_and_eval_scripts" / "merge_selected_episodes.py"
SELECT_RANDOM = WORK2_ROOT / "duibi" / "train_and_eval_scripts" / "select_random_episodes.py"
SELECT_GRID = WORK2_ROOT / "duibi" / "train_and_eval_scripts" / "select_grid_uniform.py"
SELECT_FPS = WORK2_ROOT / "duibi" / "fps" / "select_visual_fps.py"
SELECT_OUR_V6 = WORK2_ROOT / "our_v6" / "experiments" / "select_episodes_v6.py"
DEMINF = WORK2_ROOT / "deminf" / "run_deminf.py"
ENSURE_VISUAL = WORK2_ROOT / "embedding_utils" / "ensure_embeddings.py"

if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

DATASETS = [
    {"dataset": "coffee-button-v3_corner", "task": "coffee-button-v3"},
    {"dataset": "disassemble-v3_corner", "task": "disassemble-v3"},
    {"dataset": "pick_place-v3_corner", "task": "pick-place-v3"},
]

METHOD_ALIASES = {
    "ours": "our_v6",
    "ours_v6": "our_v6",
    "v6": "our_v6",
    "grid": "grid_uniform",
    "grid-uniform": "grid_uniform",
    "gird_uniform": "grid_uniform",  # tolerate common typo
    "gird-uniform": "grid_uniform",
    "visual_fps": "fps",
}
MODEL_ALIASES = {
    "smovla": "smolvla",  # compatibility spelling used in project scripts
    "smolvla": "smolvla",
    "minivla": "minivla",
    "tinyvla": "tinyvla_s",
    "tinyvla-s": "tinyvla_s",
    "tinyvla_s": "tinyvla_s",
    "tinyvla-b": "tinyvla_b",
    "tinyvla_b": "tinyvla_b",
}


def run(cmd: list[str], env: dict | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def normalize_method(name: str) -> str:
    key = name.strip().lower()
    key = METHOD_ALIASES.get(key, key)
    supported = {"our_v6", "random", "grid_uniform", "deminf", "fps"}
    if key not in supported:
        raise ValueError(f"Unsupported selection method '{name}'. Supported: {sorted(supported)}")
    return key


def normalize_model(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unsupported model '{name}'. Supported: smovla/smolvla, minivla, tinyvla_s, tinyvla_b"
        )
    return MODEL_ALIASES[key]


def _load_selected(path: Path, expected: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("selected_episode_indices")
    if ids is None:
        ids = payload.get("selected_episodes")
    if ids is None:
        raise KeyError(f"No selected episode list in {path}")
    ids = [int(x) for x in ids]
    if len(ids) != expected:
        raise RuntimeError(f"{path}: selected {len(ids)}, expected {expected}")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{path}: duplicate selected episode indices")
    return ids


def _copy_selection(generated: Path, output_file: Path, expected: int) -> None:
    if not generated.exists():
        raise FileNotFoundError(f"Selector did not generate expected file: {generated}")
    _load_selected(generated, expected)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if generated.resolve() != output_file.resolve():
        shutil.copy2(generated, output_file)


def select_one(
    method: str,
    dataset_root: Path,
    dataset_name: str,
    output_file: Path,
    work_dir: Path,
    n: int,
    seed: int,
    gpu_id: int,
    visual_variant: str,
    our_v6_ablation: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        try:
            _load_selected(output_file, n)
            print(f"[selection cache] {dataset_name}: {output_file}")
            return
        except Exception as exc:
            print(f"[selection cache invalid] {output_file}: {exc}")
            output_file.unlink()

    if method == "random":
        run([
            sys.executable, str(SELECT_RANDOM),
            "--num-episodes", str(n),
            "--seed", str(seed),
            "--dataset-root", str(dataset_root),
            "--output-dir", str(work_dir),
        ])
        _copy_selection(work_dir / f"random_{n}_seed{seed}.json", output_file, n)
        return

    if method == "grid_uniform":
        run([
            sys.executable, str(SELECT_GRID),
            "--num-episodes", str(n),
            "--seed", str(seed),
            "--dataset-root", str(dataset_root),
            "--output-dir", str(work_dir),
        ])
        _copy_selection(work_dir / f"grid_uniform_{n}_seed{seed}.json", output_file, n)
        return

    if method == "our_v6":
        run([
            sys.executable, str(SELECT_OUR_V6),
            "--dataset-root", str(dataset_root),
            "--dataset-name", dataset_name,
            "--output-dir", str(work_dir),
            "--total-budget", str(n),
            "--visual-variant", visual_variant,
            "--device", "cuda",
            "--seed", str(seed),
            "--ablation", our_v6_ablation,
        ])
        _copy_selection(work_dir / "selected_episodes_v6.json", output_file, n)
        return

    if method == "fps":
        embedding_path_file = work_dir / "visual_embedding_path.txt"
        run([
            sys.executable, str(ENSURE_VISUAL),
            "--dataset-root", str(dataset_root),
            "--dataset-name", dataset_name,
            "--gpu-id", str(gpu_id),
            "--pca-dim", "32",
            "--path-file", str(embedding_path_file),
        ])
        visual_dir = embedding_path_file.read_text(encoding="utf-8").strip()
        if not visual_dir:
            raise RuntimeError(f"Empty embedding path file: {embedding_path_file}")
        run([
            sys.executable, str(SELECT_FPS),
            "--dataset-dir", str(dataset_root),
            "--dataset-name", dataset_name,
            "--visual-embedding-dir", visual_dir,
            "--output-dir", str(work_dir),
            "--num-selected", str(n),
            "--seed", str(seed),
            "--pca-dim", "32",
        ])
        _copy_selection(work_dir / "subsets" / f"fps_{n}_seed{seed}.json", output_file, n)
        return

    if method == "deminf":
        run([
            sys.executable, str(DEMINF),
            "--dataset-path", str(dataset_root),
            "--output-dir", str(work_dir),
            "--target-episodes", str(n),
            "--seed", str(seed),
            "--device", "cuda",
            "--vae-steps", "50000",
            "--vae-lr", "1e-4",
            "--vae-batch-size", "256",
            "--state-latent-dim", "12",
            "--action-latent-dim", "6",
            "--ks", "5", "6", "7",
            "--state-source", "observation.environment_state",
            "--quality-batch-size", "1024",
            "--quality-repeat", "4",
        ])
        generated = work_dir / "subsets" / f"deminf_{n}_seed{seed}.json"
        if not generated.exists():
            candidates: list[Path] = []
            for path in work_dir.rglob("*.json"):
                try:
                    _load_selected(path, n)
                    candidates.append(path)
                except Exception:
                    pass
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Cannot uniquely locate DemInf subset under {work_dir}; candidates={candidates}"
                )
            generated = candidates[0]
        _copy_selection(generated, output_file, n)
        return

    raise AssertionError(method)


def build_manifest(
    selections: dict[str, list[int]],
    method: str,
    n: int,
    seed: int,
    visual_variant: str,
    our_v6_ablation: str,
    output_file: Path,
) -> Path:
    dataset_names = [item["dataset"] for item in DATASETS]
    flattened: list[int] = []
    for ds in dataset_names:
        ids = selections[ds]
        if len(ids) != n:
            raise RuntimeError(f"{ds}: selected {len(ids)}, expected {n}")
        flattened.extend(ids)

    payload = {
        "method": method,
        "datasets": dataset_names,
        "episodes_per_dataset": n,
        "total_selected": len(flattened),
        "seed": seed,
        "selected_episode_indices": flattened,
        "selected_by_dataset": selections,
    }
    if method == "our_v6":
        payload["visual_variant"] = visual_variant
        payload["ablation"] = our_v6_ablation

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Selection manifest: {output_file}")
    print(f"Total selected: {len(flattened)} ({n} x {len(dataset_names)})")
    return output_file


def merge_selected(manifest: Path, dataset_base: Path, merged_root: Path, force: bool) -> None:
    if force and merged_root.exists():
        shutil.rmtree(merged_root)
    if merged_root.exists() and (merged_root / "meta" / "info.json").exists():
        print(f"[merge cache] using existing merged dataset: {merged_root}")
        return

    run([
        sys.executable, str(MERGER),
        "--subset-file", str(manifest),
        "--output-dir", str(merged_root),
        "--dataset-base-dir", str(dataset_base),
    ])
    if not (merged_root / "meta" / "info.json").exists():
        raise RuntimeError(f"Merged LeRobot dataset was not created correctly: {merged_root}")


def common_train_args(
    merged_root: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    num_workers: int,
) -> list[str]:
    return [
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=lerobot/metaworld_three_tasks_selected",
        f"--dataset.root={merged_root}",
        "--dataset.eval_split=0.0",
        "--env.type=metaworld",
        "--env.task=disassemble-v3",
        "--env.camera_name=corner,gripperPOV",
        "--env_eval_freq=0",
        f"--steps={steps}",
        "--save_freq=2000",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
        "--eval.n_episodes=10",
        "--eval.batch_size=4",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
        '--remove_features=["observation.environment_state"]',
        "--wandb.enable=true",
    ]


def train_smolvla(
    merged_root: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    num_workers: int,
    job_name: str,
) -> None:
    cmd = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--rename_map={\"observation.images.top\":\"observation.images.camera1\",\"observation.images.wrist\":\"observation.images.camera2\"}",
        "--policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=false",
        "--policy.optimizer_lr=1e-4",
        *common_train_args(merged_root, output_dir, seed, steps, batch_size, num_workers),
        f"--job_name={job_name}",
    ]
    run(cmd)


def _resolve_minivla_checkpoint(pretrained_checkpoint: str) -> str:
    if pretrained_checkpoint.endswith(".pt"):
        path = Path(pretrained_checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path.resolve())

    from differentvlm.minivla.train.train_minivla import _resolve_hf_checkpoint_path

    resolved = _resolve_hf_checkpoint_path(pretrained_checkpoint)
    if not resolved:
        raise FileNotFoundError(
            f"MiniVLA pretrained checkpoint '{pretrained_checkpoint}' is not available in the HuggingFace cache"
        )
    return resolved


def train_minivla(
    merged_root: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    num_workers: int,
    job_name: str,
    init_mode: str,
    pretrained_checkpoint: str,
) -> None:
    cmd = [
        "lerobot-train",
        "--policy.type=minivla_wrist",
        "--policy.use_amp=true",
        "--policy.primary_image_key=observation.images.top",
        "--policy.wrist_image_key=observation.images.wrist",
        "--policy.action_tokenizer_type=extra_action_tokenizer",
        "--policy.vq_model_path=Stanford-ILIAD/pretrain_vq",
        "--policy.optimizer_lr=2e-5",
        "--policy.optimizer_weight_decay=0.0",
        "--policy.optimizer_grad_clip_norm=1.0",
        "--policy.scheduler_type=constant",
        "--policy.scheduler_warmup_ratio=0.0",
    ]
    if init_mode == "backbone_only":
        resolved = _resolve_minivla_checkpoint(pretrained_checkpoint)
        cmd.extend([
            "--policy.official_init_mode=backbone_only",
            f"--policy.official_pretrained_checkpoint={resolved}",
        ])

    cmd.extend(common_train_args(merged_root, output_dir, seed, steps, batch_size, num_workers))
    cmd.append(f"--job_name={job_name}")
    run(cmd)


def train_tinyvla(
    policy_type: str,
    merged_root: Path,
    output_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    num_workers: int,
    job_name: str,
) -> None:
    warmup_steps = max(1, int(round(steps * 0.005)))
    cmd = [
        "lerobot-train",
        f"--policy.type={policy_type}",
        "--policy.optimizer_lr=2e-4",
        "--policy.optimizer_weight_decay=0",
        f"--policy.scheduler_warmup_steps={warmup_steps}",
        f"--policy.scheduler_decay_steps={steps}",
        "--policy.scheduler_decay_lr=2.5e-6",
        "--env.use_self_mw=true",
        *common_train_args(merged_root, output_dir, seed, steps, batch_size, num_workers),
        f"--job_name={job_name}",
    ]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select N episodes from each of three MetaWorld datasets and jointly train a VLA"
    )
    parser.add_argument(
        "--method",
        required=True,
        help="our_v6 | random | grid_uniform | deminf | fps (grid-uniform/gird_uniform aliases supported)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="smovla/smolvla | minivla | tinyvla_s | tinyvla_b",
    )
    parser.add_argument("--episodes-per-task", type=int, default=112)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-variant", choices=["l2", "online_pca"], default="l2")
    parser.add_argument(
        "--our-v6-ablation",
        choices=["full", "wo_action", "wo_adaptive_priority"],
        default="full",
    )
    parser.add_argument("--dataset-base", default=str(DATASET_BASE_DEFAULT))
    parser.add_argument("--output-root", default=str(WORK2_ROOT / "metaworld_runs"))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--minivla-init", choices=["backbone_only", "none"], default="backbone_only")
    parser.add_argument("--minivla-pretrained", default="Stanford-ILIAD/minivla-libero90-prismatic")
    args = parser.parse_args()

    method = normalize_method(args.method)
    model = normalize_model(args.model)
    n = int(args.episodes_per_task)
    if n < 1:
        raise ValueError("--episodes-per-task must be >= 1")

    dataset_base = Path(args.dataset_base).resolve()
    output_root = Path(args.output_root).resolve()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    exp_name = f"metaworld_{method}_{model}_{n}x3_seed{args.seed}"
    if method == "our_v6":
        exp_name += f"_{args.visual_variant}_{args.our_v6_ablation}"
    exp_root = output_root / exp_name
    selection_root = exp_root / "selection"
    exp_root.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("MetaWorld three-task selection + joint training pipeline")
    print(f"method:             {method}")
    print(f"model:              {model} (input={args.model})")
    print(f"episodes/task:      {n}")
    print(f"total train eps:    {n * len(DATASETS)}")
    print(f"gpu:                {args.gpu_id}")
    print(f"dataset base:       {dataset_base}")
    print(f"output:             {exp_root}")
    if method == "our_v6":
        print(f"visual variant:     {args.visual_variant}")
        print(f"our_v6 ablation:    {args.our_v6_ablation}")
    print("=" * 76)

    selections: dict[str, list[int]] = {}
    for spec in DATASETS:
        dataset_name = spec["dataset"]
        dataset_root = dataset_base / dataset_name
        if not (dataset_root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"MetaWorld LeRobot dataset not found: {dataset_root}")
        if method in {"grid_uniform", "our_v6"} and not (dataset_root / "episode_initial_states.json").exists():
            raise FileNotFoundError(
                f"{method} requires rand_vec metadata: {dataset_root / 'episode_initial_states.json'}"
            )

        output_file = selection_root / dataset_name / f"{method}_{n}_seed{args.seed}.json"
        work_dir = selection_root / dataset_name / "work"
        print(f"\n=== Selecting {dataset_name} with {method} ===")
        select_one(
            method=method,
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            output_file=output_file,
            work_dir=work_dir,
            n=n,
            seed=args.seed,
            gpu_id=args.gpu_id,
            visual_variant=args.visual_variant,
            our_v6_ablation=args.our_v6_ablation,
        )
        selections[dataset_name] = _load_selected(output_file, n)
        print(f"{dataset_name}: selected {len(selections[dataset_name])} episodes")

    manifest = build_manifest(
        selections=selections,
        method=method,
        n=n,
        seed=args.seed,
        visual_variant=args.visual_variant,
        our_v6_ablation=args.our_v6_ablation,
        output_file=exp_root / "subsets" / f"{method}_{n}x3_seed{args.seed}.json",
    )

    if args.selection_only:
        print("Selection-only complete.")
        return

    merged_root = exp_root / "merged_dataset"
    merge_selected(manifest, dataset_base, merged_root, args.force_merge)

    if model == "smolvla":
        steps = args.train_steps or 12000
        batch_size = args.batch_size or 64
        num_workers = args.num_workers or 16
        train_smolvla(
            merged_root, exp_root / "train_smolvla", args.seed,
            steps, batch_size, num_workers, exp_name,
        )
        training_output = exp_root / "train_smolvla"
    elif model == "minivla":
        steps = args.train_steps or 50000
        batch_size = args.batch_size or 2
        num_workers = args.num_workers or 16
        train_minivla(
            merged_root, exp_root / "train_minivla", args.seed,
            steps, batch_size, num_workers, exp_name,
            args.minivla_init, args.minivla_pretrained,
        )
        training_output = exp_root / "train_minivla"
    else:
        steps = args.train_steps or 50000
        batch_size = args.batch_size or 4
        num_workers = args.num_workers or 4
        train_tinyvla(
            model, merged_root, exp_root / f"train_{model}", args.seed,
            steps, batch_size, num_workers, exp_name,
        )
        training_output = exp_root / f"train_{model}"

    summary = {
        "experiment": exp_name,
        "method": method,
        "model": model,
        "input_model_name": args.model,
        "episodes_per_task": n,
        "total_episodes": n * len(DATASETS),
        "seed": args.seed,
        "datasets": [item["dataset"] for item in DATASETS],
        "manifest": str(manifest),
        "merged_dataset": str(merged_root),
        "training_output": str(training_output),
        "train_steps": steps,
        "batch_size": batch_size,
    }
    (exp_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nMetaWorld training pipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
