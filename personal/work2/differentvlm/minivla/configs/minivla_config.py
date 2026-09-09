"""
MiniVLA Experiment Configuration

Manages all experiment configurations for MiniVLA experiments.
Mirrors the duibi/grid_uniform experiment flow but uses MiniVLA policy instead of SmolVLA.

Key design:
- For our_v5 mode: auto-extracts VLM embeddings if not cached
- Direct episode selection via various algorithms (grid_uniform, random, our_v5, deminf)
- Training uses lerobot-train with policy.type=minivla
- Evaluation uses lerobot-eval with MiniVLA checkpoint
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
WORK2_ROOT = Path(__file__).resolve().parents[3]

# Dataset configurations: dataset_name -> (dataset_root, env_task, camera)
DATASET_CONFIGS = {
    "pick_place-v3_corner": {
        "dataset_root": "personal/work2/dataset_view/pick_place_corner",
        "env_task": "pick-place-v3",
        "camera": "corner",
        "camera_names": "corner,gripperPOV",
        "primary_image_key": "observation.images.corner",
        "wrist_image_key": "observation.images.gripperPOV",
        "eval_camera": "corner",
        "eval_camera_names": "corner,gripperPOV",
    },
    "disassemble-v3_corner": {
        "dataset_root": "personal/work2/dataset_view/disassemble-v3_corner",
        "env_task": "disassemble-v3",
        "camera": "corner",
        "camera_names": "top,wrist",
        "primary_image_key": "observation.images.top",
        "wrist_image_key": "observation.images.wrist",
        "eval_camera": "corner",
        "eval_camera_names": "top,wrist",
    },
}

SELECTION_NUM_EPISODES = 112
SELECTION_SEED = 42

TRAIN_STEPS = 20000
TRAIN_SAVE_FREQ = 2000
TRAIN_BATCH_SIZE = 4
TRAIN_NUM_WORKERS = 16
TRAIN_LR = 2e-5

EVAL_N_EPISODES = 200
EVAL_SEED = 42
EVAL_BATCH_SIZE = 16


VQ_MODEL_PATH = "Stanford-ILIAD/pretrain_vq"

RENAME_MAP = '{}'


@dataclass
class MiniVLAExperimentConfig:
    """
    Configuration for a single MiniVLA experiment.

    Directory structure: differentvlm/minivla/experiments/{exp_name}/
    ├── subsets/             # Episode selection results
    ├── checkpoints/         # Training checkpoints
    ├── logs/                # Training and eval logs
    ├── eval_results/        # Evaluation results
    └── wandb/               # Wandb logs

    Fields:
        exp_name: experiment name (e.g. "grid_uniform_112_seed42")
        dataset_name: short name for dataset (e.g. "disassemble-v3_corner")
        dataset_root: path to the dataset directory
        env_task: environment task name (e.g. "disassemble-v3")
        camera: camera configuration name (e.g. "corner")
        camera_names: camera names for training/eval
        primary_image_key: primary camera observation key
        wrist_image_key: wrist camera observation key
        gpu_id: GPU device id
        selection_num_episodes: number of episodes to select
        selection_seed: random seed for selection
        selection_mode: episode selection mode (grid_uniform, random, etc.)
        train_steps: training steps
        train_save_freq: checkpoint save frequency
        train_batch_size: training batch size
        train_num_workers: training num workers
        train_lr: training learning rate
        eval_n_episodes: number of evaluation episodes
        eval_seed: evaluation seed
        eval_batch_size: evaluation batch size
        vq_model_path: path to pre-trained VQ-VAE model
        rename_map: feature rename map for training
        experiment_dir: base directory for all experiment outputs
        subsets_dir: output directory for episode subsets
        checkpoints_dir: output directory for training checkpoints
        logs_dir: output directory for logs
        eval_results_dir: output directory for eval results
    """
    exp_name: str
    dataset_name: str
    dataset_root: str = ""
    env_task: str = ""
    camera: str = "top"
    camera_names: str = "top,wrist"
    primary_image_key: str = "observation.images.top"
    wrist_image_key: str = "observation.images.wrist"
    eval_camera: str = "corner"
    eval_camera_names: str = "corner,gripperPOV"
    gpu_id: int = 0
    selection_num_episodes: int = SELECTION_NUM_EPISODES
    selection_seed: int = SELECTION_SEED
    selection_mode: str = "grid_uniform"
    train_steps: int = TRAIN_STEPS
    train_save_freq: int = TRAIN_SAVE_FREQ
    train_batch_size: int = TRAIN_BATCH_SIZE
    train_num_workers: int = TRAIN_NUM_WORKERS
    train_lr: float = TRAIN_LR
    eval_n_episodes: int = EVAL_N_EPISODES
    eval_seed: int = EVAL_SEED
    eval_batch_size: int = EVAL_BATCH_SIZE
    vq_model_path: str = VQ_MODEL_PATH
    rename_map: str = RENAME_MAP
    experiment_dir: Optional[str] = None
    subsets_dir: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    eval_results_dir: Optional[str] = None

    def __post_init__(self):
        if self.experiment_dir is None:
            self.experiment_dir = str(WORK2_ROOT / "differentvlm" / "minivla" / "experiments" / self.exp_name)
        if self.subsets_dir is None:
            self.subsets_dir = str(Path(self.experiment_dir) / "subsets")
        if self.checkpoints_dir is None:
            self.checkpoints_dir = str(Path(self.experiment_dir) / "checkpoints")
        if self.logs_dir is None:
            self.logs_dir = str(Path(self.experiment_dir) / "logs")
        if self.eval_results_dir is None:
            self.eval_results_dir = str(Path(self.experiment_dir) / "eval_results")

    def ensure_dirs(self):
        for d in [self.experiment_dir, self.subsets_dir, self.logs_dir, self.eval_results_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


def get_minivla_config(
    gpu_id: int = 0,
    dataset_name: str = "disassemble-v3_corner",
    num_episodes: int = 112,
    seed: int = 42,
    selection_mode: str = "grid_uniform",
) -> MiniVLAExperimentConfig:
    ds_config = DATASET_CONFIGS.get(dataset_name, {})
    
    # Build experiment name with dataset, model and algorithm info
    # Format: {selection_mode}_{num_episodes}_seed{seed}_{dataset_name}_{model}_{algorithm}
    # Example: random_112_seed42_disassemblyv3corner_minivla_random
    dataset_short = dataset_name.replace("-", "").replace("_", "")
    model_name = "minivla"
    exp_name = f"{selection_mode}_{num_episodes}_seed{seed}_{dataset_short}_{model_name}_{selection_mode}"
    
    return MiniVLAExperimentConfig(
        exp_name=exp_name,
        dataset_name=dataset_name,
        dataset_root=ds_config.get("dataset_root", ""),
        env_task=ds_config.get("env_task", ""),
        camera=ds_config.get("camera", "corner"),
        camera_names=ds_config.get("camera_names", "corner,gripperPOV"),
        primary_image_key=ds_config.get("primary_image_key", "observation.images.top"),
        wrist_image_key=ds_config.get("wrist_image_key", "observation.images.wrist"),
        eval_camera=ds_config.get("eval_camera", "corner"),
        eval_camera_names=ds_config.get("eval_camera_names", "corner,gripperPOV"),
        gpu_id=gpu_id,
        selection_num_episodes=num_episodes,
        selection_seed=seed,
        selection_mode=selection_mode,
        train_steps=TRAIN_STEPS,
        train_save_freq=TRAIN_SAVE_FREQ,
        train_batch_size=TRAIN_BATCH_SIZE,
        train_num_workers=TRAIN_NUM_WORKERS,
        train_lr=TRAIN_LR,
        eval_n_episodes=EVAL_N_EPISODES,
        eval_seed=EVAL_SEED,
        eval_batch_size=EVAL_BATCH_SIZE,
        vq_model_path=VQ_MODEL_PATH,
        rename_map=RENAME_MAP,
    )