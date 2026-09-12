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

TRAIN_STEPS = 50000
TRAIN_SAVE_FREQ = 2000
TRAIN_BATCH_SIZE = 2
TRAIN_NUM_WORKERS = 16

# Action tokenizer: MetaWorld uses extra_action_tokenizer (non-VQ, supports 4D natively)
# Do NOT use libero_vq_action_tokenizer for MetaWorld (7D VQ != 4D MetaWorld action)
ACTION_TOKENIZER_TYPE = "extra_action_tokenizer"

# VQ model path (only used when action_tokenizer_type is a VQ type)
VQ_MODEL_PATH = "Stanford-ILIAD/pretrain_vq"

# Official MiniVLA pretrained checkpoint for backbone-only initialization.
# This checkpoint provides vision_backbone, projector, and llm_backbone weights.
# Action head is NOT loaded from this checkpoint; it is reinitialized for MetaWorld 4D actions.
# Use HuggingFace model name (will be resolved via HF cache) or local path.
# Set to empty string to disable official pretrained initialization.
OFFICIAL_PRETRAINED_CHECKPOINT = "Stanford-ILIAD/minivla-libero90-prismatic"

# Official MiniVLA default learning rate (from teach_code/MiniVLA prismatic/conf/vla.py)
# This is the default AdamW learning rate used in official MiniVLA training.
OFFICIAL_LR = 2e-5

# Official MiniVLA optimizer defaults (from teach_code/MiniVLA/prismatic/conf/vla.py)
OFFICIAL_WEIGHT_DECAY = 0.0
OFFICIAL_MAX_GRAD_NORM = 1.0
OFFICIAL_LR_SCHEDULER_TYPE = "constant"
OFFICIAL_WARMUP_RATIO = 0.0

EVAL_N_EPISODES = 10
EVAL_SEED = 42
EVAL_BATCH_SIZE = 4

RENAME_MAP = '{"observation.images.camera1": "observation.images.top", "observation.images.camera2": "observation.images.wrist"}'


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
    train_lr: float = 2e-5  # Official MiniVLA default AdamW learning rate
    policy_type: str = "minivla_wrist"  # minivla (single camera) or minivla_wrist (dual camera)
    action_tokenizer_type: str = ACTION_TOKENIZER_TYPE
    official_vla_checkpoint: str = ""
    official_init_mode: str = "backbone_only"
    official_pretrained_checkpoint: str = OFFICIAL_PRETRAINED_CHECKPOINT
    official_lr: float = OFFICIAL_LR
    official_weight_decay: float = OFFICIAL_WEIGHT_DECAY
    official_max_grad_norm: float = OFFICIAL_MAX_GRAD_NORM
    official_lr_scheduler_type: str = OFFICIAL_LR_SCHEDULER_TYPE
    official_warmup_ratio: float = OFFICIAL_WARMUP_RATIO
    resume: bool = False
    restart: bool = False
    scheduler_warmup_steps: int = 0
    projector_lr: float = 0.0
    backbone_lr: float = 0.0
    eval_n_episodes: int = EVAL_N_EPISODES
    eval_seed: int = EVAL_SEED
    eval_batch_size: int = EVAL_BATCH_SIZE
    env_eval_freq: int = 0  # 0 = disabled, >0 = eval every N steps
    use_self_mw: bool = True  # Use self-collected Meta-World dataset format
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
    train_steps: int = TRAIN_STEPS,
    train_save_freq: int = TRAIN_SAVE_FREQ,
    env_eval_freq: int = 0,
    eval_n_episodes: int = EVAL_N_EPISODES,
    use_self_mw: bool = True,
    policy_type: str = "minivla_wrist",
    action_tokenizer_type: str = ACTION_TOKENIZER_TYPE,
    official_vla_checkpoint: str = "",
    official_init_mode: str = "backbone_only",
    official_pretrained_checkpoint: str = OFFICIAL_PRETRAINED_CHECKPOINT,
    resume: bool = False,
    restart: bool = False,
    scheduler_warmup_steps: int = 0,
    projector_lr: float = 0.0,
    backbone_lr: float = 0.0,
    exp_suffix: str = "",
) -> MiniVLAExperimentConfig:
    ds_config = DATASET_CONFIGS.get(dataset_name, {})
    
    # Build experiment name with dataset, model and algorithm info
    # Format: {selection_mode}_{num_episodes}_seed{seed}_{dataset_name}_{model}_{algorithm}[_{suffix}]
    # Example: random_112_seed42_disassemblyv3corner_minivla_random
    # Example with suffix: random_200_seed42_disassemblev3corner_minivla_random_official
    dataset_short = dataset_name.replace("-", "").replace("_", "")
    model_name = "minivla"
    exp_name = f"{selection_mode}_{num_episodes}_seed{seed}_{dataset_short}_{model_name}_{selection_mode}"
    if exp_suffix:
        exp_name = f"{exp_name}_{exp_suffix}"
    
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
        train_steps=train_steps,
        train_save_freq=train_save_freq,
        train_batch_size=TRAIN_BATCH_SIZE,
        train_num_workers=TRAIN_NUM_WORKERS,
        train_lr=2e-5,
        policy_type=policy_type,
        action_tokenizer_type=action_tokenizer_type,
        official_vla_checkpoint=official_vla_checkpoint,
        official_init_mode=official_init_mode,
        official_pretrained_checkpoint=official_pretrained_checkpoint,
        resume=resume,
        restart=restart,
        scheduler_warmup_steps=scheduler_warmup_steps,
        projector_lr=projector_lr,
        backbone_lr=backbone_lr,
        eval_n_episodes=eval_n_episodes,
        eval_seed=EVAL_SEED,
        eval_batch_size=EVAL_BATCH_SIZE,
        env_eval_freq=env_eval_freq,
        use_self_mw=use_self_mw,
        vq_model_path=VQ_MODEL_PATH,
        rename_map=RENAME_MAP,
    )