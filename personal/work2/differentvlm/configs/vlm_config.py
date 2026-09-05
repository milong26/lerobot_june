"""
DifferentVLM Experiment Configuration

Manages all experiment configurations for different VLM backbone experiments.
Supports llava_pythia400m and prismatic_qwen25_05b experiments.

Key design:
- selection_vlm_model_id: the VLM backbone used ONLY for dataset selection embedding extraction
- SmolVLA policy backbone is FIXED to HuggingFaceTB/SmolVLM2-500M-Video-Instruct (not configurable)
- The experiment variable is ONLY the selection VLM, NOT the training policy
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORK2_ROOT = Path(__file__).resolve().parents[2]

# Dataset configurations: dataset_name -> (dataset_root, env_task, camera)
DATASET_CONFIGS = {
    "pick_place-v3_corner": {
        "dataset_root": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner",
        "env_task": "pick-place-v3",
        "camera": "corner",
        "camera_names": "corner,gripperPOV",
    },
    "disassemble-v3_corner": {
        "dataset_root": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner",
        "env_task": "disassemble-v3",
        "camera": "corner",
        "camera_names": "corner,gripperPOV",
    },
}

LEROBOT_REPO_ID = "1/2"

SELECTION_NUM_EPISODES = 112
SELECTION_SEED = 42

TRAIN_STEPS = 12000
TRAIN_SAVE_FREQ = 2000
TRAIN_BATCH_SIZE = 64
TRAIN_NUM_WORKERS = 16
TRAIN_LR = 1e-4

EVAL_N_EPISODES = 200
EVAL_SEED = 42
EVAL_BATCH_SIZE = 16

SmolVLA_POLICY_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TINYVLA_S_POLICY_MODEL = "tinyvla_s"
TINYVLA_B_POLICY_MODEL = "tinyvla_b"
CAMERA_NAMES = "corner,gripperPOV"
RENAME_MAP = '{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

PCA_DIM = 32

TINYVLA_TRAIN_STEPS = 12000
TINYVLA_SAVE_FREQ = 2000
TINYVLA_BATCH_SIZE = 32
TINYVLA_NUM_WORKERS = 8
TINYVLA_LR = 2e-4
TINYVLA_WEIGHT_DECAY = 0.0
TINYVLA_WARMUP_RATIO = 0.005


@dataclass
class VLMExperimentConfig:
    """
    Configuration for a single differentvlm experiment.

    Directory structure: differentvlm/experiments/{vlm_name}_{dataset_name}/
    ├── embeddings/          # VLM-extracted episode embeddings
    ├── results/             # Selection results, experiment summary
    ├── checkpoints/         # Training checkpoints
    ├── logs/                # Training and eval logs
    └── selected_dataset/    # Selected episode dataset

    Fields:
        vlm_name: short name for this VLM (e.g. "tinyvla_s")
        dataset_name: short name for dataset (e.g. "pick_place-v3_corner")
        selection_vlm_model_id: HuggingFace model id of the VLM used for SELECTION embedding extraction
        selection_vlm_description: human-readable description of the selection VLM
        is_prismatic: whether this VLM uses Prismatic architecture
        prismatic_base_model_id: base LLM model id for Prismatic construction
        prismatic_vision_model_id: vision encoder model id for Prismatic construction
        dataset_root: path to the dataset directory
        env_task: environment task name (e.g. "pick-place-v3")
        camera: camera configuration name (e.g. "corner")
        camera_names: camera names for training/eval
        gpu_id: GPU device id
        selection_num_episodes: number of episodes to select
        selection_seed: random seed for selection
        train_steps: training steps
        train_save_freq: checkpoint save frequency
        train_batch_size: training batch size
        train_num_workers: training num workers
        train_lr: training learning rate
        eval_n_episodes: number of evaluation episodes
        eval_seed: evaluation seed
        eval_batch_size: evaluation batch size
        smolvla_policy_model: SmolVLA policy model
        tinyvla_policy_type: TinyVLA policy type (tinyvla_s or tinyvla_b)
        rename_map: feature rename map for training
        pca_dim: PCA dimensionality for embeddings
        experiment_dir: base directory for all experiment outputs
        results_dir: output directory for results
        checkpoints_dir: output directory for training checkpoints
        logs_dir: output directory for logs
        selected_dataset_dir: output directory for selected dataset
        embedding_cache_dir: directory to store VLM-extracted episode embeddings
    """
    vlm_name: str
    dataset_name: str
    selection_vlm_model_id: str
    selection_vlm_description: str = ""
    is_prismatic: bool = False
    prismatic_base_model_id: Optional[str] = None
    prismatic_vision_model_id: Optional[str] = None
    dataset_root: str = ""
    env_task: str = ""
    camera: str = "corner"
    camera_names: str = CAMERA_NAMES
    gpu_id: int = 0
    selection_num_episodes: int = SELECTION_NUM_EPISODES
    selection_seed: int = SELECTION_SEED
    train_steps: int = TRAIN_STEPS
    train_save_freq: int = TRAIN_SAVE_FREQ
    train_batch_size: int = TRAIN_BATCH_SIZE
    train_num_workers: int = TRAIN_NUM_WORKERS
    train_lr: float = TRAIN_LR
    eval_n_episodes: int = EVAL_N_EPISODES
    eval_seed: int = EVAL_SEED
    eval_batch_size: int = EVAL_BATCH_SIZE
    smolvla_policy_model: str = SmolVLA_POLICY_MODEL
    tinyvla_policy_type: str = ""
    lerobot_repo_id: str = LEROBOT_REPO_ID
    rename_map: str = RENAME_MAP
    pca_dim: int = PCA_DIM
    experiment_dir: Optional[str] = None
    results_dir: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    selected_dataset_dir: Optional[str] = None
    embedding_cache_dir: Optional[str] = None

    def __post_init__(self):
        # Base experiment directory: differentvlm/experiments/{vlm_name}_{dataset_name}
        if self.experiment_dir is None:
            self.experiment_dir = str(WORK2_ROOT / "differentvlm" / "experiments" / f"{self.vlm_name}_{self.dataset_name}")
        
        # All subdirectories under experiment_dir
        if self.embedding_cache_dir is None:
            self.embedding_cache_dir = str(Path(self.experiment_dir) / "embeddings")
        if self.results_dir is None:
            self.results_dir = str(Path(self.experiment_dir) / "results")
        if self.checkpoints_dir is None:
            self.checkpoints_dir = str(Path(self.experiment_dir) / "checkpoints")
        if self.logs_dir is None:
            self.logs_dir = str(Path(self.experiment_dir) / "logs")
        if self.selected_dataset_dir is None:
            self.selected_dataset_dir = str(Path(self.experiment_dir) / "selected_dataset")

    def ensure_dirs(self):
        for d in [self.experiment_dir, self.results_dir, self.checkpoints_dir, 
                  self.logs_dir, self.selected_dataset_dir, self.embedding_cache_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


def get_llava_pythia400m_config(gpu_id: int = 0, dataset_name: str = "pick_place-v3_corner") -> VLMExperimentConfig:
    """
    LLaVA-Pythia-400M configuration for selection embedding extraction.

    Model: lesjie/Llava-Pythia-400M
    - This is the VLM used by TinyVLA-S for visual representation
    - Uses LLaVA architecture with Pythia-400M as language backbone
    - Loaded via transformers.AutoModelForImageTextToText
    - Visual embedding extracted from the vision encoder's last hidden state
    - Only visual features are used, no text generation
    """
    ds_config = DATASET_CONFIGS.get(dataset_name, {})
    return VLMExperimentConfig(
        vlm_name="llava_pythia400m",
        dataset_name=dataset_name,
        selection_vlm_model_id="lesjie/Llava-Pythia-400M",
        selection_vlm_description="LLaVA-Pythia-400M (TinyVLA-S VLM backbone)",
        is_prismatic=False,
        dataset_root=ds_config.get("dataset_root", ""),
        env_task=ds_config.get("env_task", ""),
        camera=ds_config.get("camera", "corner"),
        camera_names=ds_config.get("camera_names", CAMERA_NAMES),
        gpu_id=gpu_id,
    )


def get_prismatic_qwen25_05b_config(gpu_id: int = 0, dataset_name: str = "pick_place-v3_corner") -> VLMExperimentConfig:
    """
    Prismatic-Qwen2.5-0.5B configuration for selection embedding extraction.

    Model: Prismatic VLM architecture with Qwen2.5-0.5B as language backbone
    - Architecture: Vision Encoder (SigLIP/CLIP) -> Projector (MLP) -> Language Model (Qwen2.5-0.5B)
    - Visual embedding extracted from: image -> vision_encoder -> projector -> visual_feature
    - Does NOT use Qwen language hidden states as visual embedding
    - If lucidrains/prismatic-qwen2.5-0.5b checkpoint is unavailable, falls back to manual construction:
        1. Vision encoder: google/siglip-so400m-patch14-384 (or openai/clip-vit-large-patch14 as fallback)
        2. Projector: linear layer mapping vision_dim -> LLM_dim
        3. Language backbone: Qwen/Qwen2.5-0.5B
    """
    ds_config = DATASET_CONFIGS.get(dataset_name, {})
    return VLMExperimentConfig(
        vlm_name="prismatic_qwen25_05b",
        dataset_name=dataset_name,
        selection_vlm_model_id="lucidrains/prismatic-qwen2.5-0.5b",
        selection_vlm_description="Prismatic-Qwen2.5-0.5B (MiniVLA VLM backbone)",
        is_prismatic=True,
        prismatic_base_model_id="Qwen/Qwen2.5-0.5B",
        prismatic_vision_model_id="google/siglip-so400m-patch14-384",
        dataset_root=ds_config.get("dataset_root", ""),
        env_task=ds_config.get("env_task", ""),
        camera=ds_config.get("camera", "corner"),
        camera_names=ds_config.get("camera_names", CAMERA_NAMES),
        gpu_id=gpu_id,
    )


def get_tinyvla_s_config(gpu_id: int = 0, dataset_name: str = "pick_place-v3_corner", num_episodes: int = 112) -> VLMExperimentConfig:
    """
    TinyVLA-S configuration for fine-tuning.

    Model: lesjie/Llava-Pythia-400M as backbone with diffusion action head
    - Policy type: tinyvla_s (registered in LeRobot)
    - Action head: droid_diffusion (UNet-based diffusion policy)
    - LoRA: enabled with r=64, alpha=256, targeting 'vit llm' modules
    - Training: lr=2e-4, weight_decay=0, warmup_ratio=0.005, cosine scheduler
    - Freezing: vision_tower and backbone frozen, only LoRA and action head trained
    - Chunk size: 16, n_action_steps: 16
    """
    ds_config = DATASET_CONFIGS.get(dataset_name, {})
    return VLMExperimentConfig(
        vlm_name="tinyvla_s",
        dataset_name=dataset_name,
        selection_vlm_model_id="lesjie/Llava-Pythia-400M",
        selection_vlm_description="TinyVLA-S (LLaVA-Pythia-400M + Diffusion Action Head)",
        is_prismatic=False,
        dataset_root=ds_config.get("dataset_root", ""),
        env_task=ds_config.get("env_task", ""),
        camera=ds_config.get("camera", "corner"),
        camera_names=ds_config.get("camera_names", CAMERA_NAMES),
        gpu_id=gpu_id,
        selection_num_episodes=num_episodes,
        train_steps=TINYVLA_TRAIN_STEPS,
        train_save_freq=TINYVLA_SAVE_FREQ,
        train_batch_size=TINYVLA_BATCH_SIZE,
        train_num_workers=TINYVLA_NUM_WORKERS,
        train_lr=TINYVLA_LR,
        tinyvla_policy_type="tinyvla_s",
    )


def get_tinyvla_b_config(gpu_id: int = 0, dataset_name: str = "pick_place-v3_corner", num_episodes: int = 112) -> VLMExperimentConfig:
    """
    TinyVLA-B configuration for fine-tuning.

    Model: lesjie/Llava-Pythia-700M as backbone with diffusion action head
    - Policy type: tinyvla_b (registered in LeRobot)
    - Action head: droid_diffusion (UNet-based diffusion policy)
    - LoRA: enabled with r=64, alpha=256, targeting 'vit llm' modules
    - Training: lr=2e-4, weight_decay=0, warmup_ratio=0.005, cosine scheduler
    - Freezing: vision_tower and backbone frozen, only LoRA and action head trained
    - Chunk size: 16, n_action_steps: 16
    """
    ds_config = DATASET_CONFIGS.get(dataset_name, {})
    return VLMExperimentConfig(
        vlm_name="tinyvla_b",
        dataset_name=dataset_name,
        selection_vlm_model_id="lesjie/Llava-Pythia-700M",
        selection_vlm_description="TinyVLA-B (LLaVA-Pythia-700M + Diffusion Action Head)",
        is_prismatic=False,
        dataset_root=ds_config.get("dataset_root", ""),
        env_task=ds_config.get("env_task", ""),
        camera=ds_config.get("camera", "corner"),
        camera_names=ds_config.get("camera_names", CAMERA_NAMES),
        gpu_id=gpu_id,
        selection_num_episodes=num_episodes,
        train_steps=TINYVLA_TRAIN_STEPS,
        train_save_freq=TINYVLA_SAVE_FREQ,
        train_batch_size=TINYVLA_BATCH_SIZE,
        train_num_workers=TINYVLA_NUM_WORKERS,
        train_lr=TINYVLA_LR,
        tinyvla_policy_type="tinyvla_b",
    )


VLM_CONFIGS: Dict[str, callable] = {
    "llava_pythia400m": get_llava_pythia400m_config,
    "prismatic_qwen25_05b": get_prismatic_qwen25_05b_config,
    "tinyvla_s": get_tinyvla_s_config,
    "tinyvla_b": get_tinyvla_b_config,
}


def get_config(vlm_name: str, gpu_id: int = 0, dataset_name: str = "pick_place-v3_corner") -> VLMExperimentConfig:
    if vlm_name not in VLM_CONFIGS:
        raise ValueError(f"Unknown VLM: {vlm_name}. Available: {list(VLM_CONFIGS.keys())}")
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    cfg = VLM_CONFIGS[vlm_name](gpu_id=gpu_id, dataset_name=dataset_name)
    cfg.ensure_dirs()
    return cfg