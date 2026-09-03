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

DATASET_ROOT = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner"
LEROBOT_REPO_ID = "work2/metaworld_pick_place"

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
CAMERA_NAMES = "corner,gripperPOV"
RENAME_MAP = '{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

PCA_DIM = 32


@dataclass
class VLMExperimentConfig:
    """
    Configuration for a single differentvlm experiment.

    Fields:
        vlm_name: short name for this VLM (e.g. "llava_pythia400m")
        selection_vlm_model_id: HuggingFace model id of the VLM used for SELECTION embedding extraction
        selection_vlm_description: human-readable description of the selection VLM
        is_prismatic: whether this VLM uses Prismatic architecture (vision_encoder + projector + LLM)
        prismatic_base_model_id: base LLM model id for Prismatic construction (only if is_prismatic=True)
        prismatic_vision_model_id: vision encoder model id for Prismatic construction
        embedding_cache_dir: directory to store VLM-extracted episode embeddings
        dataset_root: path to the dataset directory
        lerobot_repo_id: LeRobot dataset repo id
        camera: camera configuration name (e.g. "corner")
        gpu_id: GPU device id
        selection_num_episodes: number of episodes to select
        selection_seed: random seed for selection
        train_steps: training steps for SmolVLA
        train_save_freq: checkpoint save frequency
        train_batch_size: training batch size
        train_num_workers: training num workers
        train_lr: training learning rate
        eval_n_episodes: number of evaluation episodes (fixed to 200)
        eval_seed: evaluation seed (fixed to 42)
        eval_batch_size: evaluation batch size
        smolvla_policy_model: SmolVLA policy model (FIXED, not experiment variable)
        camera_names: camera names for training/eval
        rename_map: feature rename map for training
        pca_dim: PCA dimensionality for embeddings
        results_dir: output directory for results
        checkpoints_dir: output directory for training checkpoints
        logs_dir: output directory for logs
        selected_dataset_dir: output directory for selected dataset
    """
    vlm_name: str
    selection_vlm_model_id: str
    selection_vlm_description: str = ""
    is_prismatic: bool = False
    prismatic_base_model_id: Optional[str] = None
    prismatic_vision_model_id: Optional[str] = None
    embedding_cache_dir: Optional[str] = None
    dataset_root: str = DATASET_ROOT
    lerobot_repo_id: str = LEROBOT_REPO_ID
    camera: str = "corner"
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
    camera_names: str = CAMERA_NAMES
    rename_map: str = RENAME_MAP
    pca_dim: int = PCA_DIM
    results_dir: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    selected_dataset_dir: Optional[str] = None

    def __post_init__(self):
        if self.results_dir is None:
            self.results_dir = str(WORK2_ROOT / "differentvlm" / "results" / f"{self.vlm_name}_{self.camera}")
        if self.checkpoints_dir is None:
            self.checkpoints_dir = str(WORK2_ROOT / "differentvlm" / "checkpoints" / f"{self.vlm_name}_{self.camera}")
        if self.logs_dir is None:
            self.logs_dir = str(WORK2_ROOT / "differentvlm" / "logs" / f"{self.vlm_name}_{self.camera}")
        if self.selected_dataset_dir is None:
            self.selected_dataset_dir = str(WORK2_ROOT / "differentvlm" / "selected_dataset" / f"{self.vlm_name}_{self.camera}")
        if self.embedding_cache_dir is None:
            self.embedding_cache_dir = str(WORK2_ROOT / "differentvlm" / "embeddings" / f"{self.vlm_name}_{self.camera}")

    def ensure_dirs(self):
        for d in [self.results_dir, self.checkpoints_dir, self.logs_dir,
                  self.selected_dataset_dir, self.embedding_cache_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


def get_llava_pythia400m_config(gpu_id: int = 0, camera: str = "corner") -> VLMExperimentConfig:
    """
    LLaVA-Pythia-400M configuration for selection embedding extraction.

    Model: lesjie/Llava-Pythia-400M
    - This is the VLM used by TinyVLA-S for visual representation
    - Uses LLaVA architecture with Pythia-400M as language backbone
    - Loaded via transformers.AutoModelForImageTextToText
    - Visual embedding extracted from the vision encoder's last hidden state
    - Only visual features are used, no text generation
    """
    return VLMExperimentConfig(
        vlm_name="llava_pythia400m",
        selection_vlm_model_id="lesjie/Llava-Pythia-400M",
        selection_vlm_description="LLaVA-Pythia-400M (TinyVLA-S VLM backbone)",
        camera=camera,
        gpu_id=gpu_id,
        is_prismatic=False,
    )


def get_prismatic_qwen25_05b_config(gpu_id: int = 0, camera: str = "corner") -> VLMExperimentConfig:
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
    return VLMExperimentConfig(
        vlm_name="prismatic_qwen25_05b",
        selection_vlm_model_id="lucidrains/prismatic-qwen2.5-0.5b",
        selection_vlm_description="Prismatic-Qwen2.5-0.5B (MiniVLA VLM backbone)",
        camera=camera,
        gpu_id=gpu_id,
        is_prismatic=True,
        prismatic_base_model_id="Qwen/Qwen2.5-0.5B",
        prismatic_vision_model_id="google/siglip-so400m-patch14-384",
    )


VLM_CONFIGS: Dict[str, callable] = {
    "llava_pythia400m": get_llava_pythia400m_config,
    "prismatic_qwen25_05b": get_prismatic_qwen25_05b_config,
}


def get_config(vlm_name: str, gpu_id: int = 0, camera: str = "corner") -> VLMExperimentConfig:
    if vlm_name not in VLM_CONFIGS:
        raise ValueError(f"Unknown VLM: {vlm_name}. Available: {list(VLM_CONFIGS.keys())}")
    cfg = VLM_CONFIGS[vlm_name](gpu_id=gpu_id, camera=camera)
    cfg.ensure_dirs()
    return cfg