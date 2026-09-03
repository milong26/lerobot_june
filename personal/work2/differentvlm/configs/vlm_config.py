"""
DifferentVLM Experiment Configuration

Manages all experiment configurations for different VLM backbone experiments.
Supports llava_pythia400m and prismatic_qwen25_05b experiments.
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

VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
CAMERA_NAMES = "corner,gripperPOV"
RENAME_MAP = '{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

PCA_DIM = 32


@dataclass
class VLMExperimentConfig:
    vlm_name: str
    hf_model_id: str
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
    vlm_model_name: str = VLM_MODEL_NAME
    camera_names: str = CAMERA_NAMES
    rename_map: str = RENAME_MAP
    pca_dim: int = PCA_DIM
    results_dir: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    selected_dataset_dir: Optional[str] = None
    is_prismatic: bool = False
    prismatic_base_model_id: Optional[str] = None

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
    return VLMExperimentConfig(
        vlm_name="llava_pythia400m",
        hf_model_id="lesjie/Llava-Pythia-400M",
        camera=camera,
        gpu_id=gpu_id,
        is_prismatic=False,
    )


def get_prismatic_qwen25_05b_config(gpu_id: int = 0, camera: str = "corner") -> VLMExperimentConfig:
    return VLMExperimentConfig(
        vlm_name="prismatic_qwen25_05b",
        hf_model_id="lucidrains/prismatic-qwen2.5-0.5b",
        camera=camera,
        gpu_id=gpu_id,
        is_prismatic=True,
        prismatic_base_model_id="Qwen/Qwen2.5-0.5B",
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