"""Shared configuration for work5 experiment pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

WORK_DIR = Path(__file__).parent

@dataclass
class Config:
    # Paths
    work_dir: Path = WORK_DIR
    datasets_dir: Path = field(default_factory=lambda: WORK_DIR / "datasets")
    checkpoints_dir: Path = field(default_factory=lambda: WORK_DIR / "checkpoints")
    figures_dir: Path = field(default_factory=lambda: WORK_DIR / "figures")

    # MetaWorld task
    task_name: str = "pick-place-v3"
    image_size: int = 224
    fps: int = 80
    max_steps: int = 500

    # B0 grid: 5x5 object positions x 1 demo each = 25 demos
    grid_n_per_axis: int = 5
    b0_demos: int = 25
    total_budget: int = 75  # 3x B0
    n_candidates: int = 50  # candidates to select from

    # VLM model
    vlm_model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    device: str = "cuda"

    # SIC hyperparameters
    d_pca: int = 32
    alpha: float = 0.05
    lambda_weight: float = 0.5

    # Noise augmentation for SIC-Noise
    noise_sigma_pos: float = 0.02  # meters
    noise_sigma_rot_deg: float = 15.0  # degrees
    n_noise_per_undercovered: int = 5
    n_undercovered: int = 10  # most under-covered positions to augment

    # Training
    training_steps: int = 8000
    eval_every: int = 2000
    batch_size: int = 16
    learning_rate: float = 1e-4
    action_chunk_size: int = 10
    n_eval_episodes: int = 100
    n_eval_runs: int = 3  # for error bars

    # Camera keys (match LeRobot dataset features)
    global_cam_key: str = "observation.images.top"
    wrist_cam_key: str = "observation.images.wrist"

    # Figure settings
    figure_dpi_preview: int = 150
    figure_dpi_high: int = 300

    def setup(self):
        for d in [self.datasets_dir, self.checkpoints_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

CFG = Config()