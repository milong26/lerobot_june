"""SIC Framework for MetaWorld - Global Configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import json


@dataclass
class ProjectConfig:
    # ============ 数据路径 ============
    dataset_root: str = ""
    config_map_path: str = "personal/work3/data/config_map.json"
    success_rates_path: str = "personal/work3/data/success_rates.json"

    # ============ 模型配置 ============
    vlm_model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    device: str = "cuda"
    batch_size: int = 4

    # 相机键名（与 LeRobot 数据集中的键名一致）
    global_cam_key: str = "observation.images.top"
    wrist_cam_key: str = "observation.images.wrist"

    # ============ SIC 超参数 ============
    d_pca: int = 32
    lambda_weight: float = 0.5
    alpha: float = 0.05
    t_max: int = 4

    # ============ 实验配置 ============
    budget_B: int = 144
    n_positions: int = 9
    n_rotations: int = 8

    # ============ 输出配置 ============
    figures_dir: str = "personal/work3/figures"
    results_dir: str = "personal/work3/results"
    cache_dir: str = "personal/work3/cache"
    figure_dpi: int = 300
    figure_format: str = "pdf"

    def setup(self):
        for d in [self.figures_dir, self.results_dir, self.cache_dir,
                  "personal/work3/data"]:
            os.makedirs(d, exist_ok=True)

    def load_success_rates(self) -> Optional[Dict]:
        if os.path.exists(self.success_rates_path):
            with open(self.success_rates_path) as f:
                return json.load(f)
        return None


CFG = ProjectConfig()