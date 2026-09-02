"""
DemInf (Demonstration Information) - LeRobot/PyTorch Implementation

Faithful reimplementation of DemInf core algorithm in the current LeRobot pipeline.
Not a copy of the original JAX/OpenX code.

Core API:
    - DemInfConfig: unified configuration dataclass
    - score_dataset: compute per-episode DemInf scores
    - select_top_episodes: select top-K episodes by score
"""

from deminf.config import DemInfConfig
from deminf.score_episodes import score_dataset, score_latents
from deminf.select_subset import select_top_episodes, save_subset_json

__all__ = [
    "DemInfConfig",
    "score_dataset",
    "score_latents",
    "select_top_episodes",
    "save_subset_json",
]