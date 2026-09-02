"""
DemInf (Demonstration Information) - LeRobot/PyTorch Implementation

Faithful reimplementation of DemInf core algorithm in the current LeRobot pipeline.
Not a copy of the original JAX/OpenX code.

Core API:
    - DemInfConfig: unified configuration dataclass
    - validate_deminf_config: configuration validation function
    - score_latents: core scoring function on pre-computed latents
    - select_top_episodes: select top-K episodes by score
    - save_subset_json: save subset JSON with DemInf metadata
"""

from deminf.config import DemInfConfig, validate_deminf_config
from deminf.score_episodes import score_latents
from deminf.select_subset import select_top_episodes, save_subset_json

__all__ = [
    "DemInfConfig",
    "validate_deminf_config",
    "score_latents",
    "select_top_episodes",
    "save_subset_json",
]