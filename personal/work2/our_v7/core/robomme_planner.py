"""RoboMME adapter for the paper-faithful Our-V7 planner.

RoboMME does not expose MetaWorld ``rand_vec``.  Its local datasets store reset
metadata in ``episode_initial_states.json -> episodes[*].initial_configuration``.
Our-V7 converts only that reset-time task configuration into a semantic
configuration vector. Full ``scene_state`` and dynamic velocity/qvel terms are
excluded. Named entities are keyed by semantic name, categorical task factors
are deterministically one-hot encoded, and ordered task sequences keep explicit
slot semantics.

The causal acquisition logic (deterministic KMeans, B0 farthest traversal,
acquired-only visual/action priority, maximin target, and region-restricted
archive mapping) is inherited unchanged from :class:`V7Planner`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from our_v7.config import (
    B0_REGION_RATIO,
    DEFAULT_VISUAL_VARIANT,
    MAX_REGIONS,
    MIN_REGIONS,
    PCA_DIM,
    REGION_RATIO,
    SEED,
    TOTAL_BUDGET,
)
from our_v7.core.planner import V7Planner
from our_v7.core.robomme_configuration import load_v7_robomme_configuration_metadata
from our_v7.core.visual_embedding import CachedFrozenVLMEpisodeEncoder, CausalVisualProjector


class RoboMMEV7Planner(V7Planner):
    """V7 with a RoboMME configuration/dataset adapter.

    RoboMME does not currently expose a separate analytic Bounds object analogous
    to MetaWorld's ``_random_reset_space``. After semantic reset-configuration
    construction, each active dimension is normalized with the low/high values
    of the globally visible reset-metadata support. This remains within the V7
    causal boundary because no trajectory image/action/state/reward is used
    before acquisition.
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        total_budget: int = TOTAL_BUDGET,
        visual_variant: str = DEFAULT_VISUAL_VARIANT,
        device: str = "cuda",
        seed: int = SEED,
        region_ratio: float = REGION_RATIO,
        min_regions: int = MIN_REGIONS,
        max_regions: int = MAX_REGIONS,
        b0_region_ratio: float = B0_REGION_RATIO,
        ablation: str = "full",
        raw_visual_cache_root: str | None = None,
    ) -> None:
        if ablation not in {"full", "wo_action", "wo_adaptive_priority"}:
            raise ValueError("ablation must be full, wo_action, or wo_adaptive_priority")

        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.total_budget = int(total_budget)
        self.visual_variant = visual_variant
        self.device = device
        self.seed = int(seed)
        self.region_ratio = float(region_ratio)
        self.min_regions = int(min_regions)
        self.max_regions = int(max_regions)
        self.b0_region_ratio = float(b0_region_ratio)
        self.ablation = ablation

        ids, raw_configs, config_keys, config_schema = load_v7_robomme_configuration_metadata(
            str(self.dataset_root)
        )
        self.episode_ids = ids
        self.raw_configs = np.asarray(raw_configs, dtype=np.float32)
        self.configuration_keys = list(config_keys)
        self.configuration_schema = dict(config_schema)

        if self.raw_configs.ndim != 2 or self.raw_configs.shape[0] != len(self.episode_ids):
            raise ValueError(
                f"Invalid RoboMME configuration matrix shape={self.raw_configs.shape}, "
                f"episode_ids={len(self.episode_ids)}"
            )
        if self.raw_configs.shape[1] != len(self.configuration_keys):
            raise ValueError(
                f"RoboMME configuration key mismatch: dim={self.raw_configs.shape[1]}, "
                f"keys={len(self.configuration_keys)}"
            )
        if not np.all(np.isfinite(self.raw_configs)):
            raise ValueError("RoboMME configuration matrix contains non-finite values")

        # The JSON contains the globally visible admissible archive support. It
        # does not contain a separate task-space Bounds object analogous to
        # MetaWorld's _random_reset_space, so its per-dimension support bounds
        # are the reproducible archive-backed feasible bounds used here.
        self.config_low = self.raw_configs.min(axis=0).astype(np.float32)
        self.config_high = self.raw_configs.max(axis=0).astype(np.float32)
        spans = self.config_high - self.config_low
        if np.any(spans <= 1e-12):
            # The V7 parser removes constant columns before returning.
            bad = np.where(spans <= 1e-12)[0].tolist()
            raise ValueError(f"Constant RoboMME configuration dimensions remain after filtering: {bad}")
        self.config_bounds_source = (
            "RoboMME:episode_initial_states.json/initial_configuration semantic admissible-support bounds"
        )
        self.configs = self._normalize_with_predefined_bounds(self.raw_configs)

        if self.total_budget < 1:
            raise ValueError("total_budget must be >= 1")
        if self.total_budget > len(self.episode_ids):
            raise ValueError(
                f"budget={self.total_budget} exceeds archive size={len(self.episode_ids)}"
            )

        self.id_to_row = {ep: i for i, ep in enumerate(self.episode_ids)}
        self.regions, self.episode_to_region, self.kmeans_init_episode_ids = self._build_regions()
        self.num_regions = len(self.regions)

        self.acquired: List[int] = []
        self.acquired_set: set[int] = set()
        self.visual_embeddings: Dict[int, np.ndarray] = {}
        self.action_embeddings: Dict[int, np.ndarray] = {}
        self.history: List[Dict] = []
        self._rr_cursor = 0

        self.dataset = self._load_dataset()
        self.visual_encoder = CachedFrozenVLMEpisodeEncoder(
            self.dataset,
            dataset_name=self.dataset_name,
            device=self.device,
            cache_root=raw_visual_cache_root,
        )
        self.visual_projector = CausalVisualProjector(
            variant=self.visual_variant,
            pca_dim=PCA_DIM,
        )

    def _load_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=f"work2/robomme_{self.dataset_name}",
            root=str(self.dataset_root),
        )
        bad = [ep for ep in self.episode_ids if ep < 0 or ep >= dataset.num_episodes]
        if bad:
            raise ValueError(f"RoboMME configuration metadata contains invalid episode ids: {bad[:10]}")
        return dataset

    def run(self) -> dict:
        result = super().run()
        result["benchmark"] = "robomme"
        result["configuration_source"] = "initial_configuration"
        result["configuration_fields"] = [
            "movable_objects",
            "randomized_targets",
            "articulations",
            "task_config",
        ]
        result["configuration_encoding"] = {
            "entities": "semantic-name keyed; list order ignored; duplicate names rejected",
            "poses": "semantic x/y/z + canonicalized qw/qx/qy/qz",
            "articulation_state": "qpos only",
            "task_categorical": "deterministic one-hot",
            "ordered_task_sequences": "slot_XX categorical one-hot",
            "excluded": ["scene_state", "velocity", "linear_velocity", "angular_velocity", "qvel"],
        }
        result["configuration_schema"] = self.configuration_schema
        result["configuration_keys"] = self.configuration_keys
        result["scene_state_used_for_selection"] = False
        return result
