"""Causal V6 planner: V5 configuration hierarchy + V4 information boundary.

Before acquisition, the planner may inspect only initial-configuration metadata
(rand_vec). Visual observations and actions are revealed only after an episode
has been committed to the acquired set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from our_v6.config import (
    B0_REGION_RATIO,
    COVERAGE_WEIGHT,
    DEFAULT_VISUAL_VARIANT,
    MAX_REGIONS,
    MIN_REGIONS,
    PCA_DIM,
    REGION_ACTION_WEIGHT,
    REGION_RATIO,
    REGION_VISUAL_WEIGHT,
    SEED,
    TOTAL_BUDGET,
)
from our_v6.core.action_embedding import reveal_action_embedding
from our_v6.core.visual_embedding import CausalVisualProjector, FrozenVLMEpisodeEncoder


@dataclass
class RegionInfo:
    region_id: int
    member_ids: List[int]
    centroid: np.ndarray


class V6Planner:
    """Strictly causal adaptive demonstration acquisition.

    The archive exposes rand_vec for every candidate before acquisition. Hidden
    trajectory content (images/actions) is accessed only after commit.
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
        coverage_weight: float = COVERAGE_WEIGHT,
        visual_weight: float = REGION_VISUAL_WEIGHT,
        action_weight: float = REGION_ACTION_WEIGHT,
        ablation: str = "full",
    ) -> None:
        if ablation not in {"full", "wo_action", "wo_adaptive_priority"}:
            raise ValueError(
                "ablation must be one of: full, wo_action, wo_adaptive_priority"
            )
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
        self.coverage_weight = float(coverage_weight)
        self.visual_weight = float(visual_weight)
        self.action_weight = 0.0 if ablation == "wo_action" else float(action_weight)
        self.ablation = ablation

        self.episode_ids, self.raw_configs = self._load_configuration_metadata()
        self.configs = self._minmax_normalize(self.raw_configs)
        if self.total_budget < 1:
            raise ValueError("total_budget must be >= 1")
        if self.total_budget > len(self.episode_ids):
            raise ValueError(
                f"budget={self.total_budget} exceeds archive size={len(self.episode_ids)}"
            )

        self.id_to_row = {ep: i for i, ep in enumerate(self.episode_ids)}
        self.regions, self.episode_to_region = self._build_regions()
        self.num_regions = len(self.regions)

        self.acquired: List[int] = []
        self.acquired_set = set()
        self.visual_embeddings: Dict[int, np.ndarray] = {}
        self.action_embeddings: Dict[int, np.ndarray] = {}
        self.history: List[Dict] = []
        self._rr_cursor = 0

        self.dataset = self._load_dataset()
        self.visual_encoder = FrozenVLMEpisodeEncoder(
            self.dataset, dataset_name=self.dataset_name, device=self.device
        )
        self.visual_projector = CausalVisualProjector(
            variant=self.visual_variant, pca_dim=PCA_DIM
        )

    def _load_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id="work2/metaworld_pick_place",
            root=str(self.dataset_root),
        )
        bad = [ep for ep in self.episode_ids if ep < 0 or ep >= dataset.num_episodes]
        if bad:
            raise ValueError(
                f"episode_initial_states.json contains indices outside dataset: {bad[:10]}"
            )
        return dataset

    def _load_configuration_metadata(self) -> Tuple[List[int], np.ndarray]:
        import json

        path = self.dataset_root / "episode_initial_states.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration metadata: {path}")
        with path.open("r") as f:
            payload = json.load(f)

        ids: List[int] = []
        configs: List[np.ndarray] = []
        for item in payload.get("episodes", []):
            ep = item.get("episode_index")
            rv = item.get("rand_vec")
            if ep is None or rv is None:
                continue
            arr = np.asarray(rv, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Non-finite rand_vec for episode {ep}")
            ids.append(int(ep))
            configs.append(arr)

        if not ids:
            raise ValueError(
                f"No rand_vec metadata found in {path}; V6 does not fall back to hidden trajectory content"
            )
        dims = {x.shape[0] for x in configs}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent rand_vec dimensions: {sorted(dims)}")

        order = np.argsort(np.asarray(ids))
        ids = [ids[int(i)] for i in order]
        matrix = np.stack([configs[int(i)] for i in order], axis=0)
        return ids, matrix

    @staticmethod
    def _minmax_normalize(matrix: np.ndarray) -> np.ndarray:
        lo = matrix.min(axis=0)
        hi = matrix.max(axis=0)
        span = hi - lo
        span = np.where(span < 1e-12, 1.0, span)
        return ((matrix - lo) / span).astype(np.float32)

    def _build_regions(self) -> Tuple[Dict[int, RegionInfo], Dict[int, int]]:
        n = len(self.episode_ids)
        k = max(self.min_regions, min(self.max_regions, int(np.floor(self.region_ratio * n))))
        k = max(1, min(k, n))

        # Fixed clustering seed: B0/region structure is deterministic and does not
        # depend on the experimental selection seed.
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(self.configs)

        regions: Dict[int, RegionInfo] = {}
        ep_to_region: Dict[int, int] = {}
        for rid in range(k):
            members = [
                self.episode_ids[i] for i, lab in enumerate(labels) if int(lab) == rid
            ]
            regions[rid] = RegionInfo(
                region_id=rid,
                member_ids=sorted(members),
                centroid=km.cluster_centers_[rid].astype(np.float32),
            )
            for ep in members:
                ep_to_region[ep] = rid
        return regions, ep_to_region

    def _config(self, ep: int) -> np.ndarray:
        return self.configs[self.id_to_row[ep]]

    def _unused_in_region(self, rid: int) -> List[int]:
        return [ep for ep in self.regions[rid].member_ids if ep not in self.acquired_set]

    @staticmethod
    def _mean_pairwise(vectors: Sequence[np.ndarray]) -> float:
        if len(vectors) < 2:
            return 0.0
        arr = np.stack(vectors, axis=0)
        total = 0.0
        count = 0
        for i in range(arr.shape[0]):
            d = np.linalg.norm(arr[i + 1 :] - arr[i], axis=1)
            total += float(d.sum())
            count += int(d.size)
        return total / count if count else 0.0

    @staticmethod
    def _minmax(values: Dict[int, float]) -> Dict[int, float]:
        if not values:
            return {}
        vals = np.asarray(list(values.values()), dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-12:
            return {k: 0.0 for k in values}
        return {k: (float(v) - lo) / (hi - lo) for k, v in values.items()}

    def _active_regions(self) -> List[int]:
        return [rid for rid in sorted(self.regions) if self._unused_in_region(rid)]

    def _region_scores(self) -> Dict[int, Dict[str, float]]:
        active = self._active_regions()
        coverage: Dict[int, float] = {}
        visual: Dict[int, float] = {}
        action: Dict[int, float] = {}

        for rid in active:
            region = self.regions[rid]
            acquired_here = [ep for ep in region.member_ids if ep in self.acquired_set]
            coverage[rid] = 1.0 - len(acquired_here) / max(1, len(region.member_ids))
            visual[rid] = self._mean_pairwise(
                [self.visual_embeddings[ep] for ep in acquired_here if ep in self.visual_embeddings]
            )
            action[rid] = self._mean_pairwise(
                [self.action_embeddings[ep] for ep in acquired_here if ep in self.action_embeddings]
            )

        nc = self._minmax(coverage)
        nv = self._minmax(visual)
        na = self._minmax(action)
        result: Dict[int, Dict[str, float]] = {}
        for rid in active:
            priority = (
                self.coverage_weight * nc[rid]
                + self.visual_weight * nv[rid]
                + self.action_weight * na[rid]
            )
            result[rid] = {
                "coverage_gap": coverage[rid],
                "visual_heterogeneity": visual[rid],
                "action_heterogeneity": action[rid],
                "coverage_norm": nc[rid],
                "visual_norm": nv[rid],
                "action_norm": na[rid],
                "priority": priority,
            }
        return result

    def _select_region(self) -> Tuple[int, Dict[str, float]]:
        scores = self._region_scores()
        if not scores:
            raise RuntimeError("No non-exhausted region remains")

        if self.ablation == "wo_adaptive_priority":
            active = sorted(scores)
            for _ in range(len(self.regions) + 1):
                rid = self._rr_cursor % self.num_regions
                self._rr_cursor += 1
                if rid in scores:
                    return rid, scores[rid]
            return active[0], scores[active[0]]

        rid = max(scores, key=lambda r: (scores[r]["priority"], -r))
        return rid, scores[rid]

    def _representative_episode(self, rid: int) -> int:
        region = self.regions[rid]
        candidates = self._unused_in_region(rid)
        return min(
            candidates,
            key=lambda ep: (float(np.linalg.norm(self._config(ep) - region.centroid)), ep),
        )

    def _target_episode_maximin(self, rid: int) -> int:
        """Configuration-only target proposal within the selected region.

        The proposal set is the unused configuration metadata in that region.
        No unseen visual/action feature participates in this decision.
        """
        candidates = self._unused_in_region(rid)
        if not candidates:
            raise RuntimeError(f"Region {rid} exhausted")
        acquired_configs = [self._config(ep) for ep in self.acquired]
        if not acquired_configs:
            return self._representative_episode(rid)
        acquired_matrix = np.stack(acquired_configs, axis=0)

        best_ep: Optional[int] = None
        best_score = -1.0
        for ep in candidates:
            score = float(np.linalg.norm(acquired_matrix - self._config(ep), axis=1).min())
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and (best_ep is None or ep < best_ep)
            ):
                best_score = score
                best_ep = ep
        assert best_ep is not None
        return best_ep

    def _refresh_visual_dict(self) -> None:
        self.visual_embeddings = self.visual_projector.as_dict()

    def _commit_and_reveal(
        self,
        ep: int,
        stage: str,
        rid: int,
        region_score: Optional[Dict[str, float]] = None,
    ) -> None:
        if ep in self.acquired_set:
            raise RuntimeError(f"Duplicate acquisition attempted for episode {ep}")

        # Commit first. Only after this point may trajectory content be read.
        self.acquired.append(ep)
        self.acquired_set.add(ep)

        raw_visual = self.visual_encoder.extract(ep, self.acquired_set)
        self.visual_projector.add(ep, raw_visual)
        self._refresh_visual_dict()
        self.action_embeddings[ep] = reveal_action_embedding(
            self.dataset, ep, self.acquired_set
        )

        entry = {
            "step": len(self.acquired),
            "stage": stage,
            "episode_index": ep,
            "region_id": rid,
            "target_config": self._config(ep).tolist(),
            "visual_variant": self.visual_variant,
            "pca_components": dict(self.visual_projector.last_pca_components),
            "ablation": self.ablation,
        }
        if region_score is not None:
            entry.update(region_score)
        self.history.append(entry)
        print(
            f"[{len(self.acquired):03d}/{self.total_budget}] {stage}: "
            f"region={rid}, ep={ep}, variant={self.visual_variant}"
        )

    def _initial_region_order(self, b0_size: int) -> List[int]:
        ordered = sorted(
            self.regions,
            key=lambda rid: (tuple(self.regions[rid].centroid.tolist()), rid),
        )
        if b0_size >= len(ordered):
            return ordered
        # Deterministic spread across the ordered configuration regions.
        positions = np.linspace(0, len(ordered) - 1, b0_size)
        chosen_positions: List[int] = []
        for x in positions:
            p = int(round(float(x)))
            if p not in chosen_positions:
                chosen_positions.append(p)
        for p in range(len(ordered)):
            if len(chosen_positions) >= b0_size:
                break
            if p not in chosen_positions:
                chosen_positions.append(p)
        return [ordered[p] for p in chosen_positions[:b0_size]]

    def initialize(self) -> None:
        b0_size = max(1, int(np.floor(self.b0_region_ratio * self.num_regions)))
        b0_size = min(b0_size, self.total_budget, self.num_regions)
        print(
            f"V6 initialization: regions={self.num_regions}, B0={b0_size}, "
            f"variant={self.visual_variant}, ablation={self.ablation}"
        )
        for rid in self._initial_region_order(b0_size):
            ep = self._representative_episode(rid)
            self._commit_and_reveal(ep, stage="initial", rid=rid)

    def run(self) -> Dict:
        start = time.time()
        self.initialize()
        while len(self.acquired) < self.total_budget:
            rid, score = self._select_region()
            ep = self._target_episode_maximin(rid)
            self._commit_and_reveal(ep, stage="adaptive", rid=rid, region_score=score)

        return {
            "method": "our_v6",
            "causal": True,
            "dataset_name": self.dataset_name,
            "dataset_root": str(self.dataset_root),
            "visual_variant": self.visual_variant,
            "ablation": self.ablation,
            "seed": self.seed,
            "num_regions": self.num_regions,
            "num_selected": len(self.acquired),
            "selected_episode_indices": list(self.acquired),
            "initial_episode_indices": [h["episode_index"] for h in self.history if h["stage"] == "initial"],
            "adaptive_episode_indices": [h["episode_index"] for h in self.history if h["stage"] == "adaptive"],
            "history": self.history,
            "elapsed_seconds": time.time() - start,
            "information_boundary": {
                "pre_acquisition": ["episode_index", "rand_vec"],
                "post_acquisition": ["visual_observations", "actions"],
                "online_pca_fit_scope": "acquired_only",
            },
        }

    def validate_causal_access(self) -> None:
        if set(self.visual_encoder._raw_cache) - self.acquired_set:
            raise AssertionError("Visual cache contains an unacquired episode")
        if set(self.visual_projector.raw) - self.acquired_set:
            raise AssertionError("Visual projector contains an unacquired episode")
        if set(self.action_embeddings) - self.acquired_set:
            raise AssertionError("Action cache contains an unacquired episode")
        if len(self.acquired) != len(self.acquired_set):
            raise AssertionError("Duplicate selected episode")
