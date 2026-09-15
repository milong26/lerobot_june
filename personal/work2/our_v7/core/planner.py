"""Paper-faithful causal Our-V7 planner.

This implementation follows the current ICRA method text. Before acquisition,
only reset-configuration metadata is globally visible. Visual observations and
actions are revealed only after a demonstration has been committed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

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
from our_v7.core.action_embedding import reveal_action_embedding
from our_v7.core.visual_embedding import CachedFrozenVLMEpisodeEncoder, CausalVisualProjector


@dataclass
class RegionInfo:
    region_id: int
    member_ids: List[int]
    centroid: np.ndarray


def _dataset_name_to_metaworld_task(dataset_name: str) -> str:
    name = dataset_name
    for suffix in ("_corner", "_top", "_left", "_right", "_front", "_back", "_gripper"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", "-")


def _predefined_metaworld_bounds(dataset_name: str, expected_dim: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read task reset-space bounds from MetaWorld, not from the archive samples."""
    import metaworld

    task_name = _dataset_name_to_metaworld_task(dataset_name)
    mt1 = metaworld.MT1(task_name, seed=0)
    env_cls = mt1.train_classes[task_name]
    env = env_cls()
    try:
        env.set_task(mt1.train_tasks[0])
        space = getattr(env, "_random_reset_space", None)
        if space is None:
            raise RuntimeError(f"MetaWorld task {task_name} has no _random_reset_space")
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    finally:
        try:
            env.close()
        except Exception:
            pass

    if low.shape[0] != expected_dim or high.shape[0] != expected_dim:
        raise ValueError(
            f"Predefined reset-space bound dimension mismatch for {task_name}: "
            f"low/high={low.shape[0]}/{high.shape[0]}, rand_vec={expected_dim}"
        )
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError(f"Non-finite predefined reset bounds for {task_name}")
    if np.any(high <= low):
        bad = np.where(high <= low)[0].tolist()
        raise ValueError(f"Invalid predefined reset bounds for {task_name}, non-positive spans at dims {bad}")
    return low, high, f"MetaWorld:{task_name}._random_reset_space"


class V7Planner:
    """Strict causal acquisition matching the ICRA method specification."""

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

        self.episode_ids, self.raw_configs = self._load_configuration_metadata()
        self.config_low, self.config_high, self.config_bounds_source = _predefined_metaworld_bounds(
            dataset_name, self.raw_configs.shape[1]
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
            repo_id="work2/metaworld_pick_place",
            root=str(self.dataset_root),
        )
        bad = [ep for ep in self.episode_ids if ep < 0 or ep >= dataset.num_episodes]
        if bad:
            raise ValueError(f"episode_initial_states contains invalid episode ids: {bad[:10]}")
        return dataset

    def _load_configuration_metadata(self) -> Tuple[List[int], np.ndarray]:
        path = self.dataset_root / "episode_initial_states.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration metadata: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

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
            raise ValueError(f"No rand_vec configuration metadata found in {path}")
        dims = {x.shape[0] for x in configs}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent rand_vec dimensions: {sorted(dims)}")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate episode_index in episode_initial_states.json")

        order = np.argsort(np.asarray(ids))
        ids = [ids[int(i)] for i in order]
        matrix = np.stack([configs[int(i)] for i in order], axis=0)
        return ids, matrix

    def _normalize_with_predefined_bounds(self, matrix: np.ndarray) -> np.ndarray:
        normalized = (matrix - self.config_low) / (self.config_high - self.config_low)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Non-finite normalized configurations")
        # The archive should lie inside the predefined feasible reset space. Tiny
        # floating errors are tolerated; real out-of-range metadata is rejected.
        tol = 1e-5
        if np.any(normalized < -tol) or np.any(normalized > 1.0 + tol):
            rows, dims = np.where((normalized < -tol) | (normalized > 1.0 + tol))
            examples = [
                (self.episode_ids[int(r)], int(d), float(normalized[r, d]))
                for r, d in zip(rows[:8], dims[:8], strict=False)
            ]
            raise ValueError(
                f"Archive rand_vec lies outside predefined feasible bounds; examples={examples}"
            )
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _deterministic_farthest_support_indices(self, k: int) -> List[int]:
        """Paper KMeans init: centroid-nearest first, then farthest-point traversal."""
        global_centroid = self.configs.mean(axis=0)
        distances_to_global = np.linalg.norm(self.configs - global_centroid, axis=1)
        first = min(
            range(len(self.episode_ids)),
            key=lambda i: (float(distances_to_global[i]), self.episode_ids[i]),
        )
        selected = [first]
        min_dist = np.linalg.norm(self.configs - self.configs[first], axis=1)
        min_dist[first] = -np.inf

        while len(selected) < k:
            best_value = float(np.max(min_dist))
            tied = [
                i for i in range(len(self.episode_ids))
                if i not in selected and abs(float(min_dist[i]) - best_value) <= 1e-12
            ]
            if not tied:
                tied = [i for i in range(len(self.episode_ids)) if i not in selected]
            nxt = min(tied, key=lambda i: self.episode_ids[i])
            selected.append(nxt)
            d = np.linalg.norm(self.configs - self.configs[nxt], axis=1)
            min_dist = np.minimum(min_dist, d)
            for idx in selected:
                min_dist[idx] = -np.inf
        return selected

    def _build_regions(self) -> Tuple[Dict[int, RegionInfo], Dict[int, int], List[int]]:
        m = len(self.episode_ids)
        k = min(m, max(self.min_regions, min(self.max_regions, int(np.floor(self.region_ratio * m)))))
        k = max(1, k)

        init_rows = self._deterministic_farthest_support_indices(k)
        init_centers = np.stack([self.configs[i] for i in init_rows], axis=0)
        init_episode_ids = [self.episode_ids[i] for i in init_rows]

        km = KMeans(
            n_clusters=k,
            init=init_centers,
            n_init=1,
            algorithm="lloyd",
        )
        labels = km.fit_predict(self.configs)

        regions: Dict[int, RegionInfo] = {}
        ep_to_region: Dict[int, int] = {}
        for rid in range(k):
            members = [self.episode_ids[i] for i, lab in enumerate(labels) if int(lab) == rid]
            if not members:
                raise RuntimeError(f"Deterministic KMeans produced empty region {rid}")
            regions[rid] = RegionInfo(
                region_id=rid,
                member_ids=sorted(members),
                centroid=km.cluster_centers_[rid].astype(np.float32),
            )
            for ep in members:
                ep_to_region[ep] = rid
        return regions, ep_to_region, init_episode_ids

    def _config(self, ep: int) -> np.ndarray:
        return self.configs[self.id_to_row[ep]]

    def _unused_in_region(self, rid: int) -> List[int]:
        return [ep for ep in self.regions[rid].member_ids if ep not in self.acquired_set]

    def _global_unused(self) -> List[int]:
        return [ep for ep in self.episode_ids if ep not in self.acquired_set]

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
            return {rid: 0.0 for rid in values}
        return {rid: (float(v) - lo) / (hi - lo) for rid, v in values.items()}

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
            # Paper Eq.: g_k = 1 - n_k / N_k. Do NOT min-max normalize g_k.
            coverage[rid] = 1.0 - len(acquired_here) / len(region.member_ids)
            visual[rid] = self._mean_pairwise(
                [self.visual_embeddings[ep] for ep in acquired_here if ep in self.visual_embeddings]
            )
            action[rid] = self._mean_pairwise(
                [self.action_embeddings[ep] for ep in acquired_here if ep in self.action_embeddings]
            )

        visual_norm = self._minmax(visual)
        action_norm = self._minmax(action)
        result: Dict[int, Dict[str, float]] = {}
        for rid in active:
            if self.ablation == "wo_action":
                priority = coverage[rid] + visual_norm[rid]
            else:
                # Paper Eq.: p_k = g_k + u_v_bar + u_a_bar.
                priority = coverage[rid] + visual_norm[rid] + action_norm[rid]
            result[rid] = {
                "coverage_gap": coverage[rid],
                "visual_heterogeneity": visual[rid],
                "action_heterogeneity": action[rid],
                "visual_norm": visual_norm[rid],
                "action_norm": action_norm[rid],
                "priority": priority,
            }
        return result

    def _select_region(self) -> Tuple[int, Dict[str, float]]:
        scores = self._region_scores()
        if not scores:
            raise RuntimeError("No non-exhausted region remains")

        if self.ablation == "wo_adaptive_priority":
            for _ in range(self.num_regions + 1):
                rid = self._rr_cursor % self.num_regions
                self._rr_cursor += 1
                if rid in scores:
                    return rid, scores[rid]
            rid = min(scores)
            return rid, scores[rid]

        rid = max(scores, key=lambda r: (scores[r]["priority"], -r))
        return rid, scores[rid]

    def _nearest_unused_to_target(self, target: np.ndarray, rid: int) -> Tuple[int, str]:
        candidates = self._unused_in_region(rid)
        scope = "selected_region"
        if not candidates:
            # Paper fallback; normally unreachable because exhausted regions are
            # excluded from region selection.
            candidates = self._global_unused()
            scope = "global_fallback"
        if not candidates:
            raise RuntimeError("Archive exhausted before budget completion")
        ep = min(
            candidates,
            key=lambda x: (float(np.linalg.norm(self._config(x) - target)), x),
        )
        return ep, scope

    def _target_for_region(self, rid: int) -> np.ndarray:
        region = self.regions[rid]
        acquired_here = [ep for ep in region.member_ids if ep in self.acquired_set]
        if not acquired_here:
            # Paper: use region centroid when the region has not produced a demo.
            return region.centroid.copy()

        candidates = self._unused_in_region(rid)
        if not candidates:
            return region.centroid.copy()
        acquired_matrix = np.stack([self._config(ep) for ep in self.acquired], axis=0)

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
        return self._config(best_ep).copy()

    def _initial_region_order(self, b0_size: int) -> List[int]:
        """Paper B0: global-centroid-nearest region, then farthest traversal."""
        centroids = {rid: region.centroid for rid, region in self.regions.items()}
        global_centroid = self.configs.mean(axis=0)
        first = min(
            centroids,
            key=lambda rid: (float(np.linalg.norm(centroids[rid] - global_centroid)), rid),
        )
        selected = [first]
        while len(selected) < b0_size:
            remaining = [rid for rid in sorted(centroids) if rid not in selected]
            nxt = max(
                remaining,
                key=lambda rid: (
                    min(float(np.linalg.norm(centroids[rid] - centroids[s])) for s in selected),
                    -rid,
                ),
            )
            selected.append(nxt)
        return selected

    def _refresh_visual_dict(self) -> None:
        self.visual_embeddings = self.visual_projector.as_dict()

    def _commit_and_reveal(
        self,
        ep: int,
        stage: str,
        rid: int,
        target_config: np.ndarray,
        mapping_scope: str,
        region_score: Optional[Dict[str, float]] = None,
    ) -> None:
        if ep in self.acquired_set:
            raise RuntimeError(f"Duplicate acquisition attempted for episode {ep}")

        # Commit FIRST. Only then can image/action content be accessed.
        self.acquired.append(ep)
        self.acquired_set.add(ep)

        raw_visual = self.visual_encoder.extract(ep, self.acquired_set)
        self.visual_projector.add(ep, raw_visual)
        self._refresh_visual_dict()
        self.action_embeddings[ep] = reveal_action_embedding(self.dataset, ep, self.acquired_set)

        entry = {
            "step": len(self.acquired),
            "stage": stage,
            "episode_index": ep,
            "region_id": rid,
            "target_config": np.asarray(target_config, dtype=np.float32).tolist(),
            "acquired_config": self._config(ep).tolist(),
            "archive_mapping_scope": mapping_scope,
            "visual_variant": self.visual_variant,
            "pca_components": dict(self.visual_projector.last_pca_components),
            "ablation": self.ablation,
        }
        if region_score is not None:
            entry.update(region_score)
        self.history.append(entry)
        print(
            f"[{len(self.acquired):03d}/{self.total_budget}] {stage}: "
            f"region={rid}, ep={ep}, variant={self.visual_variant}, map={mapping_scope}"
        )

    def initialize(self) -> None:
        b0_size = max(1, int(np.floor(self.b0_region_ratio * self.num_regions)))
        b0_size = min(b0_size, self.total_budget, self.num_regions)
        order = self._initial_region_order(b0_size)
        print(
            f"V7 paper initialization: regions={self.num_regions}, B0={b0_size}, "
            f"variant={self.visual_variant}, ablation={self.ablation}"
        )
        print(f"B0 region order: {order}")
        for rid in order:
            target = self.regions[rid].centroid.copy()
            ep, scope = self._nearest_unused_to_target(target, rid)
            self._commit_and_reveal(
                ep,
                stage="initial",
                rid=rid,
                target_config=target,
                mapping_scope=scope,
            )

    def run(self) -> Dict:
        start = time.time()
        self.initialize()
        while len(self.acquired) < self.total_budget:
            rid, score = self._select_region()
            target = self._target_for_region(rid)
            ep, scope = self._nearest_unused_to_target(target, rid)
            self._commit_and_reveal(
                ep,
                stage="adaptive",
                rid=rid,
                target_config=target,
                mapping_scope=scope,
                region_score=score,
            )

        return {
            "method": "our_v7",
            "paper_faithful_selection": True,
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
            "configuration_normalization": {
                "type": "predefined_feasible_bounds",
                "source": self.config_bounds_source,
                "low": self.config_low.tolist(),
                "high": self.config_high.tolist(),
            },
            "kmeans_initialization": {
                "type": "deterministic_centroid_then_farthest_point",
                "support_episode_ids": self.kmeans_init_episode_ids,
            },
            "region_priority": "g_k + minmax(u_visual) + minmax(u_action)",
            "information_boundary": {
                "pre_acquisition": ["episode_index", "rand_vec", "predefined_reset_bounds"],
                "post_acquisition": ["visual_observations", "actions"],
                "online_pca_fit_scope": "acquired_only",
            },
            "raw_visual_cache": str(self.visual_encoder.cache_dir),
        }

    def validate_causal_access(self) -> None:
        if set(self.visual_encoder._raw_cache) - self.acquired_set:
            raise AssertionError("Visual in-memory cache contains an unacquired episode")
        if set(self.visual_projector.raw) - self.acquired_set:
            raise AssertionError("Visual projector contains an unacquired episode")
        if set(self.action_embeddings) - self.acquired_set:
            raise AssertionError("Action cache contains an unacquired episode")
        if len(self.acquired) != len(self.acquired_set):
            raise AssertionError("Duplicate acquired episode")
