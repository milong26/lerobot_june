#!/usr/bin/env python3
"""RoboMME episode selection adapters.

Supported methods:
  random       - seeded random subset.
  grid_uniform - deterministic configuration-space coverage.
  fps          - post-hoc visual farthest-point sampling.
  our_v6       - causal V6 using initial configuration metadata globally and
                 visual/action content only after acquisition.

RoboMME datasets store task-specific reset metadata under
``initial_configuration`` rather than MetaWorld's ``rand_vec``. This module
turns the task-relevant numeric reset metadata into a stable per-episode
configuration vector without reading trajectory observations or actions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

WORK2_ROOT = Path(__file__).resolve().parents[1]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

CONFIG_FIELDS = (
    "movable_objects",
    "randomized_targets",
    "articulations",
    "task_config",
)


def _numeric_leaves(value, prefix: str = "") -> Dict[str, float]:
    """Flatten numeric leaves from nested task configuration metadata."""
    out: Dict[str, float] = {}
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if np.isfinite(v):
            out[prefix or "value"] = v
        return out
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_numeric_leaves(value[key], child))
        return out
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            child = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(_numeric_leaves(item, child))
    return out


def _selection_configuration(initial_configuration: dict) -> dict:
    """Keep task-level reset factors and explicitly exclude full scene_state."""
    return {
        key: initial_configuration.get(key)
        for key in CONFIG_FIELDS
        if key in initial_configuration
    }


def load_configuration_metadata(dataset_root: str) -> Tuple[List[int], np.ndarray, List[str]]:
    """Load RoboMME task configuration and build a stable numeric matrix."""
    path = Path(dataset_root) / "episode_initial_states.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing RoboMME metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes in {path}")

    rows: List[Tuple[int, Dict[str, float]]] = []
    all_keys = set()
    for item in episodes:
        ep = item.get("episode_index")
        cfg = item.get("initial_configuration")
        if ep is None or not isinstance(cfg, dict):
            continue
        flat = _numeric_leaves(_selection_configuration(cfg))
        if not flat:
            continue
        rows.append((int(ep), flat))
        all_keys.update(flat)

    if not rows:
        raise ValueError(
            f"No numeric task configuration found in {path}; configuration-based selection cannot run"
        )

    rows.sort(key=lambda x: x[0])
    keys = sorted(all_keys)
    matrix = np.full((len(rows), len(keys)), np.nan, dtype=np.float32)
    for r, (_, flat) in enumerate(rows):
        for c, key in enumerate(keys):
            if key in flat:
                matrix[r, c] = flat[key]

    med = np.nanmedian(matrix, axis=0)
    inds = np.where(~np.isfinite(matrix))
    matrix[inds] = med[inds[1]]

    span = matrix.max(axis=0) - matrix.min(axis=0)
    active = span > 1e-8
    if not np.any(active):
        raise ValueError("All task configuration dimensions are constant")
    matrix = matrix[:, active]
    keys = [k for k, keep in zip(keys, active, strict=True) if bool(keep)]
    ids = [ep for ep, _ in rows]
    return ids, matrix.astype(np.float32), keys


def minmax(matrix: np.ndarray) -> np.ndarray:
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return ((matrix - lo) / span).astype(np.float32)


def _validate_budget(ids: List[int], n: int) -> None:
    if n < 1:
        raise ValueError("num_episodes must be >= 1")
    if n > len(ids):
        raise ValueError(f"Requested {n} episodes but only {len(ids)} are available")


def select_random(ids: List[int], n: int, seed: int) -> List[int]:
    _validate_budget(ids, n)
    rng = random.Random(seed)
    return sorted(rng.sample(ids, n))


def farthest_point_indices(features: np.ndarray, n: int) -> List[int]:
    """Deterministic FPS: start closest to feature-space center, then maximin."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f"Expected [N,D] features, got {features.shape}")
    if n > len(features):
        raise ValueError(f"Requested {n} points from only {len(features)} features")

    center = features.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(features - center, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(features - features[first], axis=1)
    min_dist[first] = -np.inf

    while len(selected) < n:
        best = int(np.argmax(min_dist))
        selected.append(best)
        d = np.linalg.norm(features - features[best], axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[selected] = -np.inf
    return selected


def select_grid_uniform(ids: List[int], configs: np.ndarray, n: int) -> List[int]:
    """Uniform configuration coverage for arbitrary RoboMME task configurations."""
    _validate_budget(ids, n)
    normalized = minmax(configs)
    rows = farthest_point_indices(normalized, n)
    return sorted(ids[i] for i in rows)


def _load_dataset(dataset_root: str, dataset_name: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=f"work2/robomme_{dataset_name}", root=dataset_root)


def select_visual_fps(
    dataset_root: str,
    dataset_name: str,
    ids: List[int],
    n: int,
    device: str,
) -> List[int]:
    """Post-hoc visual FPS using branch-wise L2 frozen-VLM features."""
    _validate_budget(ids, n)
    from our_v6.core.visual_embedding import CausalVisualProjector, FrozenVLMEpisodeEncoder

    dataset = _load_dataset(dataset_root, dataset_name)
    encoder = FrozenVLMEpisodeEncoder(dataset, dataset_name=dataset_name, device=device)
    projector = CausalVisualProjector(variant="l2")

    # FPS is intentionally a post-hoc full-information baseline. Mark the full
    # candidate set visible to the encoder, then compute only visual features.
    visible = set(ids)
    features = []
    for pos, ep in enumerate(ids, start=1):
        raw = encoder.extract(ep, visible)
        projector.add(ep, raw)
        features.append(projector.get(ep))
        print(f"[FPS visual] {pos}/{len(ids)} episode={ep}")
    rows = farthest_point_indices(np.stack(features, axis=0), n)
    return sorted(ids[i] for i in rows)


class RoboMMEV6PlannerMixin:
    """Overrides only metadata/dataset adapters of the generic causal V6 planner."""

    def _load_configuration_metadata(self):
        ids, matrix, _ = load_configuration_metadata(str(self.dataset_root))
        return ids, matrix

    def _load_dataset(self):
        dataset = _load_dataset(str(self.dataset_root), self.dataset_name)
        bad = [ep for ep in self.episode_ids if ep < 0 or ep >= dataset.num_episodes]
        if bad:
            raise ValueError(f"Configuration metadata contains invalid episode ids: {bad[:10]}")
        return dataset


def select_our_v6(
    dataset_root: str,
    dataset_name: str,
    n: int,
    seed: int,
    device: str,
    visual_variant: str,
) -> Tuple[List[int], dict]:
    from our_v6.core.planner import V6Planner

    class RoboMMEV6Planner(RoboMMEV6PlannerMixin, V6Planner):
        pass

    planner = RoboMMEV6Planner(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        total_budget=n,
        visual_variant=visual_variant,
        device=device,
        seed=seed,
    )
    result = planner.run()
    planner.validate_causal_access()
    return list(result["selected_episode_indices"]), result


def save_result(
    output_file: str,
    method: str,
    dataset_name: str,
    selected: Iterable[int],
    seed: int,
    extra: dict | None = None,
) -> None:
    selected = [int(x) for x in selected]
    if len(selected) != len(set(selected)):
        raise RuntimeError("Selector produced duplicate episode indices")
    payload = {
        "method": method,
        "dataset_name": dataset_name,
        "seed": int(seed),
        "num_selected": len(selected),
        "selected_episode_indices": selected,
    }
    if extra:
        payload["details"] = extra
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved selection: {path} ({len(selected)} episodes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select RoboMME episodes")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method", required=True, choices=["random", "grid_uniform", "fps", "our_v6"])
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual-variant", choices=["l2", "online_pca"], default="l2")
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    if args.method == "random":
        dataset = _load_dataset(args.dataset_root, args.dataset_name)
        all_ids = list(range(dataset.num_episodes))
        selected = select_random(all_ids, args.num_episodes, args.seed)
        details = {"available_episodes": len(all_ids)}
    elif args.method == "grid_uniform":
        ids, configs, config_keys = load_configuration_metadata(args.dataset_root)
        selected = select_grid_uniform(ids, configs, args.num_episodes)
        details = {"configuration_dim": int(configs.shape[1]), "configuration_keys": config_keys}
    elif args.method == "fps":
        dataset = _load_dataset(args.dataset_root, args.dataset_name)
        all_ids = list(range(dataset.num_episodes))
        selected = select_visual_fps(
            args.dataset_root, args.dataset_name, all_ids, args.num_episodes, args.device
        )
        details = {"feature": "branch-wise-l2-frozen-vlm", "information_access": "posthoc-full-pool"}
    else:
        selected, v6_result = select_our_v6(
            args.dataset_root,
            args.dataset_name,
            args.num_episodes,
            args.seed,
            args.device,
            args.visual_variant,
        )
        details = {
            "visual_variant": args.visual_variant,
            "causal": True,
            "num_regions": v6_result.get("num_regions"),
            "history": v6_result.get("history", []),
        }

    if len(selected) != args.num_episodes:
        raise RuntimeError(f"Selected {len(selected)} episodes, requested {args.num_episodes}")
    save_result(args.output_file, args.method, args.dataset_name, selected, args.seed, details)


if __name__ == "__main__":
    main()
