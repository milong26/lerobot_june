#!/usr/bin/env python3
"""RoboMME episode selection adapters.

Supported methods:
  random       - seeded random subset.
  grid_uniform - deterministic configuration-space coverage.
  fps          - post-hoc visual farthest-point sampling.
  our_v6       - causal V6 using initial configuration metadata globally and
                 visual/action content only after acquisition.

RoboMME datasets do not use MetaWorld ``rand_vec``. They store the reset-time
configuration in ``episode_initial_states.json -> episodes[*].initial_configuration``.
This module converts that task-specific structure into a stable configuration
matrix without reading trajectory observations/actions/rewards.

Important semantics:
- ``scene_state`` is never used for configuration-space selection.
- dynamic reset quantities such as actor velocity/qvel are excluded.
- entity lists are keyed by semantic ``name`` rather than list position.
- task-level categorical values (for example ``way`` or selected button names)
  are retained through deterministic one-hot encoding.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
ENTITY_FIELDS = ("movable_objects", "randomized_targets", "articulations")
DYNAMIC_ENTITY_KEYS = {
    "velocity",
    "linear_velocity",
    "angular_velocity",
    "qvel",
}
MISSING_CATEGORY = "<MISSING>"


def _safe_token(value: object) -> str:
    text = str(value).strip() or "unnamed"
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)


def _canonicalize_quaternion(value) -> np.ndarray | None:
    """Canonicalize q and -q to one representation before using them as features."""
    try:
        q = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if q.size != 4 or not np.all(np.isfinite(q)):
        return None
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return None
    q = q / norm
    # Sapien quaternions are normally wxyz. More generally, choosing the sign
    # of the first non-zero component removes the q/-q ambiguity deterministically.
    for component in q:
        if abs(float(component)) > 1e-8:
            if component < 0:
                q = -q
            break
    return q.astype(np.float32)


def _flatten_mixed_leaves(
    value,
    prefix: str,
    numeric: Dict[str, float],
    categorical: Dict[str, str],
) -> None:
    """Flatten task-level config leaves while preserving categorical reset factors."""
    if value is None:
        return
    if isinstance(value, bool):
        numeric[prefix] = float(value)
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if np.isfinite(v):
            numeric[prefix] = v
        return
    if isinstance(value, str):
        categorical[prefix] = value
        return
    if isinstance(value, dict):
        for key in sorted(value):
            if key in DYNAMIC_ENTITY_KEYS:
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_mixed_leaves(value[key], child, numeric, categorical)
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            child = f"{prefix}[{i}]"
            _flatten_mixed_leaves(item, child, numeric, categorical)


def _flatten_pose(
    pose: object,
    prefix: str,
    numeric: Dict[str, float],
) -> None:
    if not isinstance(pose, dict):
        return
    position = pose.get("position")
    if position is not None:
        try:
            p = np.asarray(position, dtype=np.float32).reshape(-1)
        except Exception:
            p = np.asarray([], dtype=np.float32)
        if p.size and np.all(np.isfinite(p)):
            for i, value in enumerate(p):
                numeric[f"{prefix}.position[{i}]"] = float(value)
    quaternion = _canonicalize_quaternion(pose.get("quaternion"))
    if quaternion is not None:
        for i, value in enumerate(quaternion):
            numeric[f"{prefix}.quaternion[{i}]"] = float(value)


def _flatten_named_entities(
    field_name: str,
    entities: object,
    numeric: Dict[str, float],
    categorical: Dict[str, str],
) -> None:
    if not isinstance(entities, list):
        return
    seen: Dict[str, int] = {}
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        base_name = _safe_token(entity.get("name") or f"item_{index}")
        occurrence = seen.get(base_name, 0)
        seen[base_name] = occurrence + 1
        entity_name = base_name if occurrence == 0 else f"{base_name}#{occurrence}"
        prefix = f"{field_name}.{entity_name}"

        _flatten_pose(entity.get("pose"), f"{prefix}.pose", numeric)

        # qpos is a reset-time articulation configuration; qvel/velocities are
        # state derivatives rather than the task configuration and stay hidden.
        if "qpos" in entity:
            _flatten_mixed_leaves(entity.get("qpos"), f"{prefix}.qpos", numeric, categorical)

        # Preserve any additional static/reset attributes without depending on
        # list order. Name/pose/dynamic derivatives are handled above.
        for key in sorted(entity):
            if key in {"name", "pose", "qpos"} or key in DYNAMIC_ENTITY_KEYS:
                continue
            _flatten_mixed_leaves(entity[key], f"{prefix}.{key}", numeric, categorical)


def _selection_configuration(initial_configuration: dict) -> dict:
    """Keep only pre-execution task configuration; explicitly exclude scene_state."""
    return {
        key: initial_configuration.get(key)
        for key in CONFIG_FIELDS
        if key in initial_configuration
    }


def _semantic_configuration_features(initial_configuration: dict) -> Tuple[Dict[str, float], Dict[str, str]]:
    cfg = _selection_configuration(initial_configuration)
    numeric: Dict[str, float] = {}
    categorical: Dict[str, str] = {}

    for field in ENTITY_FIELDS:
        _flatten_named_entities(field, cfg.get(field), numeric, categorical)

    task_config = cfg.get("task_config")
    if isinstance(task_config, dict):
        _flatten_mixed_leaves(task_config, "task_config", numeric, categorical)

    return numeric, categorical


def load_configuration_metadata(dataset_root: str) -> Tuple[List[int], np.ndarray, List[str]]:
    """Load RoboMME reset metadata and build a stable mixed-type feature matrix.

    Numeric reset quantities are kept directly. Categorical task configuration
    values are one-hot encoded over the globally visible admissible archive
    support. Numeric missingness receives a separate indicator when it varies.
    Constant columns are removed. All operations use only reset-time metadata.
    """
    path = Path(dataset_root) / "episode_initial_states.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing RoboMME metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes in {path}")

    records: List[Tuple[int, Dict[str, float], Dict[str, str]]] = []
    numeric_keys: set[str] = set()
    categorical_values: Dict[str, set[str]] = {}

    for item in episodes:
        ep = item.get("episode_index")
        cfg = item.get("initial_configuration")
        if ep is None or not isinstance(cfg, dict):
            continue
        numeric, categorical = _semantic_configuration_features(cfg)
        if not numeric and not categorical:
            continue
        ep_i = int(ep)
        records.append((ep_i, numeric, categorical))
        numeric_keys.update(numeric)
        for key, value in categorical.items():
            categorical_values.setdefault(key, set()).add(str(value))

    if not records:
        raise ValueError(
            f"No task-relevant initial_configuration found in {path}; "
            "configuration-based selection cannot run"
        )

    records.sort(key=lambda x: x[0])
    ids = [ep for ep, _, _ in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate episode_index values in {path}")

    columns: List[np.ndarray] = []
    feature_names: List[str] = []
    n = len(records)

    # Numeric features + missingness indicators when the field is not present
    # for every admissible configuration.
    for key in sorted(numeric_keys):
        values = np.full(n, np.nan, dtype=np.float32)
        present = np.zeros(n, dtype=np.float32)
        for row, (_, numeric, _) in enumerate(records):
            if key in numeric and np.isfinite(numeric[key]):
                values[row] = float(numeric[key])
                present[row] = 1.0
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        median = float(np.median(values[finite]))
        values[~finite] = median
        columns.append(values)
        feature_names.append(f"num:{key}")
        if np.any(present == 0.0) and np.any(present == 1.0):
            columns.append(present)
            feature_names.append(f"present:{key}")

    # Categorical task variables are part of the reset configuration too. Use
    # deterministic archive-support one-hot encoding, including missingness.
    all_cat_keys = sorted(categorical_values)
    for key in all_cat_keys:
        observed = set(categorical_values[key])
        if any(key not in categorical for _, _, categorical in records):
            observed.add(MISSING_CATEGORY)
        for category in sorted(observed):
            values = np.zeros(n, dtype=np.float32)
            for row, (_, _, categorical) in enumerate(records):
                current = str(categorical[key]) if key in categorical else MISSING_CATEGORY
                values[row] = 1.0 if current == category else 0.0
            columns.append(values)
            feature_names.append(f"cat:{key}={category}")

    if not columns:
        raise ValueError(f"No usable configuration features could be built from {path}")

    matrix = np.stack(columns, axis=1).astype(np.float32)
    span = matrix.max(axis=0) - matrix.min(axis=0)
    active = span > 1e-8
    if not np.any(active):
        raise ValueError(
            f"All task-relevant configuration features are constant in {path}; "
            "cannot define a configuration-space selector"
        )
    matrix = matrix[:, active]
    keys = [name for name, keep in zip(feature_names, active, strict=True) if bool(keep)]

    task_name = payload.get("task", "unknown")
    print(
        f"[RoboMME config] task={task_name}, episodes={len(ids)}, "
        f"active_dim={matrix.shape[1]}, numeric/categorical reset metadata only"
    )
    print("[RoboMME config] scene_state/velocity/qvel excluded; entities keyed by semantic name")
    return ids, matrix, keys


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
