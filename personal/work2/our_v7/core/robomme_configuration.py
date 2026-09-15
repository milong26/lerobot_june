"""Paper-faithful RoboMME reset-configuration parser for Our-V7.

Only reset-time task configuration is visible before acquisition. The parser
therefore reads ``episode_initial_states.json -> episodes[*].initial_configuration``
and deliberately ignores trajectory observations/actions/rewards as well as the
full low-level ``scene_state`` snapshot.

The three Main Results RoboMME tasks use heterogeneous reset metadata. This
module turns that metadata into one deterministic Euclidean configuration space:

- entity collections are keyed by semantic ``name`` and sorted by name;
- pose coordinates use semantic axes (x/y/z and qw/qx/qy/qz);
- velocity/qvel and ``scene_state`` are excluded;
- articulation qpos is kept as reset configuration;
- task categorical factors are deterministic one-hot variables;
- ordered categorical sequences such as selected_button_names keep their slot
  semantics (slot_00, slot_01, ...), rather than being treated as anonymous
  object-list positions;
- constant columns are removed after encoding;
- the final active feature schema is returned for selection-result auditing.

All vocabulary/support discovery below uses reset metadata only, which is allowed
by the Our-V7 causal boundary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


CONFIG_FIELDS = (
    "movable_objects",
    "randomized_targets",
    "articulations",
    "task_config",
)
ENTITY_FIELDS = ("movable_objects", "randomized_targets", "articulations")
DYNAMIC_KEYS = {
    "velocity",
    "linear_velocity",
    "angular_velocity",
    "qvel",
}
MISSING_CATEGORY = "<MISSING>"

# Integer-valued factors with categorical, not metric, semantics in the three
# RoboMME Main Results tasks. String-valued factors (difficulty, way, etc.) are
# categorical automatically.
CATEGORICAL_INTEGER_TASK_FIELDS = {
    "task_config.direction",
    "task_config.direction2",
    "task_config.obj_flag",
}


def _safe_token(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("RoboMME entity name must be non-empty")
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)


def _canonicalize_quaternion(value: object) -> np.ndarray | None:
    """Normalize a quaternion and remove the q/-q representation ambiguity."""
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
    for component in q:
        if abs(float(component)) > 1e-8:
            if component < 0:
                q = -q
            break
    return q.astype(np.float32)


def _store_numeric(numeric: Dict[str, float], key: str, value: object) -> None:
    try:
        v = float(value)
    except Exception:
        return
    if np.isfinite(v):
        numeric[key] = v


def _flatten_pose(pose: object, prefix: str, numeric: Dict[str, float]) -> None:
    if not isinstance(pose, dict):
        return

    position = pose.get("position")
    if position is not None:
        try:
            p = np.asarray(position, dtype=np.float32).reshape(-1)
        except Exception:
            p = np.asarray([], dtype=np.float32)
        if p.size and np.all(np.isfinite(p)):
            axes = ("x", "y", "z") if p.size == 3 else tuple(f"axis_{i:02d}" for i in range(p.size))
            for axis, value in zip(axes, p, strict=True):
                numeric[f"{prefix}.position.{axis}"] = float(value)

    quaternion = _canonicalize_quaternion(pose.get("quaternion"))
    if quaternion is not None:
        # Sapien serializes Pose.q as wxyz.
        for axis, value in zip(("qw", "qx", "qy", "qz"), quaternion, strict=True):
            numeric[f"{prefix}.quaternion.{axis}"] = float(value)


def _flatten_qpos(value: object, prefix: str, numeric: Dict[str, float]) -> None:
    try:
        qpos = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return
    if not qpos.size or not np.all(np.isfinite(qpos)):
        return
    for i, item in enumerate(qpos):
        numeric[f"{prefix}.dof_{i:02d}"] = float(item)


def _flatten_generic_static(
    value: object,
    prefix: str,
    numeric: Dict[str, float],
    categorical: Dict[str, str],
) -> None:
    """Flatten non-task static entity attributes without anonymous entity indexing."""
    if value is None:
        return
    if isinstance(value, bool):
        numeric[prefix] = float(value)
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        _store_numeric(numeric, prefix, value)
        return
    if isinstance(value, str):
        categorical[prefix] = value
        return
    if isinstance(value, dict):
        for key in sorted(value):
            if key in DYNAMIC_KEYS:
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_generic_static(value[key], child, numeric, categorical)
        return
    if isinstance(value, (list, tuple)):
        # A list here is not an entity collection; its order is part of the
        # serialized field itself. Use explicit element labels for auditability.
        for i, item in enumerate(value):
            _flatten_generic_static(item, f"{prefix}.element_{i:02d}", numeric, categorical)


def _flatten_named_entities(
    field_name: str,
    entities: object,
    numeric: Dict[str, float],
    categorical: Dict[str, str],
) -> None:
    if entities is None:
        return
    if not isinstance(entities, list):
        raise ValueError(f"{field_name} must be a list, got {type(entities).__name__}")

    canonical_entities: List[Tuple[str, dict]] = []
    seen: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError(f"{field_name} contains a non-dict entity")
        if "name" not in entity or entity.get("name") is None:
            raise ValueError(
                f"{field_name} contains an entity without semantic name; "
                "V7 refuses list-position identity"
            )
        name = _safe_token(entity["name"])
        if name in seen:
            raise ValueError(
                f"Duplicate semantic entity name {name!r} in {field_name}; "
                "V7 refuses order-dependent duplicate disambiguation"
            )
        seen.add(name)
        canonical_entities.append((name, entity))

    # Input list ordering cannot change the encoded schema.
    canonical_entities.sort(key=lambda item: item[0])

    for name, entity in canonical_entities:
        prefix = f"{field_name}.{name}"
        _flatten_pose(entity.get("pose"), f"{prefix}.pose", numeric)
        if "qpos" in entity:
            _flatten_qpos(entity.get("qpos"), f"{prefix}.qpos", numeric)

        for key in sorted(entity):
            if key in {"name", "pose", "qpos"} or key in DYNAMIC_KEYS:
                continue
            _flatten_generic_static(entity[key], f"{prefix}.{key}", numeric, categorical)


def _flatten_task_config(
    value: object,
    prefix: str,
    numeric: Dict[str, float],
    categorical: Dict[str, str],
) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        # Binary factors have an unambiguous 0/1 metric and are retained.
        numeric[prefix] = float(value)
        return
    if isinstance(value, (np.integer, int)):
        if prefix in CATEGORICAL_INTEGER_TASK_FIELDS:
            categorical[prefix] = str(int(value))
        else:
            numeric[prefix] = float(value)
        return
    if isinstance(value, (np.floating, float)):
        _store_numeric(numeric, prefix, value)
        return
    if isinstance(value, str):
        categorical[prefix] = value
        return
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_task_config(value[key], child, numeric, categorical)
        return
    if isinstance(value, (list, tuple)):
        # selected_button_names in PatternLock/RouteStick is an ordered task
        # sequence. Preserve order explicitly as semantic slots and one-hot each
        # slot, rather than treating the list as a collection of unnamed objects.
        for i, item in enumerate(value):
            child = f"{prefix}.slot_{i:02d}"
            if isinstance(item, str):
                categorical[child] = item
            else:
                _flatten_task_config(item, child, numeric, categorical)
        return


def _semantic_features(initial_configuration: dict) -> Tuple[Dict[str, float], Dict[str, str]]:
    numeric: Dict[str, float] = {}
    categorical: Dict[str, str] = {}

    for field in ENTITY_FIELDS:
        _flatten_named_entities(field, initial_configuration.get(field), numeric, categorical)

    task_config = initial_configuration.get("task_config")
    if task_config is not None:
        if not isinstance(task_config, dict):
            raise ValueError("initial_configuration.task_config must be a dict")
        _flatten_task_config(task_config, "task_config", numeric, categorical)

    return numeric, categorical


def load_v7_robomme_configuration_metadata(
    dataset_root: str,
) -> Tuple[List[int], np.ndarray, List[str], dict]:
    """Build the V7 configuration matrix using reset metadata only."""
    path = Path(dataset_root) / "episode_initial_states.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing RoboMME metadata: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No episodes list in {path}")

    records: List[Tuple[int, Dict[str, float], Dict[str, str]]] = []
    numeric_keys: set[str] = set()
    categorical_values: Dict[str, set[str]] = {}

    for item in episodes:
        if not isinstance(item, dict):
            continue
        ep = item.get("episode_index")
        cfg = item.get("initial_configuration")
        if ep is None or not isinstance(cfg, dict):
            continue

        # Only the explicitly allowed reset-time fields are forwarded. In
        # particular scene_state is never passed to the semantic encoder.
        selection_cfg = {field: cfg.get(field) for field in CONFIG_FIELDS if field in cfg}
        numeric, categorical = _semantic_features(selection_cfg)
        if not numeric and not categorical:
            continue

        ep_i = int(ep)
        records.append((ep_i, numeric, categorical))
        numeric_keys.update(numeric)
        for key, value in categorical.items():
            categorical_values.setdefault(key, set()).add(str(value))

    if not records:
        raise ValueError(
            f"No usable task-level initial_configuration found in {path}; "
            "Our-V7 cannot construct configuration space"
        )

    records.sort(key=lambda item: item[0])
    ids = [ep for ep, _, _ in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate episode_index values in {path}")

    columns: List[np.ndarray] = []
    feature_names: List[str] = []
    n = len(records)

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
        # Missing optional-object coordinates are represented by a presence bit;
        # median fill prevents an arbitrary numeric extreme from being invented.
        fill = float(np.median(values[finite]))
        values[~finite] = fill
        columns.append(values)
        feature_names.append(f"num:{key}")
        if np.any(present == 0.0) and np.any(present == 1.0):
            columns.append(present)
            feature_names.append(f"present:{key}")

    for key in sorted(categorical_values):
        vocabulary = set(categorical_values[key])
        if any(key not in categorical for _, _, categorical in records):
            vocabulary.add(MISSING_CATEGORY)
        for category in sorted(vocabulary):
            values = np.zeros(n, dtype=np.float32)
            for row, (_, _, categorical) in enumerate(records):
                current = str(categorical[key]) if key in categorical else MISSING_CATEGORY
                values[row] = 1.0 if current == category else 0.0
            columns.append(values)
            feature_names.append(f"cat:{key}={category}")

    if not columns:
        raise ValueError(f"No usable V7 configuration features could be built from {path}")

    matrix = np.stack(columns, axis=1).astype(np.float32)
    span = matrix.max(axis=0) - matrix.min(axis=0)
    active = span > 1e-8
    if not np.any(active):
        raise ValueError(
            f"All V7 RoboMME configuration features are constant in {path}; "
            "configuration-space acquisition is undefined"
        )

    matrix = matrix[:, active]
    active_keys = [name for name, keep in zip(feature_names, active, strict=True) if bool(keep)]
    dropped_keys = [name for name, keep in zip(feature_names, active, strict=True) if not bool(keep)]

    schema = {
        "source": "episode_initial_states.json/episodes[*].initial_configuration",
        "allowed_top_level_fields": list(CONFIG_FIELDS),
        "entity_identity": "semantic name; input list order ignored; duplicate names rejected",
        "pose_axes": {
            "position": ["x", "y", "z"],
            "quaternion": ["qw", "qx", "qy", "qz"],
            "quaternion_rule": "normalized and sign-canonicalized",
        },
        "articulation_state": "qpos only, indexed as dof_XX",
        "categorical_encoding": "deterministic archive-support one-hot",
        "categorical_integer_task_fields": sorted(CATEGORICAL_INTEGER_TASK_FIELDS),
        "ordered_task_sequences": "slot_XX categorical encoding",
        "numeric_missingness": "median fill plus presence indicator when needed",
        "excluded": ["scene_state", "velocity", "linear_velocity", "angular_velocity", "qvel"],
        "active_feature_keys": active_keys,
        "dropped_constant_feature_keys": dropped_keys,
        "active_dim": int(matrix.shape[1]),
    }

    task_name = payload.get("task", "unknown")
    print(
        f"[Our-V7 RoboMME config] task={task_name}, episodes={len(ids)}, "
        f"active_dim={matrix.shape[1]}"
    )
    print(
        "[Our-V7 RoboMME config] semantic-name entities; categorical one-hot; "
        "scene_state/velocity/qvel excluded"
    )
    return ids, matrix, active_keys, schema


__all__ = ["load_v7_robomme_configuration_metadata"]
