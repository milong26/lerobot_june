# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-agnostic visualization dispatch.

Selects a visualization backend at runtime via a display-mode string (e.g. a ``--display_mode`` CLI
flag) so callers never branch on the backend. The concrete implementations live in
:mod:`lerobot.utils.rerun_visualization` and :mod:`lerobot.utils.foxglove_visualization`; importing
this module does not import ``rerun`` or ``foxglove`` (each backend imports its SDK lazily behind a
``require_package`` guard).
"""

from lerobot.lerobot_types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package
from .foxglove_visualization import init_foxglove, log_foxglove_data, shutdown_foxglove
from .rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun

# Visualization backends selectable at runtime via a display-mode string (e.g. a --display_mode flag).
VISUALIZATION_MODES = ("rerun", "foxglove")


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
) -> None:
    """Initializes the visualization backend selected by ``display_mode``.

    For ``"rerun"``, ``ip``/``port`` point at an optional remote Rerun server. For ``"foxglove"``,
    ``ip`` is the interface to bind the WebSocket server to (``127.0.0.1`` for local only, ``0.0.0.0``
    for all interfaces) and ``port`` is its port.
    """

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(host=ip or "127.0.0.1", port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )

WOWSKIN_FORCE_SCALING = 7.0
WOWSKIN_DISPLAY_WIDTH = 400
WOWSKIN_CHIP_LOCATIONS = np.array(
    [
        [201.0, 238.0],
        [126.0, 238.0],
        [275.0, 238.0],
        [201.0, 163.0],
        [201.0, 312.0],
    ],
    dtype=np.float32,
)
WOWSKIN_CHIP_ROTATIONS = np.array(
    [-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0],
    dtype=np.float32,
)

def _extract_force_vectors(item: object) -> np.ndarray | None:
    if isinstance(item, torch.Tensor):
        item_np = item.detach().cpu().numpy()
    else:
        item_np = np.asarray(item)

    if item_np.ndim == 1:
        if item_np.size % 3 == 0:
            return item_np.reshape(-1, 3)
        if item_np.size % 4 == 0:
            return item_np.reshape(-1, 4)[..., :3]
        return None

    if item_np.ndim == 2:
        if item_np.shape[-1] == 3:
            return item_np
        if item_np.shape[-1] >= 4:
            return item_np[..., :3]

    return None

def _log_wowskin_force(rr, item: object) -> None:
    vecs = _extract_force_vectors(item)
    if vecs is None or vecs.size == 0:
        return

    chip_locations = WOWSKIN_CHIP_LOCATIONS[: vecs.shape[0]]
    chip_rotations = WOWSKIN_CHIP_ROTATIONS[: vecs.shape[0]]

    positions: list[list[float]] = []
    origins: list[list[float]] = []
    vectors: list[list[float]] = []
    colors: list[list[int]] = []
    radii: list[float] = []

    for sidx, (cx, cy) in enumerate(chip_locations):
        vx, vy, vz = vecs[sidx]
        rot = chip_rotations[sidx]
        c = np.cos(rot)
        s = np.sin(rot)

        xy = np.array([c * vx - s * vy, s * vx + c * vy], dtype=np.float32)

        positions.append([float(cx), float(cy)])
        origins.append([float(cx), float(cy)])
        vectors.append(
            [
                float(xy[0] / WOWSKIN_FORCE_SCALING),
                float(xy[1] / WOWSKIN_FORCE_SCALING),
            ]
        )
        radii.append(float(abs(vz) / WOWSKIN_FORCE_SCALING))
        colors.append([255, 0, 0] if vz >= 0 else [0, 0, 255])

    rr.log(
        "game_force/points",
        rr.Points2D(
            positions=positions,
            colors=colors,
            radii=rr.Radius.ui_points(radii),
        ),
    )
    rr.log(
        "game_force/arrows",
        rr.Arrows2D(
            origins=origins,
            vectors=vectors,
            colors=[[0, 255, 0]] * len(vectors),
            radii=rr.Radius.ui_points(2.0),
        ),
    )




def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)
        # ================= FORCE VISUALIZATION =================
        force_values = [
            (k, v)
            for k, v in observation.items()
            if "force" in str(k) and v is not None
        ]

        if force_values:
            force_values.sort(key=lambda x: int(str(x[0]).split("_")[-1]))
            force_array = np.asarray([v for _, v in force_values], dtype=np.float32)
            _log_wowskin_force(rr, force_array)
        # ================= END FORCE VISUALIZATION =================

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


def log_visualization_data(
    display_mode: str,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Logs observation/action data to the backend selected by ``display_mode``."""

    if display_mode == "rerun":
        log_rerun_data(observation=observation, action=action, compress_images=compress_images)
    elif display_mode == "foxglove":
        log_foxglove_data(observation=observation, action=action, compress_images=compress_images)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def shutdown_visualization(display_mode: str) -> None:
    """Shuts down the backend selected by ``display_mode``."""

    if display_mode == "rerun":
        shutdown_rerun()
    elif display_mode == "foxglove":
        shutdown_foxglove()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")

