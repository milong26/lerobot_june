#!/usr/bin/env python

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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Requires: pip install 'lerobot[dataset_viz]'  (includes dataset + viz extras)

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876

local$ rerun rerun+http://IP:GRPC_PORT/proxy
```

- Visualize data in Foxglove with a seekable, scrubbable timeline:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --display-mode foxglove

# then open the Foxglove app and connect to ws://127.0.0.1:8765
```
This starts a Foxglove WebSocket server that serves the episode on demand from the on-disk dataset,
so you can play/pause and scrub anywhere in the episode using Foxglove's playback controls.

"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import tqdm

from lerobot.configs import DEPTH_MILLIMETER_UNIT
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_FORCE, OBS_STATE, OBS_STR, REWARD, SUCCESS
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

DEFAULT_FOXGLOVE_PORT = 8765
DEFAULT_RERUN_PORT = 9090


def get_feature_names(dataset: LeRobotDataset, key: str) -> list[str]:
    """Return per-dimension names for a feature from the dataset metadata.

    Only flat-list ``names`` metadata is used. Dict-style ``names`` and missing names fall back to ``{key}_{i}`` indices.
    """
    feature = dataset.features[key]
    dim = feature["shape"][-1]

    names = feature.get("names")
    if isinstance(names, list) and len(names) == dim:
        return [str(name) for name in names]

    return [f"{key}_{d}" for d in range(dim)]


def check_chw_float32(frame: torch.Tensor) -> None:
    """
    Check if a frame is a channel-first, float32 tensor.
    """
    assert frame.dtype == torch.float32
    assert frame.ndim == 3
    c, h, w = frame.shape
    assert c < h and c < w, f"expect channel first images, but instead {frame.shape}"


try:
    import rerun.blueprint as rrb
except ImportError:  # pragma: no cover - rerun is an optional viz dependency
    rrb = None


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


def _load_wowskin_background() -> np.ndarray | None:
    bg_image_path = Path(__file__).resolve().parents[3] / "WowSkin" / "images" / "wowskin_bg.png"
    if not bg_image_path.exists():
        logging.warning("WowSkin background image not found at %s", bg_image_path)
        return None

    try:
        from PIL import Image as PILImage

        with PILImage.open(bg_image_path) as image:
            image = image.convert("RGBA")
            aspect_ratio = image.height / image.width
            desired_height = int(WOWSKIN_DISPLAY_WIDTH * aspect_ratio)
            resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
            image = image.resize((WOWSKIN_DISPLAY_WIDTH, desired_height), resample=resample)
            return np.asarray(image)
    except Exception as exc:  # pragma: no cover - visual aid only
        logging.warning("Failed to load WowSkin background image: %s", exc)
        return None


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


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    check_chw_float32(chw_float32_torch)
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def to_hwc_float32_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    check_chw_float32(chw_float32_torch)
    hwc_float32_numpy = chw_float32_torch.permute(1, 2, 0).numpy()
    return hwc_float32_numpy


def build_blueprint_from_dataset(dataset: LeRobotDataset):
    """Build a Rerun blueprint laying out camera images and time series for the given dataset.

    Camera images and scalar signals (action, state, reward, done, success) are arranged in a grid.
    The per-dimension series names for ``action`` and ``state`` are applied directly
    via blueprint overrides.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=key, name=key) for key in dataset.meta.camera_keys]

    # Style multi-dimensional signals (action, state) with per-dimension names.
    for origin, key in ((ACTION, ACTION), ("state", OBS_STATE)):
        if key in dataset.features:
            names = get_feature_names(dataset, key)
            styling = rr.SeriesLines(names=names)
            views.append(rrb.TimeSeriesView(origin=origin, name=origin, overrides={origin: styling}))
    for key in (DONE, REWARD, SUCCESS):
        if key in dataset.features:
            views.append(rrb.TimeSeriesView(origin=key, name=key))

    return rrb.Blueprint(rrb.Grid(*views))


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int | None = None,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    display_mode: str = "rerun",
    host: str = "127.0.0.1",
    autoplay: bool = True,
    **kwargs,
) -> Path | None:
    if display_mode == "foxglove":
        from lerobot.utils.foxglove_visualization import serve_foxglove_dataset_playback

        logging.info("Starting Foxglove server")
        serve_foxglove_dataset_playback(
            dataset,
            episode_index,
            host=host,
            port=web_port if web_port is not None else DEFAULT_FOXGLOVE_PORT,
            compress_images=display_compressed_images,
            autoplay=autoplay,
        )
        return None

    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    from lerobot.utils.import_utils import require_package

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    spawn_local_viewer = mode == "local" and not save
    blueprint = build_blueprint_from_dataset(dataset)
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer, default_blueprint=blueprint)

    bg_image = _load_wowskin_background()
    if bg_image is not None:
        rr.log("game_force", rr.Image(bg_image), static=True)

        if rrb is not None:
            rr.send_blueprint(
                rrb.Spatial2DView(
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[0, float(bg_image.shape[1])],
                        y_range=[0, float(bg_image.shape[0])],
                    )
                )
            )

    bg_image = _load_wowskin_background()
    if bg_image is not None:
        rr.log("game_force", rr.Image(bg_image), static=True)

        if rrb is not None:
            rr.send_blueprint(
                rrb.Spatial2DView(
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[0, float(bg_image.shape[1])],
                        y_range=[0, float(bg_image.shape[0])],
                    )
                )
            )

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        rr.serve_web_viewer(
            open_browser=False,
            web_port=web_port if web_port is not None else DEFAULT_RERUN_PORT,
            connect_to=server_uri,
        )

    logging.info("Logging to Rerun")



    # Depth frames and stats are dequantized to the dataset's depth_output_unit on load.
    depth_meter = 1000.0 if dataset.depth_output_unit == DEPTH_MILLIMETER_UNIT else 1.0

    # Use the dataset's q01/q99 depth statistics for robust depth range bounds
    depth_ranges = {}
    for key in dataset.meta.depth_keys:
        stats = (dataset.meta.stats or {}).get(key)
        if not stats:
            continue
        lo = stats["q01"] if "q01" in stats else stats["min"]
        hi = stats["q99"] if "q99" in stats else stats["max"]
        depth_ranges[key] = (float(np.asarray(lo).item()), float(np.asarray(hi).item()))

    first_index = None
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()

        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # display each camera image (or depth map)
            for key in dataset.meta.camera_keys:
                if key in dataset.meta.depth_keys:
                    depth = to_hwc_float32_numpy(batch[key][i])
                    depth_entity = rr.DepthImage(
                        depth,
                        meter=depth_meter,
                        colormap=rr.components.Colormap.Viridis,
                        depth_range=depth_ranges.get(key),
                    )
                    rr.log(key, entity=depth_entity)
                else:
                    img = to_hwc_uint8_numpy(batch[key][i])
                    img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                    rr.log(key, entity=img_entity)

            # display the action space (e.g. actuators command)
            if ACTION in batch:
                rr.log(ACTION, rr.Scalars(batch[ACTION][i].numpy()))

            # display the observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                rr.log("state", rr.Scalars(batch[OBS_STATE][i].numpy()))

            # Why: log any extra observation vectors, not just `observation.state`,
            # so custom streams like `observation.force` are visible in the same
            # viewer path as built-in state features.
            for key, value in batch.items():
                if not key.startswith(f"{OBS_STR}."):
                    continue
                if key in {OBS_STATE} or ".images." in key:
                    continue

                feature = dataset.features.get(key, {})
                names = feature.get("names") or []
                base_path = key.removeprefix(f"{OBS_STR}.")
                item = value[i]

                if item.ndim == 0:
                    rr.log(base_path, rr.Scalars(item.item()))
                else:
                    for dim_idx, val in enumerate(item):
                        dim_name = names[dim_idx] if dim_idx < len(names) else str(dim_idx)
                        rr.log(f"{base_path}/{dim_name}", rr.Scalars(val.item()))

            # Why: log any extra observation vectors, not just `observation.state`,
            # so custom streams like `observation.force` are visible in the same
            # viewer path as built-in state features.
            for key, value in batch.items():
                if not key.startswith(f"{OBS_STR}."):
                    continue
                if key in {OBS_STATE} or ".images." in key:
                    continue

                feature = dataset.features.get(key, {})
                names = feature.get("names") or []
                base_path = key.removeprefix(f"{OBS_STR}.")
                item = value[i]

                if item.ndim == 0:
                    rr.log(base_path, rr.Scalars(item.item()))
                else:
                    for dim_idx, val in enumerate(item):
                        dim_name = names[dim_idx] if dim_idx < len(names) else str(dim_idx)
                        rr.log(f"{base_path}/{dim_name}", rr.Scalars(val.item()))

            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if SUCCESS in batch:
                rr.log(SUCCESS, rr.Scalars(batch[SUCCESS][i].item()))

            # ================= FORCE VISUALIZATION (WOWSKIN STYLE) =================
            force_key = OBS_FORCE if OBS_FORCE in batch else None
            if force_key is None:
                for k in batch.keys():
                    if "force" in k and k not in {"index", "timestamp"}:
                        force_key = k
                        break

            if force_key is not None:
                _log_wowskin_force(rr, batch[force_key][i])
            # ================= END FORCE VISUALIZATION =================

    # save .rrd locally
    if mode == "local" and save:
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # Keep the process alive while it serves the gRPC/web connection.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help=(
            "Web/WebSocket port. For rerun `--mode distant` it is the web viewer port (default 9090); "
            "for `--display-mode foxglove` it is the server bind port (default 8765)."
        ),
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="If set, display compressed (JPEG) images instead of uncompressed ones.",
    )

    parser.add_argument(
        "--display-mode",
        type=str,
        default="rerun",
        choices=["rerun", "foxglove"],
        help=(
            "Visualization backend. 'rerun' uses the Rerun viewer (--mode/--save/--*-port apply). "
            "'foxglove' starts a Foxglove WebSocket server that serves the episode as a seekable, "
            "scrubbable timeline; connect the Foxglove app to ws://HOST:PORT (--host/--web-port)."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Host to bind the Foxglove WebSocket server to when `--display-mode foxglove` is set "
            "(127.0.0.1 for local only, 0.0.0.0 for all interfaces)."
        ),
    )
    parser.add_argument(
        "--no-autoplay",
        dest="autoplay",
        action="store_false",
        help=(
            "For `--display-mode foxglove`: don't start playing automatically when a client "
            "connects; wait for play to be pressed in the Foxglove app instead."
        ),
    )

    args = parser.parse_args()

    if args.display_mode == "foxglove":
        rerun_only = ("mode", "save", "output_dir", "grpc_port", "batch_size", "num_workers")
        ignored = [name for name in rerun_only if getattr(args, name) != parser.get_default(name)]
        if ignored:
            logging.warning(
                "These flags only apply to `--display-mode rerun` and are ignored with "
                "`--display-mode foxglove`: %s.",
                ", ".join(f"--{name.replace('_', '-')}" for name in ignored),
            )

    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    init_logging()
    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **kwargs)


if __name__ == "__main__":
    main()
