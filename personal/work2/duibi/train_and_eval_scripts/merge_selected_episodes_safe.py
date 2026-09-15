#!/usr/bin/env python3
"""Safe wrapper around merge_selected_episodes.py.

Some locally collected LeRobot datasets contain a one-frame discrepancy between
an episode's discrete ``length`` and the frame span reconstructed from video
``from_timestamp``/``to_timestamp``. At high FPS this can be caused by floating
point timestamp serialization. Upstream ``split_dataset`` intentionally asserts
exact equality and therefore aborts before subset merging.

This wrapper repairs only a +/-1 frame timestamp-rounding discrepancy in memory
for the episodes being split. The source dataset on disk is never modified.
Larger discrepancies, or a mismatch between parquet frame indices and episode
length, remain hard errors so real data corruption is not hidden.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import split_dataset as _lerobot_split_dataset
from lerobot.datasets.io_utils import load_episodes

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import merge_selected_episodes as _merge_impl


def _selected_episode_ids(splits: dict, total_episodes: int) -> set[int]:
    """Return explicit episode ids for list-based splits."""
    selected: set[int] = set()
    for value in splits.values():
        if isinstance(value, float):
            return set()
        selected.update(int(x) for x in value)
    invalid = selected - set(range(total_episodes))
    if invalid:
        raise ValueError(f"Invalid episode indices before safe split: {sorted(invalid)}")
    return selected


def _materialize_episode_metadata(dataset) -> None:
    """Convert HF Dataset episode metadata to a mutable list of dicts.

    ``load_episodes`` returns a Hugging Face Dataset. Indexing it returns a dict
    copy, so mutating ``dataset.meta.episodes[i]`` would not persist. The dataset
    tools only require list-style indexing/iteration for this path, therefore a
    materialized list is safe and makes the in-memory correction visible to the
    subsequent ``split_dataset`` call.
    """
    episodes = dataset.meta.episodes
    if episodes is None:
        episodes = load_episodes(dataset.meta.root)
    if not isinstance(episodes, list):
        episodes = [dict(episodes[i]) for i in range(len(episodes))]
    else:
        episodes = [dict(ep) for ep in episodes]
    dataset.meta.episodes = episodes


def _repair_one_frame_video_timestamp_drift(dataset, episode_ids: set[int]) -> int:
    """Repair +/-1 frame timestamp rounding errors for selected episodes.

    ``episode['length']`` and the parquet index span are treated as the discrete
    source of truth. Timestamps locate the start frame, and the end timestamp is
    corrected so the half-open video range has exactly the declared length.
    """
    if not episode_ids or not dataset.meta.video_keys:
        return 0

    _materialize_episode_metadata(dataset)

    fps = float(dataset.meta.fps)
    if fps <= 0:
        raise ValueError(f"Invalid dataset fps: {fps}")

    repaired = 0
    for ep_idx in sorted(episode_ids):
        ep = dataset.meta.episodes[ep_idx]
        declared_length = int(ep["length"])

        # The discrete parquet/data span must agree exactly with the episode
        # length. If it does not, this is not merely a timestamp-rounding issue.
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            data_length = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
            if data_length != declared_length:
                raise ValueError(
                    f"Episode {ep_idx} data-length mismatch: metadata length={declared_length}, "
                    f"dataset index span={data_length}. Refusing automatic repair."
                )

        for video_key in dataset.meta.video_keys:
            from_key = f"videos/{video_key}/from_timestamp"
            to_key = f"videos/{video_key}/to_timestamp"
            if from_key not in ep or to_key not in ep:
                continue

            from_ts = float(ep[from_key])
            to_ts = float(ep[to_key])
            from_frame = round(from_ts * fps)
            to_frame = round(to_ts * fps)
            timestamp_span = to_frame - from_frame

            if timestamp_span == declared_length:
                continue

            frame_error = timestamp_span - declared_length
            if abs(frame_error) > 1:
                raise ValueError(
                    f"Episode {ep_idx}, video '{video_key}' has a non-trivial frame mismatch: "
                    f"length={declared_length}, timestamp span={timestamp_span}, "
                    f"from={from_ts:.9f}, to={to_ts:.9f}, fps={fps}. "
                    "Refusing automatic repair because the discrepancy exceeds one frame."
                )

            corrected_to_frame = from_frame + declared_length
            corrected_to_ts = corrected_to_frame / fps
            logging.warning(
                "Repairing one-frame video timestamp drift: episode=%s key=%s "
                "length=%s timestamp_span=%s from_ts=%.9f old_to_ts=%.9f new_to_ts=%.9f",
                ep_idx,
                video_key,
                declared_length,
                timestamp_span,
                from_ts,
                to_ts,
                corrected_to_ts,
            )
            ep[to_key] = corrected_to_ts
            repaired += 1

    return repaired


def safe_split_dataset(dataset, splits, output_dir=None):
    episode_ids = _selected_episode_ids(splits, dataset.meta.total_episodes)
    repaired = _repair_one_frame_video_timestamp_drift(dataset, episode_ids)
    if repaired:
        print(f"  ✓ 已在内存中修复 {repaired} 个 one-frame video timestamp 边界（源数据未修改）")
    return _lerobot_split_dataset(dataset=dataset, splits=splits, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely merge selected episodes while tolerating one-frame video timestamp drift"
    )
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-base-dir",
        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view",
    )
    args = parser.parse_args()

    # merge_selected_episodes imported split_dataset into module scope. Replace
    # that symbol only for this process, then reuse the existing merge pipeline.
    _merge_impl.split_dataset = safe_split_dataset
    _merge_impl.merge_from_subset_file(
        subset_file=args.subset_file,
        output_dir=args.output_dir,
        dataset_base_dir=args.dataset_base_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
