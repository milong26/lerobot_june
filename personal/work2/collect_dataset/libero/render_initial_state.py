#!/usr/bin/env python
"""
渲染初始状态验证脚本。

功能：
    读取 episode_initial_states.json 或 uniform_initial_states.npz，
    根据其中 suite/task_id 自动创建相同 LIBERO task，
    读取指定 episode-index/state-index，执行 env.set_init_state(state)，
    保存 agentview 和 wrist PNG，用于证明保存的 initial_environment_state 可以恢复场景。

运行方法：
    cd /data/zhonglinye/jun/lerobot

    # 渲染 episode 0 的初始状态
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --input-dir personal/work2/dataset/libero_spatial_task0 \
        --episode-index 0

    # 渲染多个 episode
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --input-dir personal/work2/dataset/libero_spatial_task0 \
        --episode-index 0 --episode-index 5 --episode-index 10

    # 使用 NPZ 文件中的 state-index
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --input-dir personal/work2/dataset/libero_spatial_task0 \
        --state-index 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="渲染初始状态验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="数据集目录（包含 episode_initial_states.json 或 uniform_initial_states.npz）")
    parser.add_argument("--episode-index", type=int, action="append",
                        help="要渲染的 episode 索引（可多次指定）")
    parser.add_argument("--state-index", type=int, action="append",
                        help="要渲染的 state 索引（从 NPZ 文件读取）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，默认 <input-dir>/rendered_states")
    parser.add_argument("--image-size", type=int, default=360,
                        help="图像分辨率（默认 360）")
    return parser.parse_args()


def load_metadata(input_dir: Path):
    """加载 episode_initial_states.json 或 uniform_initial_states.json。"""
    json_path = input_dir / "episode_initial_states.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f), "episode"

    json_path = input_dir / "uniform_initial_states.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f), "uniform"

    raise FileNotFoundError(
        f"Neither episode_initial_states.json nor uniform_initial_states.json found in {input_dir}"
    )


def load_npz(input_dir: Path):
    """加载 uniform_initial_states.npz。"""
    npz_path = input_dir / "uniform_initial_states.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"uniform_initial_states.npz not found in {input_dir}")
    return np.load(npz_path)


def get_suite_and_task(suite_name: str, task_id: int):
    """获取指定 suite 和 task_id 对应的 task 对象。"""
    bench = benchmark.get_benchmark_dict()
    if suite_name not in bench:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {', '.join(bench.keys())}")
    suite = bench[suite_name]()
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"task_id {task_id} out of range [0, {len(suite.tasks) - 1}]")
    task = suite.get_task(task_id)
    return suite, task


def resolve_bddl_file(task):
    """获取 BDDL 文件的完整路径。"""
    bddl_root = get_libero_path("bddl_files")
    bddl_file = bddl_root / task.problem_folder / task.bddl_file
    if not bddl_file.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_file}")
    return bddl_file


def create_env(bddl_file: Path, image_size: int):
    """创建 OffScreenRenderEnv 环境。"""
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=20,
    )
    return env


def get_flattened_env_state(env) -> np.ndarray:
    """获取完整的 MuJoCo simulator state。"""
    sim = env.env.sim
    return sim.get_state().flatten()


def render_initial_state(env, state: np.ndarray, output_path: Path, image_size: int):
    """渲染指定初始状态并保存图像。"""
    env.reset()
    env.set_init_state(state)

    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])

    raw_obs = env.get_obs()

    top_image = raw_obs.get("agentview_image")
    wrist_image = raw_obs.get("robot0_eye_in_hand_image")

    if top_image is not None:
        top_path = output_path.parent / f"{output_path.stem}_top.png"
        import PIL.Image
        img = PIL.Image.fromarray(top_image)
        img.save(top_path)
        print(f"  Saved top view: {top_path}")

    if wrist_image is not None:
        wrist_path = output_path.parent / f"{output_path.stem}_wrist.png"
        import PIL.Image
        img = PIL.Image.fromarray(wrist_image)
        img.save(wrist_path)
        print(f"  Saved wrist view: {wrist_path}")

    restored_state = get_flattened_env_state(env)
    state_diff = np.linalg.norm(state - restored_state)
    print(f"  State restoration error: {state_diff:.6f}")

    return top_image, wrist_image, state_diff


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "rendered_states"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Initial State Renderer")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    metadata, metadata_type = load_metadata(input_dir)

    suite_name = metadata["suite"]
    task_id = metadata["task_id"]
    task_name = metadata.get("task_name", "unknown")
    task_description = metadata.get("task_description", "unknown")

    print(f"Suite: {suite_name}")
    print(f"Task ID: {task_id}")
    print(f"Task name: {task_name}")
    print(f"Task description: {task_description}")
    print(f"Metadata type: {metadata_type}")

    _, task = get_suite_and_task(suite_name, task_id)
    bddl_file = resolve_bddl_file(task)
    print(f"BDDL file: {bddl_file}")

    print(f"\nCreating environment...")
    env = create_env(bddl_file, args.image_size)

    npz_data = None
    if metadata_type == "uniform" or args.state_index is not None:
        npz_data = load_npz(input_dir)

    episode_indices = args.episode_index or []
    state_indices = args.state_index or []

    if not episode_indices and not state_indices:
        print("\nNo episode-index or state-index specified, rendering first 3 states...")
        if metadata_type == "episode":
            episode_indices = [0, 1, 2]
        else:
            state_indices = [0, 1, 2]

    print(f"\n{'='*60}")
    print(f"Rendering initial states...")
    print(f"{'='*60}\n")

    for idx in episode_indices:
        print(f"Rendering episode {idx}...")
        if metadata_type == "episode":
            episodes = metadata.get("episodes", [])
            if idx >= len(episodes):
                print(f"  ERROR: Episode {idx} not found (only {len(episodes)} episodes)")
                continue
            ep = episodes[idx]
            state = np.array(ep["initial_environment_state"])
            seed = ep.get("seed")
        else:
            if npz_data is None:
                print(f"  ERROR: NPZ data not available")
                continue
            if idx >= len(npz_data["states"]):
                print(f"  ERROR: State index {idx} not found (only {len(npz_data['states'])} states)")
                continue
            state = npz_data["states"][idx]
            seeds = npz_data.get("seeds", [])
            seed = seeds[idx] if idx < len(seeds) else None

        print(f"  Seed: {seed}")
        print(f"  State shape: {state.shape}")

        output_path = output_dir / f"episode_{idx:03d}"
        try:
            top_img, wrist_img, state_diff = render_initial_state(env, state, output_path, args.image_size)
            if state_diff > 1e-3:
                print(f"  WARNING: State restoration error is large ({state_diff:.6f})")
        except Exception as e:
            print(f"  ERROR: {e}")

    for idx in state_indices:
        print(f"Rendering state index {idx}...")
        if npz_data is None:
            print(f"  ERROR: NPZ data not available")
            continue
        if idx >= len(npz_data["states"]):
            print(f"  ERROR: State index {idx} not found (only {len(npz_data['states'])} states)")
            continue

        state = npz_data["states"][idx]
        seeds = npz_data.get("seeds", [])
        seed = seeds[idx] if idx < len(seeds) else None

        print(f"  Seed: {seed}")
        print(f"  State shape: {state.shape}")

        output_path = output_dir / f"state_{idx:03d}"
        try:
            top_img, wrist_img, state_diff = render_initial_state(env, state, output_path, args.image_size)
            if state_diff > 1e-3:
                print(f"  WARNING: State restoration error is large ({state_diff:.6f})")
        except Exception as e:
            print(f"  ERROR: {e}")

    env.close()

    print(f"\n{'='*60}")
    print(f"Done! Rendered images saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()