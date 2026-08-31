#!/usr/bin/env python
"""
验证 LIBERO 数据集初始状态恢复脚本。

功能：
    读取 dataset 的 episode_initial_states.json 或 uniform_initial_states.npz，
    根据其中 suite/task_id 自动创建相同 LIBERO task，
    读取指定 episode-index/state-index，执行 env.set_init_state(state)，
    保存 agentview 和 wrist PNG，用于证明保存的 initial_environment_state 可以恢复场景。

运行方法：
    cd /data/zhonglinye/jun/lerobot
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --input-dir personal/work2/dataset/libero_spatial_task0 \
        --episode-index 0 \
        --output-dir personal/work2/dataset/libero_spatial_task0/renders

    指定 state-index（从 uniform_initial_states.npz 中选择）：
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --input-dir personal/work2/dataset/libero_spatial_task0 \
        --state-index 10 \
        --output-dir personal/work2/dataset/libero_spatial_task0/renders
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

LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证 LIBERO 数据集初始状态恢复",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="数据集目录，包含 episode_initial_states.json")
    parser.add_argument("--episode-index", type=int, default=0,
                        help="要验证的 episode 索引 (默认 0)")
    parser.add_argument("--state-index", type=int, default=None,
                        help="从 uniform_initial_states.npz 中选择的 state 索引")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="渲染图像输出目录 (默认 <input-dir>/renders)")
    parser.add_argument("--image-size", type=int, default=360,
                        help="图像分辨率 (默认 360)")
    return parser.parse_args()


def get_suite_and_task(suite_name: str, task_id: int):
    """获取 LIBERO suite 和 task 对象。"""
    bench_dict = benchmark.get_benchmark_dict()
    if suite_name not in bench_dict:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {', '.join(sorted(bench_dict.keys()))}")
    suite = bench_dict[suite_name]()
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"task_id {task_id} out of range [0, {len(suite.tasks) - 1}]")
    task = suite.get_task(task_id)
    return suite, task


def resolve_bddl_file(task):
    """获取 BDDL 文件路径。"""
    bddl_path = get_libero_path("bddl_files") / task.problem_folder / task.bddl_file
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")
    return bddl_path


def create_env(bddl_file_path: str, image_size: int = 360, control_freq: int = 20):
    """创建 OffScreenRenderEnv 环境。"""
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file_path),
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=control_freq,
    )
    return env


def get_flattened_env_state(env) -> np.ndarray:
    """获取完整的 flattened MuJoCo simulator state。"""
    sim = env.env.sim
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    act = sim.data.act.copy() if sim.data.act is not None else np.array([])
    return np.concatenate([qpos, qvel, act])


def load_initial_state(input_dir: Path, episode_index: int = 0, state_index: int = None):
    """
    加载初始状态。

    Args:
        input_dir: 数据集目录
        episode_index: episode 索引（优先使用 episode_initial_states.json）
        state_index: state 索引（使用 uniform_initial_states.npz）

    Returns:
        (state, metadata) - flattened state 和元数据
    """
    episode_meta_path = input_dir / "episode_initial_states.json"

    if state_index is not None:
        npz_path = input_dir / "uniform_initial_states.npz"
        if npz_path.exists():
            npz_data = np.load(npz_path)
            if "states" in npz_data:
                state = npz_data["states"][state_index]
                metadata = {"source": "npz", "state_index": state_index}
                return state, metadata
        raise ValueError(f"state_index={state_index} specified but uniform_initial_states.npz not found")

    if episode_meta_path.exists():
        with open(episode_meta_path) as f:
            meta = json.load(f)

        if episode_index < len(meta["episodes"]):
            ep_data = meta["episodes"][episode_index]
            state = np.array(ep_data["initial_environment_state"])
            metadata = {
                "source": "episode_initial_states",
                "episode_index": episode_index,
                "seed": ep_data["seed"],
                "task_id": meta.get("task_id"),
                "suite": meta.get("suite"),
            }
            return state, metadata

    npz_path = input_dir / "uniform_initial_states.npz"
    if npz_path.exists():
        npz_data = np.load(npz_path)
        if "states" in npz_data and len(npz_data["states"]) > 0:
            state = npz_data["states"][0]
            metadata = {"source": "npz_default", "state_index": 0}
            return state, metadata

    raise FileNotFoundError(f"No initial state found in {input_dir}")


def render_initial_state(
    env,
    state: np.ndarray,
    image_size: int = 360,
) -> tuple:
    """
    渲染初始状态场景。

    Args:
        env: OffScreenRenderEnv 实例
        state: flattened state
        image_size: 图像分辨率

    Returns:
        (top_image, wrist_image) - 两个相机的图像
    """
    env.env.sim.set_state_from_flatten(state)

    for _ in range(10):
        env.step(LIBERO_DUMMY_ACTION)

    raw_obs = env.env._get_observations()

    top_image = raw_obs.get("agentview_image")
    wrist_image = raw_obs.get("robot0_eye_in_hand_image")

    if top_image is not None:
        top_image = top_image[::-1, ::-1]
    if wrist_image is not None:
        wrist_image = wrist_image[::-1, ::-1]

    return top_image, wrist_image


def save_images(
    top_image,
    wrist_image,
    output_dir: Path,
    prefix: str = "episode",
):
    """
    保存图像到文件。

    Args:
        top_image: agentview 图像
        wrist_image: wrist 图像
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    if top_image is not None:
        top_path = output_dir / f"{prefix}_agentview.png"
        Image.fromarray(top_image).save(top_path)
        print(f"Saved agentview image to {top_path}")

    if wrist_image is not None:
        wrist_path = output_dir / f"{prefix}_wrist.png"
        Image.fromarray(wrist_image).save(wrist_path)
        print(f"Saved wrist image to {wrist_path}")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if args.output_dir is None:
        output_dir = input_dir / "renders"
    else:
        output_dir = Path(args.output_dir)

    print("=== LIBERO Initial State Renderer ===")
    print(f"Input directory: {input_dir}")
    print(f"Episode index: {args.episode_index}")
    print(f"State index: {args.state_index}")
    print()

    state, metadata = load_initial_state(input_dir, args.episode_index, args.state_index)
    print(f"Loaded state from: {metadata['source']}")
    print(f"State shape: {state.shape}")

    suite_name = metadata.get("suite", "libero_spatial")
    task_id = metadata.get("task_id", 0)

    if "episode_index" in metadata:
        print(f"Suite: {suite_name}, Task ID: {task_id}, Episode: {metadata['episode_index']}")
    else:
        print(f"Suite: {suite_name}, Task ID: {task_id}, State index: {metadata.get('state_index', 'N/A')}")

    suite, task = get_suite_and_task(suite_name, task_id)
    task_language = task.language
    print(f"Task: {task.name}")
    print(f"Description: {task_language}")

    bddl_file = resolve_bddl_file(task)
    print(f"BDDL: {bddl_file}")

    print("\nCreating environment...")
    env = create_env(bddl_file, args.image_size)
    env.reset()

    print("Rendering initial state...")
    top_image, wrist_image = render_initial_state(env, state, args.image_size)

    prefix = f"episode{args.episode_index}"
    if args.state_index is not None:
        prefix = f"state{args.state_index}"

    print(f"\nSaving images to {output_dir}...")
    save_images(top_image, wrist_image, output_dir, prefix)

    env.close()

    print("\n=== Rendering complete ===")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()