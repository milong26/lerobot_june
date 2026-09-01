#!/usr/bin/env python
"""
LIBERO 初始状态渲染验证工具。

功能说明：
    读取 collect_libero_dataset.py 生成的 episode_initial_states.json 或
    uniform_initial_states.npz，根据其中 suite/task_id 自动创建相同的 LIBERO task，
    读取指定的 episode-index / state-index，执行 env.set_init_state(state)，
    保存 agentview 和 wrist 相机的 PNG 图像，用于证明保存的
    initial_environment_state 可以恢复场景。

数据来源：
    - episode_initial_states.json: 包含 suite, task_id, 每个 episode 的 initial_environment_state
    - uniform_initial_states.npz: 包含 states 数组和 seeds 数组

运行方法：
    cd /data/zhonglinye/jun/lerobot
    # 从 episode_initial_states.json 渲染 episode 0
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --dataset-dir personal/work2/dataset/libero_spatial_task0 \
        --episode-index 0

    # 从 uniform_initial_states.npz 渲染 state 5
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --dataset-dir personal/work2/dataset/libero_spatial_task0 \
        --state-index 5

    # 指定输出目录
    python personal/work2/collect_dataset/libero/render_initial_state.py \
        --dataset-dir personal/work2/dataset/libero_spatial_task0 \
        --episode-index 0 \
        --output-dir /tmp/render_test
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
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="渲染 LIBERO 初始状态验证图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=str, required=True, help="数据集目录路径")
    parser.add_argument("--episode-index", type=int, default=0, help="要渲染的 episode 索引")
    parser.add_argument("--state-index", type=int, default=None, help="从 NPZ 渲染的 state 索引（优先于 episode-index）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，默认为 dataset_dir/render_output")
    parser.add_argument("--image-size", type=int, default=360, help="图像分辨率")
    return parser.parse_args()


def _resize_image(image, target_size):
    """使用 PIL 调整图像大小。"""
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dataset_dir / "render_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LIBERO 初始状态渲染验证")
    print("=" * 80)
    print(f"数据集目录: {dataset_dir}")
    print(f"输出目录: {output_dir}")

    json_file = dataset_dir / "episode_initial_states.json"
    npz_file = dataset_dir / "uniform_initial_states.npz"

    metadata = None
    if json_file.exists():
        with open(json_file, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {json_file}")
        print(f"  Suite: {metadata.get('suite')}")
        print(f"  Task ID: {metadata.get('task_id')}")
        print(f"  Task name: {metadata.get('task_name')}")
        print(f"  State dim: {metadata.get('state_dim')}")

    initial_state = None
    suite_name = None
    task_id = None

    if args.state_index is not None and npz_file.exists():
        print(f"\n从 NPZ 加载 state index={args.state_index}...")
        npz_data = np.load(str(npz_file))
        states = npz_data["states"]
        if args.state_index >= len(states):
            print(f"ERROR: state_index {args.state_index} out of range [0, {len(states) - 1}]")
            sys.exit(1)
        initial_state = states[args.state_index]
        print(f"  State shape: {initial_state.shape}")
        if metadata:
            suite_name = metadata.get("suite")
            task_id = metadata.get("task_id")
    elif metadata and args.episode_index < len(metadata.get("episodes", [])):
        ep_data = metadata["episodes"][args.episode_index]
        initial_state = np.array(ep_data["initial_environment_state"])
        suite_name = metadata.get("suite")
        task_id = metadata.get("task_id")
        print(f"\n从 JSON 加载 episode index={args.episode_index}...")
        print(f"  State shape: {initial_state.shape}")
        print(f"  Seed: {ep_data.get('seed')}")
    else:
        print("ERROR: Cannot find initial state. Check dataset directory.")
        sys.exit(1)

    if suite_name is None or task_id is None:
        print("ERROR: Cannot determine suite/task from metadata.")
        sys.exit(1)

    print(f"\n创建 LIBERO 环境: suite={suite_name}, task_id={task_id}...")
    bench = benchmark.get_benchmark_dict()
    suite = bench[suite_name]()
    task = suite.get_task(task_id)

    bddl_root = Path(get_libero_path("bddl_files"))
    bddl_path = bddl_root / task.problem_folder / task.bddl_file
    print(f"  BDDL: {bddl_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        control_freq=20,
    )
    env.reset()

    print(f"\n设置初始状态 (dim={len(initial_state)})...")
    env.set_init_state(initial_state)

    print("渲染图像...")
    obs = env.env._get_observations()

    agentview_img = obs.get("agentview_image")
    wrist_img = obs.get("robot0_eye_in_hand_image")

    if agentview_img is None or wrist_img is None:
        print("ERROR: Failed to get camera observations.")
        env.close()
        sys.exit(1)

    agentview_img = np.flip(agentview_img, (0, 1))
    wrist_img = np.flip(wrist_img, (0, 1))

    agentview_img = _resize_image(agentview_img, args.image_size)
    wrist_img = _resize_image(wrist_img, args.image_size)

    from PIL import Image

    agentview_path = output_dir / f"episode_{args.episode_index}_agentview.png"
    wrist_path = output_dir / f"episode_{args.episode_index}_wrist.png"

    Image.fromarray(agentview_img).save(str(agentview_path))
    Image.fromarray(wrist_img).save(str(wrist_path))

    print(f"\nAgentview 图像已保存到: {agentview_path}")
    print(f"Wrist 图像已保存到: {wrist_path}")

    env.close()

    print("\n" + "=" * 80)
    print("渲染验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()