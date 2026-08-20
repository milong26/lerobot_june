#!/usr/bin/env python
"""
通过seed生成的。

从0开始采集Meta-World指定任务的数据，并保存为LeRobot数据集格式。

数据集包含：
- observation.images.top: 固定全局相机视角视频 (480x480)
- observation.images.wrist: 手腕/夹爪视角视频 (480x480)
- observation.state: 机械臂本体状态 (4维: xyz + gripper)
- observation.environment_state: 39维环境参数 (包含obj_pose等)
- action: 4维动作 (x, y, z, gripper)
- next.reward: 奖励
- next.success: 是否成功
- task: 任务名称字符串

每个episode的初始物体位置(obj_pose)和目标位置(goal_pose)记录在episode metadata中，方便快速访问。

物体位置随机化说明：
- pick-place-v3任务有两个关键位置：物体初始位置(obj_init_pos)和目标位置(goal)
- 随机化时，两个位置都会随机变化
- 通过不同seed创建MT1环境，每个seed对应一组固定的obj_init_pos和goal
- 默认启用随机化，每个episode使用不同的seed

使用示例（命令行）:
    # 采集10个episode的pick-place-v3数据（随机化物体和目标位置）
    python collect_metaworld_dataset.py \
        --task pick-place-v3 \
        --num-episodes 10 \
        --output-dir ./outputs/metaworld_pick_place \
        --repo-id your-username/metaworld_pick_place \
        --fps 80 \
        --image-size 480 \
        --randomize-obj \
        --seed-start 0

参数说明:
    --task: Meta-World任务名称，如 pick-place-v3, assembly-v3, push-v3 等
    --num-episodes: 采集的episode数量
    --output-dir: 数据集本地保存路径
    --repo-id: HuggingFace仓库ID，格式: 用户名/数据集名
    --fps: 视频帧率，Meta-World默认80fps
    --image-size: 相机图像分辨率（宽高相同）
    --randomize-obj: 是否随机化物体初始位置和目标位置（通过不同seed实现，默认启用）
    --no-randomize-obj: 禁用随机化，所有episode使用相同的固定位置
    --seed-start: 随机化起始seed，每个episode使用 seed_start + episode_index
    --max-steps: 每个episode最大步数
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 必须在任何mujoco/gymnasium import之前设置
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import mujoco
import metaworld
import metaworld.policies as policies

# 设置lerobot的本地数据路径
os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).parent / "outputs")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载metaworld配置
CONFIG_PATH = Path(__file__).parent.parent.parent / "src" / "lerobot" / "envs" / "metaworld_config.json"
with open(CONFIG_PATH) as f:
    METAWORLD_CONFIG = json.load(f)

TASK_DESCRIPTIONS = METAWORLD_CONFIG.get("TASK_DESCRIPTIONS", {})


# Meta-World 39维observation的结构说明（修正版，见 SPEC.md 3.1节）
ENV_STATE_DESCRIPTION = {
    "0:3": "末端执行器(手)位置 xyz",
    "3:4": "夹爪开合度(归一化)",
    "4:7": "物体1位置 xyz (= obj_pose)",
    "7:11": "物体1四元数朝向(4维)",
    "11:14": "物体2位置(单物体任务中恒为0)",
    "14:18": "物体2四元数(恒为0)",
    "18:36": "上一帧的[0:18]原样重复(frame-stack)",
    "36:39": "目标位置 xyz (= goal_pose)",
}


def create_metaworld_env(task_name, seed=None, camera_name="corner2"):
    """
    创建Meta-World环境，指定相机视角。

    Args:
        task_name: 任务名称，如 "pick-place-v3"
        seed: 随机种子，控制物体初始位置。None则使用默认seed 42
        camera_name: 相机名称，可用选项:
            - "corner2": 固定全局视角（从角落俯瞰整个工作空间）
            - "behindGripper": 手腕/夹爪视角（跟随机械臂末端执行器运动）
            - "gripperPOV": 另一个手腕视角
            - "topview": 俯视视角
            - "corner", "corner3": 其他全局视角

    Returns:
        env: Meta-World环境实例
    """
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    task = mt1.train_tasks[0]
    env.set_task(task)

    # 保持物体位置固定（使用task设置的随机向量）
    # _freeze_rand_vec = True 表示每次reset使用相同的随机向量
    env._freeze_rand_vec = True

    return env


def sync_env_state(src_env, dst_env):
    """
    将源环境的仿真状态同步到目标环境。
    通过复制qpos和qctrl实现状态同步。
    """
    dst_env.data.qpos[:] = src_env.data.qpos[:]
    dst_env.data.qvel[:] = src_env.data.qvel[:]
    dst_env.data.ctrl[:] = src_env.data.ctrl[:]
    mujoco.mj_forward(dst_env.model, dst_env.data)


def render_dual_camera(env_top, env_wrist, image_size=480):
    """
    渲染两个相机视角：全局固定相机 + 手腕相机。

    Meta-World可用的相机名称包括：
    - "corner2": 固定全局视角（从角落俯瞰整个工作空间）
    - "behindGripper": 手腕/夹爪视角（跟随机械臂末端执行器运动）
    - "gripperPOV": 另一个手腕视角
    - "topview": 俯视视角
    - "corner", "corner3": 其他全局视角

    Args:
        env_top: 全局相机环境实例
        env_wrist: 手腕相机环境实例
        image_size: 输出图像分辨率

    Returns:
        top_image: 全局相机RGB图像 (image_size, image_size, 3)
        wrist_image: 手腕相机RGB图像 (image_size, image_size, 3)
    """
    # 渲染全局相机（corner2）
    top_image = env_top.render()
    if top_image is not None:
        top_image = np.flip(top_image, (0, 1))
        top_image = resize_image(top_image, image_size)

    # 渲染手腕相机（gripperPOV）
    wrist_image = env_wrist.render()
    if wrist_image is not None:
        wrist_image = resize_image(wrist_image, image_size)

    return top_image, wrist_image


def resize_image(image, target_size):
    """使用numpy双线性插值调整图像大小（避免额外依赖）。"""
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def get_obj_pose_from_env(env):
    """
    从环境中获取物体初始位置。

    Meta-World的Sawyer环境在reset后会设置 obj_init_pos 属性，
    这是物体的初始3D坐标。

    Args:
        env: 已reset的Meta-World环境

    Returns:
        obj_pose: 3D numpy array (x, y, z)
    """
    return env.obj_init_pos.copy()


def get_goal_pose_from_env(env):
    """获取目标位置。"""
    return env.goal.copy()


def run_episode(env_top, env_wrist, expert_policy, task_name, max_steps=500, image_size=480, extra_frames_after_success=10):
    """
    运行一个完整的episode，使用expert policy生成示范数据。

    Args:
        env_top: 全局相机环境
        env_wrist: 手腕相机环境
        expert_policy: 专家策略，用于生成示范动作
        task_name: 任务名称
        max_steps: 最大步数
        image_size: 图像分辨率
        extra_frames_after_success: 成功后继续采集的帧数（默认10帧）

    Returns:
        frames: 列表，每个元素是一个frame的字典
        episode_info: episode元信息字典
    """
    obs, info = env_top.reset()
    env_wrist.reset()
    sync_env_state(env_top, env_wrist)

    # 记录初始环境信息
    obj_pose = get_obj_pose_from_env(env_top)
    goal_pose = get_goal_pose_from_env(env_top)

    frames = []
    success_flags = []
    success_detected = False
    frames_after_success = 0

    for step in range(max_steps):
        # 使用expert policy生成动作
        action = expert_policy.get_action(obs)

        # 执行动作
        obs, reward, terminated, truncated, info = env_top.step(action)
        sync_env_state(env_top, env_wrist)

        # 渲染双相机
        top_image, wrist_image = render_dual_camera(env_top, env_wrist, image_size)

        if top_image is None or wrist_image is None:
            print(f"  Warning: Failed to render at step {step}, skipping frame.")
            continue

        # 构建frame数据
        task_description = TASK_DESCRIPTIONS.get(task_name, task_name)
        frame = {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": obs[:4].copy().astype(np.float32),
            "observation.environment_state": obs.copy().astype(np.float32),
            "action": action.copy().astype(np.float32),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.success": np.array([info.get("success", 0)], dtype=bool),
            "task": task_description,
        }
        frames.append(frame)
        success_flags.append(info.get("success", 0))

        # 检测是否成功
        current_success = info.get("success", 0)
        if current_success and not success_detected:
            success_detected = True
            frames_after_success = 0
            print(f"  >>> Success detected at step {step}, collecting {extra_frames_after_success} more frames...")

        # 如果已经检测到成功，继续采集指定帧数后结束
        if success_detected:
            frames_after_success += 1
            if frames_after_success >= extra_frames_after_success:
                print(f"  >>> Collected {extra_frames_after_success} extra frames, ending episode.")
                break

        if terminated or truncated:
            break

    episode_info = {
        "obj_init_pos": obj_pose,
        "goal_pose": goal_pose,
        "success": any(success_flags),
        "num_frames": len(frames),
    }

    return frames, episode_info


def create_dataset(repo_id, output_dir, fps=80, image_size=480):
    """
    创建空的LeRobot数据集。

    Args:
        repo_id: HuggingFace仓库ID
        output_dir: 本地保存路径
        fps: 帧率
        image_size: 图像分辨率

    Returns:
        dataset: LeRobotDataset实例
    """
    # 定义数据集的features
    # 注意：task 字段不需要在这里定义，它是 LeRobot 的特殊必填字段，
    # validate_frame 会单独检查，不参与常规 feature 验证
    features = {
        "observation.images.top": {
            "dtype": "image",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (39,),
            "names": ["keypoints"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"axes": ["x", "y", "z", "gripper"]},
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="metaworld",
        use_videos=True,
    )

    return dataset


def save_episode_metadata(output_dir, episode_infos, task_name):
    """
    保存所有episode的初始环境信息到JSON文件，方便快速访问。

    这个文件记录了每个episode的obj_pose（物体初始位置），
    用户可以直接读取这个JSON文件来查看数据集的环境配置。

    Args:
        output_dir: 数据集根目录
        episode_infos: episode信息列表
        task_name: 任务名称
    """
    metadata_file = Path(output_dir) / "episode_initial_states.json"

    metadata = {
        "task": task_name,
        "num_episodes": len(episode_infos),
        "env_state_structure": ENV_STATE_DESCRIPTION,
        "episodes": [],
    }

    for i, info in enumerate(episode_infos):
        ep_data = {
            "episode_index": i,
            "obj_init_pos": info["obj_init_pos"].tolist(),
            "goal_pose": info["goal_pose"].tolist(),
            "success": bool(info["success"]),
            "num_frames": info["num_frames"],
        }
        metadata["episodes"].append(ep_data)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEpisode初始环境信息已保存到: {metadata_file}")
    print("你可以直接读取这个JSON文件来快速查看每个episode的obj_pose值。")


def main():
    parser = argparse.ArgumentParser(
        description="从0开始采集Meta-World指定任务数据，保存为LeRobot数据集格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 采集10个episode的pick-place-v3数据（随机化物体位置）
  python collect_metaworld_dataset.py \\
      --task pick-place-v3 \\
      --num-episodes 10 \\
      --output-dir ./outputs/metaworld_pick_place \\
      --repo-id your-username/metaworld_pick_place \\
      --randomize-obj

  # 采集50个episode，使用固定物体位置
  python collect_metaworld_dataset.py \\
      --task assembly-v3 \\
      --num-episodes 50 \\
      --output-dir ./outputs/metaworld_assembly \\
      --repo-id your-username/metaworld_assembly \\
      --no-randomize-obj

  # 快速测试（仅采集2个episode）
  python collect_metaworld_dataset.py \\
      --task push-v3 \\
      --num-episodes 2 \\
      --output-dir ./outputs/test \\
      --repo-id test/push-v3-test \\
      --randomize-obj --seed-start 100
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="pick-place-v3",
        help="Meta-World任务名称，如 pick-place-v3, assembly-v3, push-v3 等 (默认: pick-place-v3)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="要采集的episode数量 (默认: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="数据集本地保存路径 (默认: ./outputs/metaworld_<task>)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace仓库ID，格式: 用户名/数据集名 (默认: lerobot/metaworld_<task>)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=80,
        help="视频帧率，Meta-World默认80fps (默认: 80)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=480,
        help="相机图像分辨率，宽高相同 (默认: 480)",
    )
    parser.add_argument(
        "--randomize-obj",
        action="store_true",
        default=True,
        help="是否随机化物体初始位置和目标位置（通过不同seed实现，每个episode使用不同seed）(默认: True)",
    )
    parser.add_argument(
        "--no-randomize-obj",
        action="store_true",
        default=False,
        help="禁用物体位置随机化，所有episode使用相同的固定位置（seed=42）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="随机化起始seed，每个episode使用 seed_start + episode_index (默认: 0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="每个episode的最大步数 (默认: 500)",
    )

    args = parser.parse_args()

    # 处理参数
    randomize_obj = args.randomize_obj and not args.no_randomize_obj

    if args.output_dir is None:
        task_short = args.task.replace("-v3", "").replace("-", "_")
        args.output_dir = f"./outputs/metaworld_{task_short}"

    if args.repo_id is None:
        task_short = args.task.replace("-v3", "").replace("-", "_")
        args.repo_id = f"lerobot/metaworld_{task_short}"

    output_dir = Path(args.output_dir)
    # LeRobotDataset.create 要求目录不存在，如果已存在则删除重建
    if output_dir.exists():
        print(f"警告: 输出目录已存在，将删除并重建: {args.output_dir}")
        import shutil
        shutil.rmtree(output_dir)

    print("=" * 80)
    print("Meta-World 数据采集")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"Episode数量: {args.num_episodes}")
    print(f"输出目录: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"图像分辨率: {args.image_size}x{args.image_size}")
    print(f"物体位置随机化: {'是' if randomize_obj else '否'}")
    if randomize_obj:
        print(f"Seed范围: {args.seed_start} ~ {args.seed_start + args.num_episodes - 1}")
    print(f"最大步数/episode: {args.max_steps}")
    print("=" * 80)

    # 获取expert policy
    policy_class_name = f"Sawyer{args.task.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        policy_class = getattr(policies, policy_class_name)
        expert_policy = policy_class()
        print(f"\n使用专家策略: {policy_class_name}")
    except AttributeError:
        print(f"\n错误: 找不到任务 {args.task} 的专家策略 {policy_class_name}")
        print("可用的任务名称请参考 metaworld_config.json 或 Meta-World 文档")
        sys.exit(1)

    # 创建数据集
    print("\n创建LeRobot数据集...")
    dataset = create_dataset(args.repo_id, args.output_dir, args.fps, args.image_size)
    print(f"数据集创建成功: {args.output_dir}")

    # 采集数据
    episode_infos = []
    success_count = 0
    start_time = time.time()

    print(f"\n开始采集 {args.num_episodes} 个成功的episode...")
    print("-" * 80)

    ep_idx = 0
    while ep_idx < args.num_episodes:
        ep_start = time.time()

        # 设置seed
        seed = args.seed_start + ep_idx if randomize_obj else 42

        # 创建两个环境（不同相机视角）
        env_top = create_metaworld_env(args.task, seed=seed, camera_name="corner2")
        env_wrist = create_metaworld_env(args.task, seed=seed, camera_name="gripperPOV")

        # 运行episode
        frames, ep_info = run_episode(env_top, env_wrist, expert_policy, args.task, args.max_steps, args.image_size)

        env_top.close()
        env_wrist.close()

        # 只保存成功的episode
        if ep_info["success"]:
            # 添加frame到数据集
            for frame in frames:
                dataset.add_frame(frame)

            # 保存episode
            dataset.save_episode()

            # 记录episode信息
            episode_infos.append(ep_info)
            success_count += 1

            elapsed = time.time() - ep_start
            print(
                f"Episode {ep_idx + 1:3d}/{args.num_episodes} | "
                f"Frames: {ep_info['num_frames']:4d} | "
                f"Success: ✓ | "
                f"Obj pose: [{ep_info['obj_init_pos'][0]:.3f}, {ep_info['obj_init_pos'][1]:.3f}, {ep_info['obj_init_pos'][2]:.3f}] | "
                f"Time: {elapsed:.2f}s"
            )
            ep_idx += 1
        else:
            elapsed = time.time() - ep_start
            print(
                f"Episode FAILED (seed={seed}) | "
                f"Frames: {ep_info['num_frames']:4d} | "
                f"Retrying with next seed... | "
                f"Time: {elapsed:.2f}s"
            )
            # 不增加ep_idx，继续尝试下一个seed

    # 完成数据集
    print("\n" + "-" * 80)
    print("正在保存数据集...")
    dataset.finalize()

    # 保存episode初始环境信息
    save_episode_metadata(args.output_dir, episode_infos, args.task)

    # 打印统计信息
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("采集完成！")
    print("=" * 80)
    print(f"总Episode数: {args.num_episodes}")
    print(f"成功Episode: {success_count} ({success_count / args.num_episodes * 100:.1f}%)")
    print(f"总用时: {total_time:.1f}s")
    print(f"平均每Episode: {total_time / args.num_episodes:.2f}s")
    print(f"数据集路径: {args.output_dir}")
    print(f"\n快速查看obj_pose:")
    print(f"  cat {args.output_dir}/episode_initial_states.json | python -m json.tool")
    print("=" * 80)


if __name__ == "__main__":
    main()