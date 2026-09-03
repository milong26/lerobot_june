#!/usr/bin/env python
"""
通过seed生成的。

从0开始采集Meta-World指定任务的数据，并保存为LeRobot数据集格式（视频格式）。

数据集包含：
- observation.images.top: 固定全局相机视角视频 (480x480)
- observation.images.wrist: 手腕/夹爪视角视频 (480x480)
- observation.state: 机械臂本体状态 (4维: xyz + gripper)
- observation.environment_state: 39维环境参数 (包含obj_pose等)
- action: 4维动作 (x, y, z, gripper)
- next.reward: 奖励
- next.success: 是否成功
- task: 任务名称字符串

每个episode的初始状态（obj_init_pos, goal_pose, rand_vec, reset_space等）记录在
episode_initial_states.json中，方便快速访问。

物体位置随机化说明：
- pick-place-v3任务有两个关键位置：物体初始位置(obj_init_pos)和目标位置(goal)
- 随机化时，两个位置都会随机变化
- 通过不同seed创建MT1环境，每个seed对应一组固定的obj_init_pos和goal
- 默认启用随机化，每个episode使用不同的seed

使用示例（命令行）:
    # 采集10个episode的pick-place-v3数据（随机化物体和目标位置）
    python personal/work2/collect_dataset/collect_metaworld_dataset.py \
        --task pick-place-v3 \
        --num-episodes 200 \
        --output-dir personal/work2/dataset_view/pick_place_corner3/ \
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
    --streaming-encoding: 启用流式视频编码，边采集边编码，save_episode几乎瞬时
    --encoder-threads: 每个编码器的线程数，None由codec自动决定
    --image-writer-processes: 异步图像写入的子进程数，0表示只用线程
    --image-writer-threads: 每个相机的图像写入线程数
    --batch-encoding-size: 累积多少个episode后批量编码视频，1表示立即编码
    --vcodec: 视频编码器codec，如libsvtav1(默认)/h264/h264_nvenc/h264_vaapi
    --extra-frames-after-success: 检测到成功后继续采集的帧数
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
CONFIG_PATH = Path(__file__).parent.parent.parent.parent.parent / "src" / "lerobot" / "envs" / "metaworld_config.json"
with open(CONFIG_PATH) as f:
    METAWORLD_CONFIG = json.load(f)

TASK_DESCRIPTIONS = METAWORLD_CONFIG.get("TASK_DESCRIPTIONS", {})


def create_metaworld_env(task_name, seed=None, camera_name="corner3"):
    """创建Meta-World环境，指定相机视角。"""
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    task = mt1.train_tasks[0]
    env.set_task(task)
    env._freeze_rand_vec = True
    return env


def introspect_task_space(task_name, seed=42):
    """解析当前 MetaWorld task 的完整 reset randomization space。

    返回 dict:
        space: gym.spaces.Box 对象 (_random_reset_space)
        low: np.ndarray
        high: np.ndarray
        dim: int
        has_obj: bool (是否存在 obj_init_pos 属性)
        obj_pos_dim: int (obj_init_pos 的维度，不存在则为 0)
        has_goal: bool (是否存在 goal 属性)
        goal_pos_dim: int (goal 的维度，不存在则为 0)
    """
    env = create_metaworld_env(task_name, seed=seed)
    env._freeze_rand_vec = False

    space = env._random_reset_space
    assert space is not None, f"{task_name} 的 _random_reset_space 为 None"
    low = space.low.copy()
    high = space.high.copy()
    dim = space.shape[0]

    has_obj = hasattr(env, "obj_init_pos") and env.obj_init_pos is not None
    obj_pos_dim = len(env.obj_init_pos) if has_obj else 0

    has_goal = hasattr(env, "goal") and env.goal is not None
    goal_pos_dim = len(env.goal) if has_goal else 0

    env.reset()
    env.close()

    return {
        "space": space,
        "low": low,
        "high": high,
        "dim": dim,
        "has_obj": has_obj,
        "obj_pos_dim": obj_pos_dim,
        "has_goal": has_goal,
        "goal_pos_dim": goal_pos_dim,
    }


def sync_env_state(src_env, dst_env):
    """将源环境的仿真状态同步到目标环境。"""
    dst_env.data.qpos[:] = src_env.data.qpos[:]
    dst_env.data.qvel[:] = src_env.data.qvel[:]
    dst_env.data.ctrl[:] = src_env.data.ctrl[:]
    mujoco.mj_forward(dst_env.model, dst_env.data)


def render_dual_camera(env_top, env_wrist, image_size=480):
    """渲染两个相机视角：全局固定相机 + 手腕相机。"""
    top_image = env_top.render()
    if top_image is not None:
        top_image = np.flip(top_image, (0, 1))
        top_image = resize_image(top_image, image_size)

    wrist_image = env_wrist.render()
    if wrist_image is not None:
        wrist_image = resize_image(wrist_image, image_size)

    return top_image, wrist_image


def resize_image(image, target_size):
    """使用PIL双线性插值调整图像大小。"""
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def get_obj_pose_from_env(env):
    """从环境中获取物体初始位置。"""
    if hasattr(env, "obj_init_pos") and env.obj_init_pos is not None:
        return env.obj_init_pos.copy()
    return None


def get_goal_pose_from_env(env):
    """获取目标位置。"""
    if hasattr(env, "goal") and env.goal is not None:
        return env.goal.copy()
    return None


def get_rand_vec_from_env(env):
    """获取环境的 rand_vec（reset randomization vector）。"""
    if hasattr(env, "_random_reset_space") and env._random_reset_space is not None:
        if hasattr(env, "_get_state_rand_vec"):
            try:
                rand_vec = env._get_state_rand_vec()
                return rand_vec.copy()
            except Exception:
                pass
    return None


def run_episode(env_top, env_wrist, expert_policy, task_name, max_steps=500, image_size=480, extra_frames_after_success=10):
    """运行一个完整的episode，使用expert policy生成示范数据。"""
    obs, info = env_top.reset()
    env_wrist.reset()
    sync_env_state(env_top, env_wrist)

    obj_pose = get_obj_pose_from_env(env_top)
    goal_pose = get_goal_pose_from_env(env_top)
    rand_vec = get_rand_vec_from_env(env_top)

    frames = []
    success_flags = []
    success_detected = False
    frames_after_success = 0

    for step in range(max_steps):
        action = expert_policy.get_action(obs)
        obs, reward, terminated, truncated, info = env_top.step(action)
        sync_env_state(env_top, env_wrist)

        top_image, wrist_image = render_dual_camera(env_top, env_wrist, image_size)

        if top_image is None or wrist_image is None:
            print(f"  Warning: Failed to render at step {step}, skipping frame.")
            continue

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

        current_success = info.get("success", 0)
        if current_success and not success_detected:
            success_detected = True
            frames_after_success = 0
            print(f"  >>> Success detected at step {step}, collecting {extra_frames_after_success} more frames...")

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
        "rand_vec": rand_vec,
        "success": any(success_flags),
        "num_frames": len(frames),
    }

    return frames, episode_info


def create_dataset(repo_id, output_dir, fps=80, image_size=480,
                   streaming_encoding=False, encoder_threads=None,
                   image_writer_processes=0, image_writer_threads=8,
                   batch_encoding_size=1, vcodec=None):
    """创建空的LeRobot数据集（视频格式）。"""
    from lerobot.configs.video import rgb_encoder_defaults, RGBEncoderConfig

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

    rgb_enc = rgb_encoder_defaults()
    if vcodec is not None:
        rgb_enc = RGBEncoderConfig(vcodec=vcodec)

    num_cameras = 2
    total_writer_threads = image_writer_threads * num_cameras if image_writer_threads > 0 else 0

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="metaworld",
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=total_writer_threads,
        batch_encoding_size=batch_encoding_size,
        rgb_encoder=rgb_enc,
        encoder_threads=encoder_threads,
        streaming_encoding=streaming_encoding,
    )

    return dataset


def save_episode_metadata(output_dir, episode_infos, task_name, task_space_info=None):
    """保存所有episode的初始环境信息到JSON文件。

    根据 task_space_info 动态生成 env_state_structure，保存完整的 reset space 信息。
    """
    metadata_file = Path(output_dir) / "episode_initial_states.json"

    if task_space_info:
        env_state_structure = {
            "reset_space_dim": task_space_info["dim"],
            "reset_space_low": task_space_info["low"].tolist(),
            "reset_space_high": task_space_info["high"].tolist(),
            "has_obj": task_space_info["has_obj"],
            "obj_pos_dim": task_space_info["obj_pos_dim"],
            "has_goal": task_space_info["has_goal"],
            "goal_pos_dim": task_space_info["goal_pos_dim"],
        }
    else:
        env_state_structure = {}

    metadata = {
        "task": task_name,
        "num_episodes": len(episode_infos),
        "env_state_structure": env_state_structure,
        "episodes": [],
    }

    for i, info in enumerate(episode_infos):
        ep_data = {
            "episode_index": i,
            "success": bool(info["success"]),
            "num_frames": info["num_frames"],
        }
        if info.get("obj_init_pos") is not None:
            ep_data["obj_init_pos"] = info["obj_init_pos"].tolist()
        if info.get("goal_pose") is not None:
            ep_data["goal_pose"] = info["goal_pose"].tolist()
        if info.get("rand_vec") is not None:
            ep_data["rand_vec"] = info["rand_vec"].tolist()
        metadata["episodes"].append(ep_data)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEpisode初始环境信息已保存到: {metadata_file}")


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

    parser.add_argument("--task", type=str, default="pick-place-v3",
                        help="Meta-World任务名称 (默认: pick-place-v3)")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="要采集的episode数量 (默认: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="数据集本地保存路径")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="HuggingFace仓库ID (默认: lerobot/metaworld_<task>)")
    parser.add_argument("--fps", type=int, default=80,
                        help="视频帧率 (默认: 80)")
    parser.add_argument("--image-size", type=int, default=480,
                        help="相机图像分辨率 (默认: 480)")
    parser.add_argument("--randomize-obj", action="store_true", default=True,
                        help="是否随机化物体位置 (默认: True)")
    parser.add_argument("--no-randomize-obj", action="store_true", default=False,
                        help="禁用物体位置随机化")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="随机化起始seed (默认: 0)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="每个episode的最大步数 (默认: 500)")
    parser.add_argument("--streaming-encoding", action="store_true", default=False,
                        help="启用流式视频编码，边采集边编码，save_episode几乎瞬时 (默认: False)")
    parser.add_argument("--encoder-threads", type=int, default=None,
                        help="每个编码器的线程数，None由codec自动决定 (默认: None)")
    parser.add_argument("--image-writer-processes", type=int, default=0,
                        help="异步图像写入的子进程数，0表示只用线程 (默认: 0)")
    parser.add_argument("--image-writer-threads", type=int, default=8,
                        help="每个相机的图像写入线程数 (默认: 8)")
    parser.add_argument("--batch-encoding-size", type=int, default=1,
                        help="累积多少个episode后批量编码视频，1表示立即编码 (默认: 1)")
    parser.add_argument("--vcodec", type=str, default=None,
                        help="视频编码器codec，如libsvtav1/h264/h264_nvenc，None使用默认 (默认: None)")
    parser.add_argument("--extra-frames-after-success", type=int, default=10,
                        help="检测到成功后继续采集的帧数 (默认: 10)")

    args = parser.parse_args()
    randomize_obj = args.randomize_obj and not args.no_randomize_obj

    if args.output_dir is None:
        task_short = args.task.replace("-v3", "").replace("-", "_")
        args.output_dir = f"./outputs/metaworld_{task_short}"
    if args.repo_id is None:
        task_short = args.task.replace("-v3", "").replace("-", "_")
        args.repo_id = f"lerobot/metaworld_{task_short}"

    output_dir = Path(args.output_dir)
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
    print(f"流式编码: {'是' if args.streaming_encoding else '否'}")
    if args.streaming_encoding:
        print(f"  编码器线程: {args.encoder_threads or 'auto'}")
        print(f"  视频编码器: {args.vcodec or 'libsvtav1(默认)'}")
    print(f"图像写入进程/线程: {args.image_writer_processes}P / {args.image_writer_threads}T")
    print("=" * 80)

    # 获取expert policy
    policy_class_name = f"Sawyer{args.task.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        policy_class = getattr(policies, policy_class_name)
        expert_policy = policy_class()
        print(f"\n使用专家策略: {policy_class_name}")
    except AttributeError:
        print(f"\n错误: 找不到任务 {args.task} 的专家策略 {policy_class_name}")
        sys.exit(1)

    # 解析任务的reset space
    print("\n解析任务的reset randomization space...")
    task_space_info = introspect_task_space(args.task)
    print(f"  Reset space dim: {task_space_info['dim']}")
    print(f"  Has obj: {task_space_info['has_obj']} (dim={task_space_info['obj_pos_dim']})")
    print(f"  Has goal: {task_space_info['has_goal']} (dim={task_space_info['goal_pos_dim']})")
    print(f"  Space low: {task_space_info['low'].tolist()}")
    print(f"  Space high: {task_space_info['high'].tolist()}")

    # 创建数据集
    print("\n创建LeRobot数据集（视频格式）...")
    dataset = create_dataset(
        args.repo_id, args.output_dir, args.fps, args.image_size,
        streaming_encoding=args.streaming_encoding,
        encoder_threads=args.encoder_threads,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        vcodec=args.vcodec,
    )
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
        seed = args.seed_start + ep_idx if randomize_obj else 42

        env_top = create_metaworld_env(args.task, seed=seed, camera_name="corner3")
        env_wrist = create_metaworld_env(args.task, seed=seed, camera_name="gripperPOV")

        frames, ep_info = run_episode(
            env_top, env_wrist, expert_policy, args.task,
            args.max_steps, args.image_size, args.extra_frames_after_success
        )

        env_top.close()
        env_wrist.close()

        if ep_info["success"]:
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            episode_infos.append(ep_info)
            success_count += 1

            elapsed = time.time() - ep_start
            obj_str = ""
            if ep_info.get("obj_init_pos") is not None:
                obj_str = f" | obj: [{ep_info['obj_init_pos'][0]:.3f}, {ep_info['obj_init_pos'][1]:.3f}, {ep_info['obj_init_pos'][2]:.3f}]"
            print(
                f"Episode {ep_idx + 1:3d}/{args.num_episodes} | "
                f"Frames: {ep_info['num_frames']:4d} | "
                f"Success{obj_str} | "
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

    # 完成数据集
    print("\n" + "-" * 80)
    print("正在保存数据集...")
    dataset.finalize()

    # 保存episode初始环境信息
    save_episode_metadata(args.output_dir, episode_infos, args.task, task_space_info)

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
    print("=" * 80)


if __name__ == "__main__":
    main()