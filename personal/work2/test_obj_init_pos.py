"""
测试从 environment_state 中提取 obj_init_pos 的评估流程

加载预训练模型，运行 1 个 episode 的评估，验证能否成功获取物体初始位置。
使用 LeRobot 的预处理器来处理语言 token 和其他预处理步骤。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# 设置 MuJoCo 使用 EGL 渲染（无头服务器）
os.environ["MUJOCO_GL"] = "egl"

# 设置 HuggingFace 使用本地缓存，不联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_policy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def extract_obj_init_pos_from_env_state(env_state, idx=0):
    """
    从 environment_state 中提取物体初始位置。
    
    仿照 fix_dataset_json_file.py 的做法：
    obj_init_pos = env_state[4:7]
    
    Args:
        env_state: environment_state 数组，shape 为 (batch, 39) 或 (39,)
        idx: 批次索引（如果是 batched 数据）
    
    Returns:
        obj_init_pos: 3D 位置 [x, y, z]
    """
    if hasattr(env_state, "numpy"):
        env_state = env_state.numpy()
    
    if env_state.ndim == 2:
        pos = env_state[idx, 4:7]
    else:
        pos = env_state[4:7]
    
    return pos.tolist()


def test_eval_with_env_state_obj_init_pos(
    checkpoint_path: str = "/data/zhonglinye/jun/lerobot/personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model",
    dataset_path: str = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3",
    task: str = "pick-place-v3",
    device: str = "cuda",
):
    """
    测试评估流程，从 environment_state 中提取 obj_init_pos。
    """
    print("=" * 80)
    print("测试：从 environment_state 提取 obj_init_pos")
    print("=" * 80)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"错误：找不到 checkpoint 路径 {checkpoint_path}")
        sys.exit(1)
    
    # 1. 加载训练配置
    print(f"\n[1] 加载训练配置")
    with open(checkpoint_path / "train_config.json", "r") as f:
        train_config = json.load(f)
    
    # 提取 policy 配置
    policy_cfg_dict = train_config["policy"]
    policy_cfg_dict["pretrained_path"] = str(checkpoint_path)
    policy_cfg_dict["device"] = device
    
    # 提取 env 配置
    env_cfg_dict = train_config["env"]
    
    print(f"  策略类型: {policy_cfg_dict['type']}")
    print(f"  环境类型: {env_cfg_dict['type']}")
    print(f"  任务: {env_cfg_dict['task']}")
    
    # 2. 加载数据集元数据
    print(f"\n[2] 加载数据集元数据")
    dataset = LeRobotDataset(
        repo_id=train_config["dataset"]["repo_id"],
        root=dataset_path,
    )
    dataset_meta = dataset.meta
    dataset_stats = dataset.meta.stats
    print(f"  数据集: {train_config['dataset']['repo_id']}")
    print(f"  Episode 数: {dataset.num_episodes}")
    print(f"  数据集统计 keys: {list(dataset_stats.keys())}")
    
    # 3. 创建策略
    print(f"\n[3] 创建策略")
    from lerobot.policies.factory import make_policy_config
    from lerobot.configs.types import PolicyFeature, FeatureType
    
    # 移除 'type' 参数，因为 make_policy_config 会通过 policy_type 参数传入
    policy_type = policy_cfg_dict.pop("type")
    
    # 将 input_features 和 output_features 从 dict 转换为 PolicyFeature 对象
    if "input_features" in policy_cfg_dict and policy_cfg_dict["input_features"]:
        converted_input = {}
        for key, val in policy_cfg_dict["input_features"].items():
            if isinstance(val, dict):
                ft = PolicyFeature(
                    type=FeatureType(val["type"]),
                    shape=tuple(val["shape"]),
                )
                converted_input[key] = ft
            else:
                converted_input[key] = val
        policy_cfg_dict["input_features"] = converted_input
    
    if "output_features" in policy_cfg_dict and policy_cfg_dict["output_features"]:
        converted_output = {}
        for key, val in policy_cfg_dict["output_features"].items():
            if isinstance(val, dict):
                ft = PolicyFeature(
                    type=FeatureType(val["type"]),
                    shape=tuple(val["shape"]),
                )
                converted_output[key] = ft
            else:
                converted_output[key] = val
        policy_cfg_dict["output_features"] = converted_output
    
    policy_cfg = make_policy_config(
        policy_type=policy_type,
        **policy_cfg_dict,
    )
    
    # 应用 rename_map 到 cfg.input_features
    # 将策略的相机名称映射到数据集的相机名称
    rename_map = {
        "observation.images.camera1": "observation.images.top",
        "observation.images.camera2": "observation.images.wrist",
    }
    
    # 打印数据集期望的相机名称用于调试
    dataset_features = dataset_meta.features
    print(f"  数据集特征 keys: {list(dataset_features.keys())}")
    print(f"  策略输入特征 keys: {list(policy_cfg.input_features.keys())}")
    
    new_input_features = {}
    for key, ft in policy_cfg.input_features.items():
        # 跳过数据集中不存在的相机（如 camera3, empty_camera_0）
        if key in ["observation.images.camera3", "observation.images.empty_camera_0"]:
            print(f"  跳过策略特征: {key} (数据集中不存在)")
            continue
        new_key = rename_map.get(key, key)
        new_input_features[new_key] = ft
    policy_cfg.input_features = new_input_features
    
    print(f"  重命名后策略输入特征 keys: {list(policy_cfg.input_features.keys())}")
    
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=dataset_meta,
    )
    policy.eval()
    print(f"  策略创建成功")
    
    # 4. 创建预处理器
    print(f"\n[4] 创建预处理器")
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        config=policy_cfg,
        dataset_stats=dataset_stats,
    )
    print(f"  预处理器创建成功")
    
    # 5. 创建环境
    print(f"\n[5] 创建 Meta-World 环境")
    from lerobot.envs.metaworld import MetaworldEnv as MWEnv
    
    env = MWEnv(
        task=task,
        camera_name="corner2,gripperPOV",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    print(f"  环境创建成功")
    
    # 6. 运行单个 episode
    print(f"\n[6] 运行评估 episode")
    policy.reset()
    
    obs, info = env.reset(seed=42)
    
    # 检查 observation 中的 keys
    print(f"  Observation keys: {list(obs.keys())}")
    
    # 尝试从 info 中获取 obj_init_pos
    obj_init_pos_from_info = info.get("obj_init_pos")
    print(f"  从 info['obj_init_pos'] 获取: {obj_init_pos_from_info}")
    
    # 检查 observation 中是否有 environment_state
    has_env_state = False
    for key in obs.keys():
        if "environment_state" in key or "env_state" in key:
            has_env_state = True
            env_state = obs[key]
            obj_init_pos_from_env_state = extract_obj_init_pos_from_env_state(env_state)
            print(f"  从 {key}[4:7] 提取 obj_init_pos: {obj_init_pos_from_env_state}")
            break
    
    if not has_env_state:
        print(f"  警告：observation 中没有 environment_state 键")
        print(f"  将使用 info['obj_init_pos'] 作为替代方案")
    
    # 重命名 observation keys 以匹配策略期望
    def rename_obs_keys(obs):
        """将环境的 observation keys 重命名为策略期望的格式，并转换图像格式"""
        renamed = {}
        for key, val in obs.items():
            if key == "pixels/top":
                # 转换为 tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
                if hasattr(val, "numpy"):
                    val = val.numpy()
                img = torch.from_numpy(val).permute(2, 0, 1).unsqueeze(0).float()
                renamed["observation.images.top"] = img
            elif key == "pixels/wrist":
                if hasattr(val, "numpy"):
                    val = val.numpy()
                img = torch.from_numpy(val).permute(2, 0, 1).unsqueeze(0).float()
                renamed["observation.images.wrist"] = img
            elif key == "agent_pos":
                # 转换为 tensor: (dim,) -> (1, dim)
                if hasattr(val, "numpy"):
                    val = val.numpy()
                renamed["observation.state"] = torch.from_numpy(val).unsqueeze(0).float()
            else:
                renamed[key] = val
        return renamed
    
    obs = rename_obs_keys(obs)
    print(f"  重命名后 Observation keys: {list(obs.keys())}")
    
    # 7. 运行 rollout
    print(f"\n[7] 开始 rollout")
    max_steps = 500
    done = False
    step = 0
    success = False
    reward = 0.0
    
    with torch.inference_mode():
        while not done and step < max_steps:
            # 重命名 observation keys
            obs = rename_obs_keys(obs)
            
            # 添加任务描述
            obs["task"] = "pick and place the object"
            
            # 使用预处理器处理 observation
            processed_obs = preprocessor(obs)
            
            # 选择动作
            action = policy.select_action(processed_obs)
            
            # 使用后处理器处理动作
            action = postprocessor(action)
            action_numpy = action.to("cpu").numpy()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action_numpy[0])
            done = terminated or truncated
            success = info.get("is_success", False)
            
            step += 1
            if step % 50 == 0:
                print(f"  Step {step}/{max_steps}, reward={reward:.2f}, done={done}")
    
    # 8. 输出结果
    print(f"\n{'=' * 80}")
    print(f"评估结果")
    print(f"{'=' * 80}")
    print(f"  总步数: {step}")
    print(f"  是否成功: {success}")
    print(f"  最终 reward: {reward:.2f}")
    
    if "obj_init_pos" in info:
        print(f"  info['obj_init_pos']: {info['obj_init_pos']}")
    
    # 9. 保存测试结果
    output_dir = Path(checkpoint_path).parent / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_results = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": dataset_path,
        "task": task,
        "device": device,
        "total_steps": step,
        "success": success,
        "final_reward": float(reward),
        "obj_init_pos_from_info": obj_init_pos_from_info.tolist() if hasattr(obj_init_pos_from_info, "tolist") else obj_init_pos_from_info,
        "has_environment_state": has_env_state,
        "timestamp": str(Path(__file__).stat().st_mtime),
    }
    
    output_file = output_dir / "test_obj_init_pos_results.json"
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n测试结果已保存到: {output_file}")
    
    # 清理
    env.close()
    
    print(f"\n{'=' * 80}")
    if success:
        print("测试成功！")
    else:
        print("测试完成（未成功，但流程正常）")
    print(f"{'=' * 80}")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="测试从 environment_state 提取 obj_init_pos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random_42/random_112_seed42/checkpoints/000200/pretrained_model",
        help="Checkpoint 路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3",
        help="数据集路径",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick-place-v3",
        help="Meta-World 任务名称",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda 或 cpu)",
    )
    args = parser.parse_args()
    
    test_eval_with_env_state_obj_init_pos(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        task=args.task,
        device=args.device,
    )


if __name__ == "__main__":
    main()