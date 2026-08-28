"""
探针Rollout主流程

对指定checkpoint，在112格点 × 5种条件（nominal, noise-L1, noise-L2, layout-L1, layout-L2）
下跑eval，记录成功/失败。

已废弃：camera轴（对固定摄像头的单任务部署没有诊断价值）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

# TODO: 指定gpu id，默认用0
# 必须在任何mujoco/gymnasium import之前设置
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import yaml
import torch

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加tanzhen目录到path
tanzhen_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tanzhen_dir))

from perturb.noise_perturb import apply_noise, get_noise_config
from perturb.layout_perturb import perturb_layout, get_layout_config


def load_checkpoint_config(checkpoint_name: str, config: dict) -> dict:
    """加载指定checkpoint的配置"""
    return config["checkpoints"][checkpoint_name]


def load_grid_points(subset_file: str, dataset_metadata: str) -> list:
    """
    从subset文件和dataset metadata加载112个格点的物体位置

    参数:
        subset_file: subset JSON文件路径
        dataset_metadata: episode_initial_states.json路径

    返回:
        list of {grid_id, obj_pos: [x, y, z], goal_pos: [x, y, z]}
    """
    with open(subset_file) as f:
        subset_data = json.load(f)

    with open(dataset_metadata) as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]

    # subset文件格式: {"method": "random", "num_episodes": 112, "seed": 42, "selected_episode_indices": [...]}
    if isinstance(subset_data, list):
        episode_indices = subset_data
    elif isinstance(subset_data, dict):
        episode_indices = subset_data.get("selected_episode_indices", 
                           subset_data.get("episodes", 
                           subset_data.get("included_episodes", [])))
    else:
        raise ValueError(f"未知的subset文件格式: {subset_file}")

    grid_points = []
    for idx in episode_indices:
        episode = episodes[idx]
        grid_points.append({
            "grid_id": idx,
            "obj_pos": episode["obj_init_pos"],
            "goal_pos": episode.get("goal_pose", [0.1, 0.8, 0.2]),  # 默认goal位置
        })

    return grid_points


def create_metaworld_env(task_name: str = "pick-place-v3", seed: int = None):
    """
    创建Meta-World环境

    参数:
        task_name: 任务名称
        seed: 随机种子

    返回:
        env: Meta-World环境实例
    """
    import metaworld

    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name="corner2")
    task = mt1.train_tasks[0]
    env.set_task(task)
    env._freeze_rand_vec = True

    return env


def load_model(checkpoint_path: str, device: torch.device):
    """
    加载SmolVLA模型checkpoint

    参数:
        checkpoint_path: checkpoint路径
        device: 设备

    返回:
        (policy, preprocessor, postprocessor): 模型和处理器
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"加载模型: {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy = policy.to(device)
    policy.eval()
    
    # 创建preprocessor和postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
        dataset_stats=None,  # 不使用归一化
    )
    
    print(f"模型加载成功，设备: {device}")
    
    return policy, preprocessor, postprocessor


def run_single_rollout(policy, preprocessor, postprocessor, env, obj_pos: list, goal_pos: list, 
                       condition: str, perturbation_params: dict, max_steps: int = 200, 
                       device: str = "cuda:0", task_description: str = "Pick up the object") -> bool:
    """
    执行单次rollout

    参数:
        policy: SmolVLA策略模型
        preprocessor: 输入预处理器
        postprocessor: 输出后处理器
        env: Meta-World环境实例
        obj_pos: 物体位置 [x, y, z]
        goal_pos: 目标位置 [x, y, z]
        condition: 扰动条件
        perturbation_params: 扰动参数
        max_steps: 最大步数
        device: 设备
        task_description: 任务描述

    返回:
        success: 是否成功
    """
    # 应用layout扰动（修改goal位置）
    if condition.startswith("layout") and perturbation_params:
        perturb_layout(env, condition.split("-")[1], obj_pos, goal_pos, params=perturbation_params)

    # 重置环境
    obs, info = env.reset()

    success = False
    for step in range(max_steps):
        # 渲染图像
        img = env.render()  # 返回 H x W x 3 的numpy数组，值范围[0, 255]

        # 应用噪声扰动（如果适用）
        if condition.startswith("noise") and perturbation_params:
            img = apply_noise(img, condition.split("-")[1], params=perturbation_params)

        # 准备模型输入batch
        # SmolVLA需要的格式（根据checkpoint config.json）:
        # - observation.state: 机器人状态 [6]
        # - observation.images.camera1: 全局相机图像 [3, 256, 256]
        # - observation.images.camera2: 手腕相机图像 [3, 256, 256]
        # - observation.images.camera3: 第三相机图像 [3, 256, 256]
        # - observation.images.empty_camera_0: 空相机 [3, 480, 640]
        # - task: 任务描述
        
        # 从环境获取状态（joint positions等）
        # MetaWorld环境的observation格式取决于具体实现
        try:
            # 尝试获取内部状态
            if hasattr(env, 'observation'):
                state_dict = env.observation
                if isinstance(state_dict, dict):
                    state_vec = state_dict.get('state', None)
                else:
                    state_vec = state_dict
            elif hasattr(env, 'get_obs'):
                state_dict = env.get_obs()
                if isinstance(state_dict, dict):
                    state_vec = state_dict.get('state', None)
                else:
                    state_vec = state_dict
            elif hasattr(env, '_get_obs'):
                state_dict = env._get_obs()
                if isinstance(state_dict, dict):
                    state_vec = state_dict.get('state', None)
                else:
                    state_vec = state_dict
            else:
                # 如果无法获取，使用默认值
                state_vec = None
        except Exception as e:
            print(f"  警告: 获取环境状态失败 ({e})，使用默认值")
            state_vec = None
        
        # 如果state_vec是numpy数组，转换为tensor
        if state_vec is not None and isinstance(state_vec, np.ndarray):
            # 确保state是4维（根据数据集实际格式：xyz + gripper）
            if state_vec.shape[0] != 4:
                # 如果维度不匹配，填充或截断
                if state_vec.shape[0] < 4:
                    state_vec = np.pad(state_vec, (0, 4 - state_vec.shape[0]))
                else:
                    state_vec = state_vec[:4]
            state_tensor = torch.from_numpy(state_vec.copy()).float()
        else:
            # 如果无法获取状态，使用默认值（4维状态：xyz + gripper）
            state_tensor = torch.zeros(4, dtype=torch.float32)
        
        # 图像预处理：HWC -> CHW
        if isinstance(img, np.ndarray):
            # env.render()返回的是全局相机视角
            # 使用.copy()解决负步长问题
            img_tensor = torch.from_numpy(img.copy()).float().permute(2, 0, 1)
        else:
            img_tensor = img
        
        # 构建原始batch（不带batch维度，preprocessor会添加）
        # 根据preprocessor的rename_map，输入应该使用原始键名：
        # - observation.images.top -> observation.images.camera1
        # - observation.images.wrist -> observation.images.camera2
        # 注意：camera3和empty_camera_0不在rename_map中，需要直接提供
        raw_batch = {
            "observation.state": state_tensor,
            "observation.images.top": img_tensor,  # 会被rename为camera1
            "observation.images.wrist": img_tensor,  # 会被rename为camera2
            "observation.images.camera3": img_tensor,  # 第三相机
            "observation.images.empty_camera_0": img_tensor,  # 空相机
            "task": task_description,
        }
        
        # 使用preprocessor处理batch（会添加batch维度、tokenize等）
        try:
            batch = preprocessor(raw_batch)
        except Exception as e:
            print(f"  警告: preprocessor处理失败 ({e})")
            return False
        
        # 模型推理
        with torch.no_grad():
            # 使用select_action获取单个动作
            action = policy.select_action(batch)
        
        # 将action转换为numpy数组
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy().squeeze()
        else:
            action_np = action.squeeze()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action_np)
        success = info.get("success", False)

        if terminated or truncated:
            break

    return success


def run_probe_for_checkpoint(
    checkpoint_name: str,
    checkpoint_path: str,
    subset_file: str,
    dataset_metadata: str,
    config: dict,
    n_repeats: int = 1,
    output_dir: str = "results/probe_raw",
):
    """
    对单个checkpoint执行完整的探针评测

    参数:
        checkpoint_name: checkpoint名称（如"random_42"）
        checkpoint_path: checkpoint路径
        subset_file: subset文件路径
        dataset_metadata: dataset metadata路径
        config: 完整配置字典
        n_repeats: 每个格点每个条件的rollout次数
        output_dir: 输出目录
    """
    # 设置设备
    # 注意：当设置了CUDA_VISIBLE_DEVICES后，PyTorch只能看到该GPU，索引为0
    # 所以无论CUDA_VISIBLE_DEVICES设置为什么，这里都应该用cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载格点
    grid_points = load_grid_points(subset_file, dataset_metadata)
    print(f"加载了 {len(grid_points)} 个格点")

    # 加载模型（返回policy, preprocessor, postprocessor）
    policy, preprocessor, postprocessor = load_model(checkpoint_path, device)

    # 创建环境
    env = create_metaworld_env()

    # 定义测试条件（不包含camera轴）
    conditions = ["nominal"]
    if config["perturbations"]["noise"]["enabled"]:
        conditions.extend(["noise-L1", "noise-L2"])
    if config["perturbations"].get("layout", {}).get("enabled", False):
        conditions.extend(["layout-L1", "layout-L2"])

    # 执行评测
    all_results = []
    for condition in conditions:
        print(f"\n=== 测试条件: {condition} ===")

        # 获取扰动参数
        if condition == "nominal":
            perturbation_params = None
        elif condition.startswith("noise"):
            level = condition.split("-")[1]
            perturbation_params = get_noise_config(level)
        elif condition.startswith("layout"):
            level = condition.split("-")[1]
            perturbation_params = get_layout_config(level)
        else:
            perturbation_params = None

        condition_results = []
        # 使用进度条显示测试进度
        with tqdm(total=len(grid_points), desc=f"  [{condition}] 进度", unit="格点") as pbar:
            for grid_point in grid_points:
                grid_id = grid_point["grid_id"]
                obj_pos = grid_point["obj_pos"]
                goal_pos = grid_point["goal_pos"]

                successes = 0
                for repeat in range(n_repeats):
                    success = run_single_rollout(
                        policy, preprocessor, postprocessor, env, obj_pos, goal_pos, condition, perturbation_params,
                        max_steps=config["rollout"]["max_steps"],
                        device=device,
                    )
                    if success:
                        successes += 1

                success_rate = successes / n_repeats if n_repeats > 0 else 0
                condition_results.append({
                    "grid_id": grid_id,
                    "obj_pos": obj_pos,
                    "goal_pos": goal_pos,
                    "successes": successes,
                    "n_repeats": n_repeats,
                    "success_rate": success_rate,
                })
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "成功率": f"{success_rate:.0%}",
                    "累计成功": f"{sum(r['successes'] for r in condition_results)}/{len(condition_results)}"
                })

        # 保存该条件的结果
        output_file = Path(output_dir) / f"{checkpoint_name}_{condition}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "checkpoint": checkpoint_name,
                "condition": condition,
                "n_repeats": n_repeats,
                "results": condition_results,
            }, f, indent=2)

        print(f"  条件 {condition} 的结果已保存到 {output_file}")
        
        # 打印该条件的汇总统计
        total_successes = sum(r["successes"] for r in condition_results)
        total_grid_points = len(condition_results)
        avg_success_rate = total_successes / (total_grid_points * n_repeats) if n_repeats > 0 else 0
        print(f"  [{condition}] 汇总: {total_successes}/{total_grid_points * n_repeats} 格点成功 (平均成功率: {avg_success_rate:.1%})")
        
        all_results.extend(condition_results)

    env.close()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="探针Rollout评测")
    parser.add_argument("--config", type=str, default="personal/work2/tanzhen/configs/probe_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="指定要测试的checkpoint名称，不指定则使用active_checkpoint")
    parser.add_argument("--n-repeats", type=int, default=None, help="覆盖配置中的rollout次数")
    parser.add_argument("--smoke-test", action="store_true", help="冒烟测试模式（每格点1次rollout）")
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 确定rollout次数
    if args.smoke_test:
        n_repeats = config["rollout"]["n_repeats_smoke"]
    elif args.n_repeats is not None:
        n_repeats = args.n_repeats
    else:
        n_repeats = config["rollout"]["n_repeats_final"]

    print(f"Rollout次数: {n_repeats}")

    # 确定要测试的checkpoint
    if args.checkpoint:
        checkpoints_to_test = [args.checkpoint]
    else:
        # 使用active_checkpoint
        checkpoints_to_test = [config.get("active_checkpoint", "random_42")]

    # 执行评测
    for ckpt_name in checkpoints_to_test:
        ckpt_config = config["checkpoints"][ckpt_name]
        print(f"\n{'='*60}")
        print(f"测试Checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        run_probe_for_checkpoint(
            checkpoint_name=ckpt_name,
            checkpoint_path=ckpt_config["path"],
            subset_file=ckpt_config["subset_file"],
            dataset_metadata=config["dataset_metadata"],
            config=config,
            n_repeats=n_repeats,
            output_dir=config["output"]["raw_dir"],
        )

    print("\n所有探针评测完成！")


if __name__ == "__main__":
    main()