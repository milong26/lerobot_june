"""
训练与评测选中子集

根据贪心结果选中对应的 episode，训练模型并评测，
输出到 personal/work2/our/eval_results/
"""

import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


OBJ_LOW = np.array([-0.1, 0.6, 0.02])
OBJ_HIGH = np.array([0.1, 0.7, 0.02])
GRID_SHAPE = (14, 8)


def pos_to_grid_coord(obj_pos: np.ndarray) -> Tuple[int, int]:
    """将物体位置转换为网格坐标"""
    x, y = obj_pos[0], obj_pos[1]
    
    x_range = OBJ_HIGH[0] - OBJ_LOW[0]
    y_range = OBJ_HIGH[1] - OBJ_LOW[1]
    
    i = int((x - OBJ_LOW[0]) / x_range * GRID_SHAPE[0])
    j = int((y - OBJ_LOW[1]) / y_range * GRID_SHAPE[1])
    
    i = max(0, min(i, GRID_SHAPE[0] - 1))
    j = max(0, min(j, GRID_SHAPE[1] - 1))
    
    return (i, j)


def select_episodes_by_planning_result(
    planning_result: Dict[Tuple[int, int], int],
    dataset_dir: Path
) -> List[int]:
    """
    根据规划结果选中对应的episode索引
    
    参数:
        planning_result: Dict[grid_coord, repeat_count]
        dataset_dir: 数据集目录
    
    返回:
        选中的episode索引列表
    """
    meta_file = dataset_dir / "episode_initial_states.json"
    if not meta_file.exists():
        print(f"警告: 元数据文件不存在: {meta_file}")
        return []
    
    with open(meta_file) as f:
        meta = json.load(f)
        episodes = meta.get("episodes", [])
    
    grid_to_episodes: Dict[Tuple[int, int], List[int]] = {}
    
    for ep_idx, ep in enumerate(episodes):
        obj_pos = ep.get("obj_init_pos")
        if obj_pos is None:
            continue
        
        coord = pos_to_grid_coord(np.array(obj_pos))
        if coord not in grid_to_episodes:
            grid_to_episodes[coord] = []
        grid_to_episodes[coord].append(ep_idx)
    
    selected_episodes = []
    
    for grid_coord, repeat_count in planning_result.items():
        available = grid_to_episodes.get(grid_coord, [])
        selected = available[:repeat_count]
        selected_episodes.extend(selected)
    
    selected_episodes = sorted(list(set(selected_episodes)))
    
    print(f"选中 {len(selected_episodes)} 个episodes")
    print(f"  episode索引: {selected_episodes[:20]}{'...' if len(selected_episodes) > 20 else ''}")
    
    return selected_episodes


def run_training(
    selected_episodes: List[int],
    dataset_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    n_steps: int = 200
):
    """
    运行训练
    
    参数:
        selected_episodes: 选中的episode索引列表
        dataset_dir: 数据集目录
        output_dir: 输出目录
        gpu_id: GPU ID
        n_steps: 训练步数
    """
    episodes_str = "[" + ",".join(map(str, selected_episodes)) + "]"
    
    train_script = PROJECT_ROOT / "personal/work2/duibi/train_and_eval_scripts/train_and_eval.sh"
    
    if not train_script.exists():
        print(f"错误: 训练脚本不存在: {train_script}")
        return
    
    cmd = [
        "bash", str(train_script),
        "dynamicanchor",
        str(len(selected_episodes)),
        "42",
        str(gpu_id),
        str(output_dir)
    ]
    
    print(f"执行训练命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"训练失败，返回码: {result.returncode}")
    else:
        print("训练完成")


def run_evaluation(
    checkpoint_dir: Path,
    output_dir: Path,
    gpu_id: int = 0
):
    """
    运行评估
    
    参数:
        checkpoint_dir: 检查点目录
        output_dir: 输出目录
        gpu_id: GPU ID
    """
    eval_results_dir = output_dir / "eval_results"
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"评估结果将保存到: {eval_results_dir}")
    
    from lerobot.envs.metaworld import MetaworldEnv
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import torch
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_dir))
    policy.to(device)
    policy.eval()
    
    env = MetaworldEnv(
        task="pick-place-v3",
        image_size=480,
        fps=80
    )
    
    n_episodes = 50
    success_count = 0
    results = []
    
    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        
        episode_success = False
        
        for step in range(200):
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get("success", False):
                episode_success = True
                break
        
        if episode_success:
            success_count += 1
        
        results.append({
            "episode": ep_idx,
            "success": episode_success
        })
        
        print(f"  Episode {ep_idx}: {'success' if episode_success else 'fail'}")
    
    success_rate = success_count / n_episodes * 100
    
    print(f"\n评估完成: {success_count}/{n_episodes} = {success_rate:.1f}%")
    
    output_file = eval_results_dir / "dynamicanchor_eval.json"
    with open(output_file, "w") as f:
        json.dump({
            "success_rate": success_rate,
            "n_success": success_count,
            "n_total": n_episodes,
            "results": results
        }, f, indent=2)
    
    print(f"评估结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="训练与评测选中子集")
    parser.add_argument("--planning-result", type=str, required=True,
                       help="规划结果JSON文件路径")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="数据集目录")
    parser.add_argument("--output-dir", type=str,
                       default="personal/work2/our/eval_results",
                       help="输出目录")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=200,
                       help="训练步数")
    args = parser.parse_args()
    
    with open(args.planning_result) as f:
        planning_result = json.load(f)
    
    planning_result = {
        tuple(map(int, k.strip("()").split(","))): v
        for k, v in planning_result.items()
    }
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selected_episodes = select_episodes_by_planning_result(
        planning_result, dataset_dir
    )
    
    if not selected_episodes:
        print("没有选中任何episodes，退出")
        return
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    run_training(
        selected_episodes=selected_episodes,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        gpu_id=args.gpu_id,
        n_steps=args.n_steps
    )
    
    run_evaluation(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        gpu_id=args.gpu_id
    )


if __name__ == "__main__":
    main()