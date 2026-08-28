"""
检查lerobot/metaworld_mt50这个数据集
中pick-place-v3相关的数据集
以episode为单位，得到每一episode的first frame中environment_state[4:7]的值
"""

import os

# 设置 HuggingFace 镜像（可选）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置 LeRobot 数据集本地存储路径
# os.environ["HF_LEROBOT_HOME"] = "/data/zhonglinye/hfdata/lerobot"

import numpy as np
from pathlib import Path
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

REPO_ID = "lerobot/metaworld_mt50"
TASK_NAME = "pick-place-v3"

# pick-place-v3 的任务描述（根据你提供的输出）
TASK_DESCRIPTION = "Pick and place a puck to a goal"

def main():
    print("=" * 80)
    print(f"检查数据集: {REPO_ID}")
    print(f"目标任务: {TASK_NAME}")
    print(f"任务描述: {TASK_DESCRIPTION}")
    print("=" * 80)

    # 1. 加载数据集元数据
    print("\n加载数据集元数据...")
    dataset_meta = LeRobotDatasetMetadata(REPO_ID)
    print(f"总 episode 数: {dataset_meta.total_episodes}")
    print(f"总 frame 数: {dataset_meta.total_frames}")

    # 2. 查找 pick-place-v3 对应的 episode
    print(f"\n查找 {TASK_NAME} 相关的 episode...")
    
    # 将 episodes 转换为 pandas DataFrame
    episodes_df = dataset_meta.episodes.to_pandas()
    
    # 根据任务描述筛选 episode
    pick_place_episodes = []
    for i in range(len(episodes_df)):
        ep = episodes_df.iloc[i]
        tasks = ep.get("tasks", [])
        # tasks 是列表，如 ['Pick up a nut and place it onto a peg']
        if TASK_DESCRIPTION in tasks:
            pick_place_episodes.append(i)
            print(i)
    
    print(f"找到 {len(pick_place_episodes)} 个 {TASK_NAME} 的 episode")
    
    if not pick_place_episodes:
        print("未找到匹配的 episode！")
        print("\n数据集中实际存在的任务描述（前 20 个唯一值）:")
        # 收集所有唯一的任务描述
        all_tasks = set()
        for i in range(min(100, len(episodes_df))):
            ep = episodes_df.iloc[i]
            tasks = ep.get("tasks", [])
            for t in tasks:
                all_tasks.add(t)
        for idx, t in enumerate(sorted(all_tasks)):
            print(f"  {idx+1}. {t}")
        return

    # 打印前几个匹配的 episode 用于验证
    print("\n前 5 个匹配 episode 的信息:")
    for idx in pick_place_episodes[:5]:
        ep = episodes_df.iloc[idx]
        print(f"  Episode {idx}: tasks={ep.get('tasks', [])}, length={ep.get('length', 'N/A')}")

    # 3. 加载数据集（只加载相关 episode）
    print(f"\n加载 {TASK_NAME} 的 episode 数据...")
    dataset = LeRobotDataset(REPO_ID, episodes=pick_place_episodes)
    print(f"加载完成: {dataset.num_episodes} 个 episode, {dataset.num_frames} 个 frame")

    # 4. 提取每个 episode 第一帧的 environment_state[4:7]
    print(f"\n提取每个 episode 第一帧的 environment_state[4:7]（物体位置 xyz）...")
    print("-" * 80)
    
    results = []
    
    for ep_idx, global_ep_idx in enumerate(pick_place_episodes):
        # 获取该 episode 在数据集中的起始 frame 索引
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        
        # 获取第一帧
        first_frame = dataset[from_idx]
        
        # 提取 environment_state
        env_state = first_frame["observation.environment_state"]
        
        # 转换为 numpy 数组
        if hasattr(env_state, "numpy"):
            env_state_np = env_state.numpy()
        else:
            env_state_np = np.array(env_state)
        
        # 提取 [4:7] 索引的值（物体位置 xyz）
        obj_position = env_state_np[4:7]
        goal_position = env_state_np[7:10]
        
        results.append({
            "episode_index": global_ep_idx,
            "obj_position": obj_position,
        })
        
        # 只打印前 10 个和最后 5 个
        if ep_idx < 10 or ep_idx >= len(pick_place_episodes) - 5:
            print(f"Episode {global_ep_idx}: environment_state[4:7] = {obj_position}")
            print(f"  environment_state[7:10] = {goal_position}")
        elif ep_idx == 10:
            print("  ...")
    
    # 5. 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    
    all_positions = np.array([r["obj_position"] for r in results])
    
    print(f"\n共处理 {len(results)} 个 episode")
    print(f"\n物体位置 (environment_state[4:7]) 统计:")
    print(f"  X 轴 - 均值: {all_positions[:, 0].mean():.4f}, 标准差: {all_positions[:, 0].std():.4f}, 范围: [{all_positions[:, 0].min():.4f}, {all_positions[:, 0].max():.4f}]")
    print(f"  Y 轴 - 均值: {all_positions[:, 1].mean():.4f}, 标准差: {all_positions[:, 1].std():.4f}, 范围: [{all_positions[:, 1].min():.4f}, {all_positions[:, 1].max():.4f}]")
    print(f"  Z 轴 - 均值: {all_positions[:, 2].mean():.4f}, 标准差: {all_positions[:, 2].std():.4f}, 范围: [{all_positions[:, 2].min():.4f}, {all_positions[:, 2].max():.4f}]")
    
    # 保存结果
    output_file = Path(__file__).parent / "pick_place_obj_positions.npy"
    np.save(output_file, all_positions)
    print(f"\n结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()