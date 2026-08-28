"""
检查lerobot/metaworld_mt50这个数据集
中pick-place-v3相关的数据集
以episode为单位，得到每一episode的first frame中environment_state[4:7]的值
"""

import os

# 设置 HuggingFace 使用本地缓存
# os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_LEROBOT_HOME"] = "/data/zhonglinye/hfdata/lerobot"

import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

REPO_ID = "lerobot/metaworld_mt50"
TASK_NAME = "pick-place-v3"


def main():
    print("=" * 80)
    print(f"检查数据集: {REPO_ID}")
    print(f"目标任务: {TASK_NAME}")
    print("=" * 80)

    # 1. 加载数据集元数据
    print("\n加载数据集元数据...")
    dataset_meta = LeRobotDatasetMetadata(REPO_ID)
    print(f"总 episode 数: {dataset_meta.total_episodes}")
    print(f"总 frame 数: {dataset_meta.total_frames}")

    # 2. 查找 pick-place-v3 对应的 episode
    print(f"\n查找 {TASK_NAME} 相关的 episode...")
    
    # 根据任务名称筛选 episode
    pick_place_episodes = []
    episodes_df = dataset_meta.episodes.to_pandas()  # 先转换
    for i in range(len(episodes_df)):
        ep = episodes_df.iloc[i]
        # 检查 tasks 字段是否包含 pick-place-v3
        tasks = ep.get("tasks", [])
        if TASK_NAME in str(tasks):
            pick_place_episodes.append(i)
    
    print(f"找到 {len(pick_place_episodes)} 个 {TASK_NAME} 的 episode")
    
    if not pick_place_episodes:
        print("未找到相关 episode，尝试使用 task_index 筛选...")
        # 备选方案：如果 tasks 字段不可用，尝试其他方式
        # pick-place-v3 在 MT50 中通常是特定的 task_index
        for i in range(len(dataset_meta.episodes)):
            ep = episodes_df.iloc[i]
            task_index = ep.get("task_index", None)
            if task_index is not None:
                pick_place_episodes.append(i)
        print(f"找到 {len(pick_place_episodes)} 个 episode（基于 task_index）")
    
    if not pick_place_episodes:
        print("错误: 无法找到任何 episode")
        return

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
        
        results.append({
            "episode_index": global_ep_idx,
            "obj_position": obj_position,
        })
        
        print(f"Episode {global_ep_idx:4d}: environment_state[4:7] = {obj_position}")
    
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
    from pathlib import Path
    main()