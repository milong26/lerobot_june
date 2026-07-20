from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 加载数据集
ds = LeRobotDataset(repo_id="click_alarmclock/aloha-agilex_clean_50_v2",
                    root="/home/qwe/.cache/huggingface/lerobot/robotwin/" \
                    "place_can_basket/aloha-agilex_clean_50_v2/")
    
# 2. 随机选一集
ep_idx = 29
print(f"选中 episode {ep_idx}，共 {ds.num_episodes} 集")

# 3. 获取该集的元数据
ep_meta = ds.meta.episodes[ep_idx]
print(ep_meta)

# 4. 获取该集在全局数据集中的帧范围（新索引方式）
from_idx = ds.episode_data_index["from"][ep_idx].item()
to_idx = ds.episode_data_index["to"][ep_idx].item()
print(f"episode {ep_idx} 帧范围: [{from_idx}, {to_idx}), 共 {to_idx - from_idx} 帧")

# 第 50 个帧（0-indexed 的 49）
frame_idx = from_idx + 49
print(f"第 50 帧的全局索引: {frame_idx}")

# 方式 1：完整帧（含解码后的视频帧，较慢）
frame = ds[frame_idx]

# 查看有哪些 key

# 取 state 和 action
print("action:", frame["action"])
print("state:", frame["observation.state"])

# 取所有 observation（图像）
for key in frame:
    if key.startswith("observation."):
        val = frame[key]
        if val.ndim >= 2:
            print(f"{key}: dtype={val.dtype}, shape={val.shape}, first_5={val.flatten()[:5]}")
        else:
            print(f"{key}: dtype={val.dtype}, shape={val.shape}, first_5={val[:5].tolist()}")