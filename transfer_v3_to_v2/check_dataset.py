from lerobot.datasets import LeRobotDataset
import random
 
# 1. 加载数据集
ds = LeRobotDataset(repo_id="lerobot/pusht",
                    root="/home/qwe/.cache/huggingface/lerobot/robotwin/" \
                    "place_can_basket/aloha-agilex_clean_50/")
 
# 2. 随机选一集
ep_idx = 29
print(f"随机选中 episode {ep_idx}，共 {ds.num_episodes} 集")
 
# 3. 获取该集的帧范围（来自 meta/episodes 元数据）
ep_meta = ds.meta.episodes[ep_idx]
print(ep_meta)
# 你已经知道 ep_meta 了，提取帧范围
from_idx = ep_meta["dataset_from_index"]
to_idx = ep_meta["dataset_to_index"]
 
# 第 50 个帧（1-indexed 的"第50个" → 0-indexed 的 49）
frame_idx = from_idx + 49  # = 976
 
# ── 方式 1：完整帧（含解码后的视频帧，较慢）──
frame = ds[frame_idx]
 
# 查看有哪些 key
print("所有字段:", list(frame.keys()))
 
# 取 state 和 action
print("action:", frame["action"])         # tensor
print("state:", frame["observation.state"])  # tensor，具体 key 名取决于数据集
 
# 取所有 observation（图像）
# 取所有 observation（图像）
for key in frame:
    if key.startswith("observation."):
        val = frame[key]
        if val.ndim >= 2:
            print(f"{key}: dtype={val.dtype}, shape={val.shape}, first_5={val.flatten()[:5]}")
        else:
            print(f"{key}: dtype={val.dtype}, shape={val.shape}, first_5={val[:5].tolist()}")