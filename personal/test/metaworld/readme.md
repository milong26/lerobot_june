这个文件夹的目的是
1. 在简单代码中测试服务器能否使用metaworld
2. 测试eval_policy的时候怎么使用正常

## pick-place-v3 数据集分析结论

### 初始物体位置 (obj_init_pos)

- **50 个 episode 使用不同的随机种子生成 50 个不同的初始物体位置**
- **目标位置 (goal) 固定为 `[0.1, 0.8, 0.2]`**
- 物体位置分布：
  - X: 范围 `[-0.096, 0.099]`，均值 `-0.003`，标准差 `0.060`
  - Y: 范围 `[0.602, 0.698]`，均值 `0.643`，标准差 `0.029`
  - Z: 基本固定在 `0.02`（桌面高度）
- 这 50 个位置是通过不同 seed 随机采样的，覆盖了物体可能出现的区域

### 如何从数据集中提取初始物体位置

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载 episode
dataset = LeRobotDataset("lerobot/metaworld_mt50", episodes=[1500])

# 提取第一帧的物体位置
frame0 = dataset[0]
obj_pose = frame0['observation.environment_state'][4:7].numpy()
print(f"初始物体位置: {obj_pose}")
```

### 如何从外部对 initial pose 进行分析

```python
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import metaworld
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

REPO_ID = "lerobot/metaworld_mt50"

# 1. 找到 pick-place-v3 的所有 episode
dataset_meta = LeRobotDatasetMetadata(REPO_ID)
pick_place_episodes = []
for i in range(len(dataset_meta.episodes)):
    ep = dataset_meta.episodes[i]
    if 'Pick and place a puck to a goal' in str(ep.get('tasks', [])):
        pick_place_episodes.append(i)

# 2. 提取每个 episode 的初始物体位置
all_obj_positions = []
for ep_idx in pick_place_episodes:
    dataset = LeRobotDataset(REPO_ID, episodes=[ep_idx])
    frame0 = dataset[0]
    obj_pose = frame0['observation.environment_state'][4:7].numpy()
    all_obj_positions.append(obj_pose)

all_obj_positions = np.array(all_obj_positions)
print(f"物体位置统计:\n  X: mean={all_obj_positions[:, 0].mean():.4f}, std={all_obj_positions[:, 0].std():.4f}")
print(f"  Y: mean={all_obj_positions[:, 1].mean():.4f}, std={all_obj_positions[:, 1].std():.4f}")
print(f"  Z: mean={all_obj_positions[:, 2].mean():.4f}, std={all_obj_positions[:, 2].std():.4f}")

# 3. 创建环境获取目标位置
mt1 = metaworld.MT1("pick-place-v3", seed=42)
env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
env.set_task(mt1.train_tasks[0])
env.reset()
print(f"目标位置 (goal): {env.goal}")

# 4. 分析不同初始位置对模型泛化性的影响
# - 训练集内位置：使用现有 50 个 episode 的位置
# - 训练集外位置：修改 seed 生成新的物体位置进行测试
```

### 研究泛化性的建议

1. **插泛化测试**：在训练集见过的物体位置范围内评估模型
2. **外推泛化测试**：在训练集未见过的物体位置（扩大范围）评估模型
3. **控制变量**：固定目标位置，只改变物体初始位置
4. **扩大随机化**：如果需要更多样本，可以生成新数据集，显式控制 `obj_init_pos` 的分布

## 如何拓展数据集

### 方法：使用 expert policy 自动生成数据

Meta-World 每个任务都有预训练的 expert policy，可以用它自动生成任意数量的示范数据。

```bash
# 运行生成脚本
python personal/test/metaworld/generate_metaworld_dataset.py
```

**脚本功能：**
- 使用 expert policy 自动生成示范数据
- 可以控制生成多少个 episode
- 支持随机化物体初始位置（通过不同 seed）
- 支持随机化目标位置（可选）
- 生成符合 LeRobot 格式的数据集

**关键配置：**
```python
NUM_EPISODES = 100  # 想生成多少个 episode

OBJ_RANDOMIZATION = {
    "use_random": True,
    "seed_start": 0,
    "seed_end": NUM_EPISODES,
}

GOAL_RANDOMIZATION = {
    "use_random": False,  # 是否随机化目标位置
    "goal_range": {
        "x": (0.05, 0.15),
        "y": (0.75, 0.85),
        "z": (0.18, 0.22),
    }
}
```

**生成后的使用：**
```python
# 加载生成的数据集
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("your-username/metaworld_pick_place_expanded")
print(f"总 episode 数: {dataset.num_episodes}")

# 或者用于训练
# lerobot-train --dataset.repo_id=your-username/metaworld_pick_place_expanded ...
```

**注意事项：**
- 生成速度约 1-2 秒/episode（取决于硬件）
- 生成的数据包含视频，需要较多磁盘空间（约 10-20MB/episode）
- 可以设置 `use_videos=False` 使用图片格式（占用更多空间但加载更快）