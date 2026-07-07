from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    "test/first",
    root="/home/qwe/.cache/huggingface/lerobot/force/first",
    episodes=[0]
)

print(dataset.features.keys())

# 输出结果是dict_keys(['action', 'observation.state', 'observation.images.wrist', 'observation.images.top', 'observation.force', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'])

import numpy as np

dataset = LeRobotDataset(
    "test/first",
    root="/home/qwe/.cache/huggingface/lerobot/force/first",
    episodes=[0]
)

force_data = []

for i in range(len(dataset)):
    sample = dataset[i]

    force = sample["observation.force"]

    force_data.append(force.numpy())

force_data = np.array(force_data)

print(force_data.shape)

np.savetxt("force.txt", force_data)

print("saved to force.txt")