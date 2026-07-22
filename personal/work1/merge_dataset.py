from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

root = Path("/data/zhonglinye/jun/ep10/ep10")

names = [
    "bushing",
    "charger",
    "cylinder_lay",
    "cylinder_standing",
    "plug",
    "rectangular",
    "ring",
    "ushape",
    "valve",
]

datasets = [
    LeRobotDataset(
        repo_id=name,
        root=root / name,
    )
    for name in names
]

merge_datasets(
    datasets=datasets,
    output_repo_id="ep10/all",
    output_dir="/data/zhonglinye/jun/lerobot/personal/work1/ep10_all",
)

print("Done.")