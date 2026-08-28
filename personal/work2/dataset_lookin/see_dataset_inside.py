"""
root在代码开头指定
python personal/work2/dataset_lookin/see_dataset_inside.py
"""

import sys
from pathlib import Path

# 添加 lerobot 到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 默认数据集路径
DEFAULT_ROOT = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3"

def main():
    root = DEFAULT_ROOT
    
    print(f"加载数据集: {root}")
    print("="*60)
    
    # 加载数据集
    dataset = LeRobotDataset(repo_id="1/2", root=root)
    
    print(f"\n数据集信息:")
    print(f"  总帧数: {len(dataset)}")
    print(f"  Episodes 数: {len(dataset.meta.episodes)}")
    
    # 打印 meta 的所有 key
    print(f"\nMeta 的所有 key:")
    for key in dataset.meta.__dict__.keys():
        if not key.startswith('_'):
            print(f"  - {key}")
    
    # 打印 features 的所有 key
    print(f"\nFeatures 的所有 key:")
    for key in dataset.meta.features.keys():
        print(f"  - {key}")
    
    # 打印 stats 的所有 key
    print(f"\nStats 的所有 key:")
    for key in dataset.meta.stats.keys():
        print(f"  - {key}")
        for stat_key in dataset.meta.stats[key].keys():
            print(f"    - {stat_key}: shape={dataset.meta.stats[key][stat_key].shape if hasattr(dataset.meta.stats[key][stat_key], 'shape') else 'N/A'}")
    
    # 打印 episodes 信息
    print(f"\nEpisodes 信息:")
    print(f"  Episodes keys: {list(dataset.meta.episodes.keys()) if hasattr(dataset.meta.episodes, 'keys') else type(dataset.meta.episodes)}")
    if hasattr(dataset.meta.episodes, 'keys'):
        for key in dataset.meta.episodes.keys():
            print(f"  - {key}: {len(dataset.meta.episodes[key]) if hasattr(dataset.meta.episodes[key], '__len__') else dataset.meta.episodes[key]}")
    
    # 打印 cameras
    print(f"\nCameras:")
    if hasattr(dataset.meta, 'cameras'):
        print(f"  {dataset.meta.cameras}")
    
    # 尝试打印第一个样本（需要 FFmpeg/torchcodec）
    print(f"\n尝试获取第一个样本 (index=0)...")
    try:
        sample = dataset[0]
        print(f"第一个样本的所有 key:")
        for key in sample.keys():
            value = sample[key]
            if hasattr(value, 'shape'):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  - {key}: {type(value)}")
    except Exception as e:
        print(f"  无法获取样本数据（需要 FFmpeg/torchcodec）: {type(e).__name__}")
        print(f"  但 meta 信息已打印完成")
    
    print("\n" + "="*60)
    print("完成！")


if __name__ == "__main__":
    main()