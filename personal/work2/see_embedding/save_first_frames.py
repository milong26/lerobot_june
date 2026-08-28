"""
加载三个corner数据集的第0集第0帧的top相机视角并保存为图片
数据集路径从结果与分析.md中读取：
- corner1: personal/work2/dataset_view/pick_place_camcorner
- corner2: personal/work2/dataset_view/pickplacev3
- corner3: personal/work2/dataset_view/pick_place_corner3
"""
import numpy as np
import os
import sys
import torch
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from lerobot.datasets import LeRobotDataset

# 数据集配置
DATASETS = {
    'corner1': 'personal/work2/dataset_view/pick_place_camcorner',
    'corner2': 'personal/work2/dataset_view/pickplacev3',
    'corner3': 'personal/work2/dataset_view/pick_place_corner3',
}

OUTPUT_DIR = 'results_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_save_first_frame(corner_name, dataset_path):
    """加载指定数据集的第0集第0帧的top相机视角并保存"""
    print(f"\n{'='*60}")
    print(f"处理 {corner_name}: {dataset_path}")
    print(f"{'='*60}")
    
    # 加载数据集
    print(f"正在加载数据集...")
    dataset = LeRobotDataset(repo_id="1/2",root=dataset_path)
    print(f"数据集加载完成!")
    print(f"  总Episode数: {dataset.num_episodes}")
    print(f"  总帧数: {len(dataset)}")
    
    # 获取第0集的帧范围
    ep_start = dataset.meta.episodes["dataset_from_index"][0]
    ep_end = dataset.meta.episodes["dataset_to_index"][0]
    print(f"  Episode 0 帧范围: [{ep_start}, {ep_end})")
    
    # 获取第0帧（即ep_start）
    frame_idx = ep_start
    sample = dataset[frame_idx]
    
    # 获取top相机图像
    if 'observation.images.top' in sample:
        top_img_tensor = sample['observation.images.top']
        print(f"\n  Top相机图像张量形状: {top_img_tensor.shape}")
        
        # 转换为PIL Image
        if hasattr(top_img_tensor, 'numpy'):
            np_img = top_img_tensor.numpy()
        else:
            np_img = top_img_tensor
        
        # 如果是CHW格式，转换为HWC
        if np_img.ndim == 3 and np_img.shape[0] < np_img.shape[-1]:
            np_img = np.transpose(np_img, (1, 2, 0))
        
        # 归一化到0-255
        if np_img.max() <= 1.0:
            pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray(np_img.astype(np.uint8))
        
        # 保存
        output_path = os.path.join(OUTPUT_DIR, f'{corner_name}_first.png')
        pil_img.save(output_path)
        print(f"  ✓ 已保存到: {output_path}")
        print(f"  图像尺寸: {pil_img.size}")
    else:
        print(f"  ⚠️ 未找到 'observation.images.top'，可用的keys: {list(sample.keys())}")
    
    print(f"\n{corner_name} 处理完成!")


def main():
    print("开始加载三个corner数据集的第0集第0帧...")
    
    for corner_name, dataset_path in DATASETS.items():
        try:
            load_and_save_first_frame(corner_name, dataset_path)
        except Exception as e:
            print(f"\n❌ {corner_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("所有处理完成!")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()