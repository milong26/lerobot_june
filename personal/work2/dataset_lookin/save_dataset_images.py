"""
从personal/work2/dataset_view/pick_place_corner、corner2、corner3这三个数据集里面
得到每个数据集第一帧的图像
并分别用cornerx和key命名，保存在personal/work2/dataset_lookin/first_frame下
如果first_frame文件夹不存在，需要创建
用lerobot的形式打开数据集，需要用占位的repo_id="1/2"，root=分别的路径
python personal/work2/dataset_lookin/save_dataset_images.py
"""
import sys
from pathlib import Path

import numpy as np
import cv2

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from lerobot.datasets import LeRobotDataset

# 数据集配置
DATASETS = {
    "corner": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner",
    "corner2": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner2",
    "corner3": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3",
}

OUTPUT_DIR = Path(__file__).parent / "first_frame"


def save_first_frame_images():
    """从每个数据集中提取第一帧图像并保存"""
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for corner_name, dataset_root in DATASETS.items():
        
        # 加载数据集
        dataset = LeRobotDataset(
            repo_id="1/2",  # 占位repo_id
            root=dataset_root
        )
        
        print(f"    - 总帧数: {len(dataset)}")
        print(f"    - Features: {list(dataset.meta.features.keys())}")
        
        # 获取第一帧 (index=0)
        first_frame = dataset[0]
        
        # 创建该corner的输出子目录
        corner_output_dir = OUTPUT_DIR / corner_name
        corner_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历所有图像特征并保存
        image_keys = [key for key in first_frame.keys() if "observation.images" in key]
        
        print(f"  图像keys: {image_keys}")
        
        for key in image_keys:
            # 获取图像数据 (可能是 torch.Tensor 或 numpy array)
            img = first_frame[key]
            
            # 转换为 numpy array
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            
            # 处理形状: (C, H, W) -> (H, W, C)
            if img.shape[0] in [1, 3, 4]:  # 通道在前
                img = np.transpose(img, (1, 2, 0))
            
            # 归一化到 0-255
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.max() <= 255.0:
                img = img.astype(np.uint8)
            
            # 确保是3通道
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            elif img.shape[-1] == 4:  # RGBA -> RGB
                img = img[:, :, :3]
            
            # 保存图像 (OpenCV 使用 BGR)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output_path = corner_output_dir / f"{key}.png"
            cv2.imwrite(str(output_path), img_bgr)
            
            print(f" 保存: {output_path}")
            print(f"    Shape: {img.shape}, dtype: {img.dtype}")
    
    print(f"\n所有图像已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    save_first_frame_images()