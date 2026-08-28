"""
传感器噪声扰动模块

参考LIBERO-Plus论文的sensor noise轴，对渲染好的观测图像施加后处理噪声。
不修改仿真器状态，纯图像后处理，成本最低。

可用扰动类型：
- 高斯模糊 (gaussian_blur)
- 运动模糊 (motion_blur) - 可选实现
"""

import cv2
import numpy as np
from typing import Dict, Any


def apply_noise(image: np.ndarray, level: str, params: Dict[str, Any] = None) -> np.ndarray:
    """
    对图像施加噪声扰动

    参数:
        image: 输入RGB图像，shape (H, W, 3)，值域 [0, 255] 或 [0.0, 1.0]
        level: 扰动等级，"L0" 表示无扰动
        params: 扰动参数字典，来自probe_config.yaml

    返回:
        扰动后的图像，shape和值域与输入相同
    """
    if level == "L0" or params is None:
        return image.copy()

    # 确保输入是uint8格式
    original_dtype = image.dtype
    if image.dtype == np.float32 or image.dtype == np.float64:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    perturb_type = params.get("type", "gaussian_blur")

    if perturb_type == "gaussian_blur":
        kernel_size = params.get("kernel_size", 5)
        sigma_x = params.get("sigma_x", 1.5)
        sigma_y = params.get("sigma_y", 1.5)
        # kernel_size必须是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        perturbed = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y)

    elif perturb_type == "motion_blur":
        kernel_size = params.get("kernel_size", 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        # 创建运动模糊kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)
        kernel = kernel / kernel_size
        perturbed = cv2.filter2D(img_uint8, -1, kernel)

    else:
        raise ValueError(f"未知的扰动类型: {perturb_type}")

    # 恢复原始dtype
    if original_dtype == np.float32 or original_dtype == np.float64:
        perturbed = perturbed.astype(np.float32) / 255.0

    return perturbed


def get_noise_config(level: str) -> Dict[str, Any]:
    """
    获取预定义的噪声配置

    参数:
        level: "L0", "L1", 或 "L2"

    返回:
        扰动参数字典
    """
    configs = {
        "L0": None,
        "L1": {
            "type": "gaussian_blur",
            "kernel_size": 5,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
        },
        "L2": {
            "type": "gaussian_blur",
            "kernel_size": 9,
            "sigma_x": 3.0,
            "sigma_y": 3.0,
        },
    }
    return configs.get(level, None)


if __name__ == "__main__":
    # 简单测试
    import matplotlib.pyplot as plt

    # 创建测试图像
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # 测试各等级扰动
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(test_img)
    axes[0].set_title("Original (L0)")
    axes[0].axis("off")

    l1_img = apply_noise(test_img, "L1", get_noise_config("L1"))
    axes[1].imshow(l1_img)
    axes[1].set_title("L1 (轻度模糊)")
    axes[1].axis("off")

    l2_img = apply_noise(test_img, "L2", get_noise_config("L2"))
    axes[2].imshow(l2_img)
    axes[2].set_title("L2 (重度模糊)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("noise_perturb_test.png", dpi=150)
    print("测试图像已保存到 noise_perturb_test.png")