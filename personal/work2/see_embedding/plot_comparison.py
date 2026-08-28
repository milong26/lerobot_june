"""
生成Top相机Vision Encoder对比图
核心结论：Corner2的Top相机无法区分不同的episode
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 输出目录
output_dir = Path("results_comparison")
output_dir.mkdir(exist_ok=True)

# ============================================================
# 数据提取
# ============================================================

# Corner1 - Top相机 Vision Encoder L2（恒定值）
corner1_vis_top_mean = 632.0

# Corner2 - Top相机 Vision Encoder L2（完全相同，为0）
corner2_vis_top_mean = 0.0

# Corner3 - Top相机 Vision Encoder L2（有变化）
corner3_vis_top_mean = 314.86  # 从数据计算的均值

# ============================================================
# 绘制柱状图：三个Corner Top相机 Vision Encoder对比
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

corners = ['Corner1', 'Corner2\n⚠️ Problem', 'Corner3']
vision_means = [corner1_vis_top_mean, corner2_vis_top_mean, corner3_vis_top_mean]

x = np.arange(3)
width = 0.5

# 使用颜色突出Corner2的问题
colors = ['#4C72B0', '#FF0000', '#55A868']

bars = ax.bar(x, vision_means, width, color=colors, alpha=0.85, edgecolor='black', linewidth=2)

ax.set_ylabel('Mean Vision Encoder L2 Distance (Top Camera)', fontsize=13)
ax.set_title('Top Camera Vision Encoder: Episode Distinguishability', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(corners, fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (bar, value) in enumerate(zip(bars, vision_means)):
    if value > 0:
        ax.text(bar.get_x() + bar.get_width()/2., value + 15,
                f'{value:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    else:
        # Corner2的特殊标注
        ax.text(bar.get_x() + bar.get_width()/2., 25,
                '0.0', ha='center', va='bottom', fontsize=16, fontweight='bold', color='red')

# 添加箭头和注释指向Corner2
# ax.annotate('Cannot Distinguish Episodes!', 
#             xy=(1, 0), xytext=(1.8, 350),
#             fontsize=14, fontweight='bold', color='red',
#             arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
#             ha='center',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / '04_top_camera_statistics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 保存: 04_top_camera_statistics.png")

# ============================================================
# 打印统计摘要
# ============================================================
print("\n" + "="*60)
print("Top相机 Vision Encoder L2距离对比")
print("="*60)
print(f"Corner1: {corner1_vis_top_mean:>10.2f}")
print(f"Corner2: {corner2_vis_top_mean:>10.2f} ⚠️ 无法区分episode")
print(f"Corner3: {corner3_vis_top_mean:>10.2f}")
print("="*60)
print("\n🔴 关键发现: Corner2的Top相机L2距离为0，完全无法区分不同episode！")
print("="*60)
print(f"\n图表已保存到: {output_dir.absolute() / '04_top_camera_statistics.png'}")