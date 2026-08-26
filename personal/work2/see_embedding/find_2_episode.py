"""
找到obj_init_pos差距最大的两个episode
"""

import json
import numpy as np
from itertools import combinations

def find_max_distance_episodes(json_path):
    # 加载数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    episodes = data["episodes"]
    
    # 提取所有episode的obj_init_pos
    positions = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        obj_pos = np.array(ep["obj_init_pos"])
        positions.append((ep_idx, obj_pos))
    
    print(f"总Episode数: {len(positions)}")
    print(f"正在计算所有episode对之间的距离（共 {len(positions)*(len(positions)-1)//2} 对）...\n")
    
    # 计算所有episode对之间的L2距离
    max_distance = 0
    max_pair = None
    max_pos_1 = None
    max_pos_2 = None
    
    all_distances = []
    
    for (ep1_idx, ep1_pos), (ep2_idx, ep2_pos) in combinations(positions, 2):
        distance = np.linalg.norm(ep1_pos - ep2_pos)
        all_distances.append((ep1_idx, ep2_idx, distance))
        
        if distance > max_distance:
            max_distance = distance
            max_pair = (ep1_idx, ep2_idx)
            max_pos_1 = ep1_pos
            max_pos_2 = ep2_pos
    
    # 打印结果
    print("="*80)
    print("差距最大的两个Episode")
    print("="*80)
    print(f"\nEpisode {max_pair[0]} vs Episode {max_pair[1]}")
    print(f"L2距离: {max_distance:.6f}")
    print(f"\nEpisode {max_pair[0]} 的 obj_init_pos:")
    print(f"  X: {max_pos_1[0]:.6f}")
    print(f"  Y: {max_pos_1[1]:.6f}")
    print(f"  Z: {max_pos_1[2]:.6f}")
    print(f"\nEpisode {max_pair[1]} 的 obj_init_pos:")
    print(f"  X: {max_pos_2[0]:.6f}")
    print(f"  Y: {max_pos_2[1]:.6f}")
    print(f"  Z: {max_pos_2[2]:.6f}")
    print(f"\n各维度差异:")
    print(f"  ΔX: {abs(max_pos_1[0] - max_pos_2[0]):.6f}")
    print(f"  ΔY: {abs(max_pos_1[1] - max_pos_2[1]):.6f}")
    print(f"  ΔZ: {abs(max_pos_1[2] - max_pos_2[2]):.6f}")
    
    # 统计信息
    all_distances.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"Top 10 差距最大的Episode对")
    print(f"{'='*80}")
    print(f"\n{'Rank':>4} | {'Ep1':>5} | {'Ep2':>5} | {'L2距离':>10} | {'ΔX':>8} | {'ΔY':>8} | {'ΔZ':>8}")
    print(f"{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for i, (ep1, ep2, dist) in enumerate(all_distances[:10]):
        pos1 = next(pos for idx, pos in positions if idx == ep1)
        pos2 = next(pos for idx, pos in positions if idx == ep2)
        print(f"{i+1:4d} | {ep1:5d} | {ep2:5d} | {dist:10.6f} | {abs(pos1[0]-pos2[0]):8.6f} | {abs(pos1[1]-pos2[1]):8.6f} | {abs(pos1[2]-pos2[2]):8.6f}")
    
    # 距离分布统计
    distances_only = [d for _, _, d in all_distances]
    
    print(f"\n{'='*80}")
    print(f"距离分布统计")
    print(f"{'='*80}")
    print(f"最小距离: {min(distances_only):.6f}")
    print(f"最大距离: {max(distances_only):.6f}")
    print(f"平均距离: {np.mean(distances_only):.6f}")
    print(f"中位数:   {np.median(distances_only):.6f}")
    print(f"标准差:   {np.std(distances_only):.6f}")
    
    # 分位数
    print(f"\n分位数:")
    print(f"  25%: {np.percentile(distances_only, 25):.6f}")
    print(f"  50%: {np.percentile(distances_only, 50):.6f}")
    print(f"  75%: {np.percentile(distances_only, 75):.6f}")
    print(f"  90%: {np.percentile(distances_only, 90):.6f}")
    print(f"  95%: {np.percentile(distances_only, 95):.6f}")
    print(f"  99%: {np.percentile(distances_only, 99):.6f}")

if __name__ == "__main__":
    json_path = "personal/work2/dataset_view/pickplacev3/episode_initial_states.json"
    find_max_distance_episodes(json_path)