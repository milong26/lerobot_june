#!/usr/bin/env python3
"""
测试 ep10 目录下所有数据集能否访问指定的 episode（默认第 29 个）
使用与 lerobot_train.py 相同的数据集访问方式
"""

import os
import sys
from pathlib import Path

# 添加 lerobot 到路径
sys.path.insert(0, "/home/qwe/jun/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置路径
EP10_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "ep10"
TEST_EPISODE_INDEX = 29  # 要测试的 episode 索引


def test_dataset_all_episodes(folder_name: str) -> dict:
    """测试指定数据集能否访问所有 episode"""
    repo_id = f"ep10/{folder_name}"
    root = EP10_DIR / folder_name
    
    result = {
        "folder": folder_name,
        "repo_id": repo_id,
        "success": False,
        "error": None,
        "num_episodes": None,
        "total_frames": None,
        "failed_episodes": [],
    }
    
    if not root.exists():
        result["error"] = f"目录不存在: {root}"
        return result
    
    try:
        # 先加载完整数据集（不指定 episodes）
        full_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(root),
            return_uint8=True,
        )
        
        total_episodes = full_dataset.num_episodes
        result["num_episodes"] = total_episodes
        result["total_frames"] = len(full_dataset)
        
        print(f"  共 {total_episodes} 个 episode, {len(full_dataset)} 帧", end="")
        
        # 遍历所有 episode
        failed = []
        for ep_idx in range(total_episodes):
            try:
                dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=str(root),
                    episodes=[ep_idx],
                    return_uint8=True,
                )
                # 尝试访问最后一帧
                if len(dataset) > 0:
                    _ = dataset[-1]
            except Exception as e:
                failed.append((ep_idx, f"{type(e).__name__}: {str(e)[:100]}"))
        
        if failed:
            result["failed_episodes"] = failed
            result["error"] = f"{len(failed)} 个 episode 访问失败"
            print(f" - ✗ {len(failed)} 个 episode 失败")
        else:
            result["success"] = True
            print(" - ✓ 全部成功")
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f" - ✗ 加载失败: {result['error'][:100]}")
    
    return result


def main():
    """主函数：遍历 ep10 目录下的所有文件夹，测试数据集访问"""
    print(f"扫描目录: {EP10_DIR}\n")
    
    if not EP10_DIR.exists():
        print(f"✗ 目录不存在: {EP10_DIR}")
        return
    
    results = []
    
    # 遍历 ep10 下的所有文件夹
    for folder in sorted(EP10_DIR.iterdir()):
        if not folder.is_dir():
            continue
        
        print(f"测试数据集: {folder.name}...", end=" ")
        
        result = test_dataset_all_episodes(folder.name)
        results.append(result)
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print(f"测试完成！共测试 {len(results)} 个数据集")
    
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print(f"成功: {success_count}, 失败: {fail_count}\n")
    
    if fail_count > 0:
        print("失败的数据集详情:")
        print("-" * 80)
        for r in results:
            if not r["success"]:
                print(f"  {r['folder']}: {r['error']}")
                if r.get("failed_episodes"):
                    for ep_idx, err in r["failed_episodes"]:
                        print(f"    Episode {ep_idx}: {err}")
        print()
    
    # 输出成功的数据集
    if success_count > 0:
        print("成功的数据集:")
        print("-" * 80)
        for r in results:
            if r["success"]:
                print(f"  {r['folder']}: {r['num_episodes']} episodes, {r['total_frames']} frames")


if __name__ == "__main__":
    main()