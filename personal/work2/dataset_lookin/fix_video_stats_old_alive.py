"""
从旧数据集的 stats.json 恢复 camera stats 并迁移 episode_initial_states.json
旧数据集: personal/work2/dataset_view/pick_place_camcorner (图像格式)
新数据集: /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner (视频格式)

python personal/work2/dataset_lookin/fix_video_stats_old_alive.py
"""

import json
from pathlib import Path
import shutil

# 旧数据集路径(图像格式)
OLD_DATASET_ROOT = Path("personal/work2/dataset_view/pickplacev3")
OLD_STATS_FILE = OLD_DATASET_ROOT / "meta" / "stats.json"

# 新数据集路径(视频格式)
NEW_DATASET_ROOT = Path("/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner2")
NEW_STATS_FILE = NEW_DATASET_ROOT / "meta" / "stats.json"

# Camera keys
CAMERA_KEYS = ["observation.images.top", "observation.images.wrist"]

def restore_camera_stats():
    """从旧 stats.json 恢复 camera stats"""
    print(f"\n从旧数据集恢复 camera stats...")
    
    if not OLD_STATS_FILE.exists():
        print(f"  错误: 旧 stats.json 不存在: {OLD_STATS_FILE}")
        return False
    
    # 加载旧 stats
    with open(OLD_STATS_FILE, 'r') as f:
        old_stats = json.load(f)
    
    # 加载新 stats
    if not NEW_STATS_FILE.exists():
        print(f"  错误: 新 stats.json 不存在: {NEW_STATS_FILE}")
        return False
    
    with open(NEW_STATS_FILE, 'r') as f:
        new_stats = json.load(f)
    
    # 检查缺失的 camera keys
    missing_keys = [key for key in CAMERA_KEYS if key not in new_stats]
    
    if not missing_keys:
        print("  所有 camera keys 已存在,无需恢复")
        return True
    
    print(f"  缺失的 camera keys: {missing_keys}")
    
    # 创建备份
    backup_file = NEW_STATS_FILE.with_suffix('.json.backup')
    print(f"  创建备份: {backup_file}")
    shutil.copy(NEW_STATS_FILE, backup_file)
    
    # 从旧 stats 复制缺失的 camera stats
    restored_count = 0
    for key in missing_keys:
        if key in old_stats:
            new_stats[key] = old_stats[key]
            print(f"  ✓ 已恢复: {key}")
            restored_count += 1
        else:
            print(f"  ✗ 旧 stats 中也不存在: {key}")
    
    # 保存更新后的 stats
    if restored_count > 0:
        print(f"\n  保存更新后的 stats...")
        with open(NEW_STATS_FILE, 'w') as f:
            json.dump(new_stats, f, indent=2)
        print(f"  已恢复 {restored_count} 个 camera key")
    
    return restored_count > 0

def migrate_episode_initial_states():
    """迁移 episode_initial_states.json 文件"""
    print(f"\n迁移 episode_initial_states.json...")
    
    old_file = OLD_DATASET_ROOT / "episode_initial_states.json"
    new_file = NEW_DATASET_ROOT / "episode_initial_states.json"
    
    if not old_file.exists():
        print(f"  警告: 旧数据集中不存在 {old_file}")
        return False
    
    if new_file.exists():
        backup_file = new_file.with_suffix('.json.backup')
        print(f"  创建备份: {backup_file}")
        shutil.copy(new_file, backup_file)
    
    shutil.copy(old_file, new_file)
    print(f"  ✓ 已复制: {old_file} -> {new_file}")
    
    return True

def main():
    print(f"旧数据集(图像): {OLD_DATASET_ROOT}")
    print(f"新数据集(视频): {NEW_DATASET_ROOT}")
    print(f"旧 Stats 文件: {OLD_STATS_FILE}")
    print(f"新 Stats 文件: {NEW_STATS_FILE}")
    
    if not OLD_DATASET_ROOT.exists():
        print(f"\n错误: 旧数据集不存在: {OLD_DATASET_ROOT}")
        return
    
    if not NEW_DATASET_ROOT.exists():
        print(f"\n错误: 新数据集不存在: {NEW_DATASET_ROOT}")
        return
    
    # 恢复 camera stats
    stats_restored = restore_camera_stats()
    
    # 迁移 episode_initial_states.json
    file_migrated = migrate_episode_initial_states()
    
    # 验证
    print(f"\n验证 camera keys:")
    with open(NEW_STATS_FILE, 'r') as f:
        final_stats = json.load(f)
    
    for key in CAMERA_KEYS:
        if key in final_stats:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key} (仍然缺失)")
    
    print(f"\n总结:")
    if stats_restored:
        print(f"  ✓ Camera stats 恢复成功")
    else:
        print(f"  - Camera stats 无需恢复或恢复失败")
    
    if file_migrated:
        print(f"  ✓ episode_initial_states.json 迁移成功")
    else:
        print(f"  ✗ episode_initial_states.json 迁移失败")
    
    print(f"\n全部完成!")

if __name__ == "__main__":
    main()