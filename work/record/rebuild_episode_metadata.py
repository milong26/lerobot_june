#!/usr/bin/env python3
"""从实际 data 文件重建 episode metadata"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json

def rebuild_metadata(repo_id, root=None):
    from lerobot.utils.constants import HF_LEROBOT_HOME
    
    root_path = Path(root) if root else HF_LEROBOT_HOME / repo_id
    data_dir = root_path / 'data'
    episodes_dir = root_path / 'meta' / 'episodes'
    info_path = root_path / 'meta' / 'info.json'
    
    # 备份
    backup_dir = episodes_dir.parent / 'episodes_backup'
    if not backup_dir.exists():
        print(f"备份原始 episodes 目录到 {backup_dir}")
        shutil.copytree(episodes_dir, backup_dir)
    else:
        print(f"从备份恢复...")
        shutil.rmtree(episodes_dir)
        shutil.copytree(backup_dir, episodes_dir)
    
    # 清空 episodes 目录
    shutil.rmtree(episodes_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描所有 data parquet 文件
    data_files = sorted(data_dir.glob('**/*.parquet'))
    print(f"找到 {len(data_files)} 个 data parquet 文件")
    
    # 加载所有 data，按 episode_index 分组
    all_data = []
    for f in data_files:
        table = pq.read_table(str(f))
        df = table.to_pandas()
        all_data.append(df)
        print(f"  {f}: {len(df)} frames")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n总 frame 数: {len(combined_df)}")
    
    # 获取视频 keys（从 info.json 或 videos 目录）
    video_keys = []
    info_path_tmp = root_path / 'meta' / 'info.json'
    if info_path_tmp.exists():
        with open(info_path_tmp, 'r') as f:
            info_data = json.load(f)
        if 'features' in info_data:
            for key, val in info_data['features'].items():
                if val.get('dtype') in ['image', 'video']:
                    video_keys.append(key)
    
    # 如果 info.json 中没有，尝试从 videos 目录获取
    if not video_keys:
        videos_dir = root_path / 'videos'
        if videos_dir.exists():
            video_keys = [d.name for d in videos_dir.iterdir() if d.is_dir()]
    
    print(f"视频 keys: {video_keys}")
    
    # 按 episode_index 分组
    episodes = combined_df.groupby('episode_index')
    episode_list = []
    
    for ep_idx, ep_df in episodes:
        ep_length = len(ep_df)
        tasks = ep_df['task_index'].unique()
        task_list = [f"task_{t}" for t in tasks]  # 简化处理
        
        # 找到对应的 data chunk/file
        data_chunk = None
        data_file = None
        for f in data_files:
            f_df = pd.read_parquet(f)
            if ep_idx in f_df['episode_index'].values:
                # 从文件名提取 chunk 和 file index
                parts = f.parts
                data_chunk = int(parts[-2].replace("chunk-", "").replace(".parquet", ""))
                data_file = int(parts[-1].replace("file-", "").replace(".parquet", ""))
                break
        
        episode_dict = {
            'episode_index': int(ep_idx),
            'tasks': task_list,
            'length': ep_length,
            'data/chunk_index': data_chunk,
            'data/file_index': data_file,
            'dataset_from_index': int(ep_df['index'].min()),
            'dataset_to_index': int(ep_df['index'].max()) + 1,
        }
        
        # 添加视频信息
        for vid_key in video_keys:
            # 视频文件通常和 data 文件在相同的 chunk/file
            episode_dict[f'videos/{vid_key}/chunk_index'] = data_chunk
            episode_dict[f'videos/{vid_key}/file_index'] = data_file
            episode_dict[f'videos/{vid_key}/from_timestamp'] = float(ep_df['timestamp'].min())
            episode_dict[f'videos/{vid_key}/to_timestamp'] = float(ep_df['timestamp'].max())
        
        episode_list.append(episode_dict)
    
    print(f"\n重建了 {len(episode_list)} 个 episode 的 metadata")
    
    # 尝试加载旧的 stats（如果有）
    stats_path = root_path / 'meta' / 'stats.json'
    old_stats = {}
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            old_stats = json.load(f)
        print(f"加载了旧的 stats")
    
    # 添加 stats 到每个 episode
    for ep_dict in episode_list:
        ep_idx = ep_dict['episode_index']
        
        # 从旧 stats 中提取这个 episode 的 stats（如果存在）
        # 这里简化处理，使用全局 stats
        if old_stats:
            for feature_key, feature_stats in old_stats.items():
                for stat_key, stat_value in feature_stats.items():
                    full_key = f'stats/{feature_key}/{stat_key}'
                    if isinstance(stat_value, list):
                        ep_dict[full_key] = stat_value
                    else:
                        ep_dict[full_key] = [stat_value]
        
        # 添加 meta/episodes 信息（所有 episode 都指向同一个文件）
        ep_dict['meta/episodes/chunk_index'] = 0
        ep_dict['meta/episodes/file_index'] = 0
    
    # 写入新的 episode parquet 文件
    chunk_dir = episodes_dir / 'chunk-000'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    file_path = chunk_dir / 'file-000.parquet'
    
    df = pd.DataFrame(episode_list)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(file_path))
    
    print(f"写入 {len(episode_list)} 个 episode 到 {file_path}")
    
    # 更新 info.json
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        info['total_episodes'] = len(episode_list)
        info['total_frames'] = len(combined_df)
        info['splits'] = {"train": f"0:{len(episode_list)}"}
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"更新 info.json: total_episodes={len(episode_list)}, total_frames={len(combined_df)}")
    
    print("\n重建完成!")
    
    # 验证
    print("\n验证重建后的 metadata:")
    rebuilt_files = sorted(episodes_dir.glob('**/*.parquet'))
    for f in rebuilt_files:
        table = pq.read_table(str(f))
        print(f"  {f}: {table.num_rows} episodes")
        
        df = table.to_pandas()
        print(f"    Episode 索引: {sorted(df['episode_index'].tolist())}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()
    
    rebuild_metadata(args.repo_id, args.root)