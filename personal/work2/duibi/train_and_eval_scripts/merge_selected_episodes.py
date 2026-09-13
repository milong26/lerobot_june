#!/usr/bin/env python3
"""
Merge selected episodes from multiple LeRobot datasets into a single dataset.
使用官方的 split_dataset 和 merge_datasets 函数，确保数据结构正确。
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import (
    split_dataset,
    merge_datasets,
    recompute_stats,
)


def _process_episode_initial_states(
    dataset_root,
    dataset_names,
    selected_episode_lists,
    output_dir,
):
    """处理 episode_initial_states.json（自定义文件）"""
    print(f"\n{'=' * 60}")
    print(f"处理 episode_initial_states.json")
    print(f"{'=' * 60}")
    
    # 需要合并各子数据集的 episode_initial_states.json
    # 按照合并后的 episode offset 重新组织
    merged_episodes = []
    
    for dataset_name, selected_episodes in zip(
        dataset_names,
        selected_episode_lists,
        strict=True,
    ):
        src_root = Path(dataset_root) / dataset_name
        states_file = src_root / "episode_initial_states.json"
        
        if states_file.exists():
            with open(states_file, "r") as f:
                states = json.load(f)
            
            # 检查 JSON 格式
            if "episodes" in states:
                episodes_list = states["episodes"]
            elif "initial_states" in states:
                episodes_list = states["initial_states"]
            elif isinstance(states, list):
                episodes_list = states
            else:
                print(f"  ⚠️ {dataset_name}: episode_initial_states.json 格式未知，跳过")
                continue
            
            # 只复制选中 episode 的 initial states
            for old_ep_idx in selected_episodes:
                if old_ep_idx < len(episodes_list):
                    merged_episodes.append(episodes_list[old_ep_idx])
    
    if merged_episodes:
        # 保存为 episodes 格式
        merged_states = {
            "task": "multi_task",
            "num_episodes": len(merged_episodes),
            "episodes": merged_episodes,
        }
        with open(output_dir / "episode_initial_states.json", "w") as f:
            json.dump(merged_states, f, indent=2)
        print(f"  ✓ 已保存 episode_initial_states.json ({len(merged_episodes)} episodes)")
    else:
        print(f"  ⚠️ 没有找到 episode_initial_states.json 或格式不匹配")
    
    # 生成 subset_file.json（兼容 train_minivla.py）
    print(f"\n{'=' * 60}")
    print(f"生成 subset_file.json")
    print(f"{'=' * 60}")
    
    # 加载数据集获取总 episode 数
    merged_dataset = LeRobotDataset(
        repo_id=output_dir.name,
        root=output_dir,
    )
    
    subset_file_data = {
        "selected_episode_indices": list(range(merged_dataset.meta.total_episodes)),
        "total_episodes": merged_dataset.meta.total_episodes,
        "source_datasets": dataset_names,
    }
    
    with open(output_dir / "subset_file.json", "w") as f:
        json.dump(subset_file_data, f, indent=2)
    print(f"  ✓ 已保存 subset_file.json")
    
    print(f"\n{'=' * 60}")
    print(f"处理完成！")
    print(f"{'=' * 60}")


def select_episodes(
    dataset_root,
    dataset_names,
    selected_episode_lists,
    output_dir,
):
    """
    使用官方 split_dataset 和 merge_datasets 合并选中的 episode。
    
    Args:
        dataset_root: 源数据集根目录
        dataset_names: 数据集名称列表
        selected_episode_lists: 每个数据集选中的 episode 索引列表
        output_dir: 输出目录
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    # 检查数据集是否已经存在
    if output_dir.exists():
        print(f"数据集已存在: {output_dir}")
        print(f"跳过合并，直接处理 episode_initial_states.json")
        
        # 直接跳到第五步：处理 episode_initial_states.json
        return _process_episode_initial_states(
            dataset_root, dataset_names, selected_episode_lists, output_dir
        )
    
    # 临时目录存放子数据集
    temp_root = output_dir.parent / f"{output_dir.name}_subsets"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    
    subset_datasets = []
    
    # 第一步：对每个数据集使用 split_dataset 创建子数据集
    for dataset_name, selected_episodes in zip(
        dataset_names,
        selected_episode_lists,
        strict=True,
    ):
        src_root = dataset_root / dataset_name
        
        print(f"\n{'=' * 60}")
        print(f"处理数据集: {dataset_name}")
        print(f"选中 episode 数: {len(selected_episodes)}")
        print(f"{'=' * 60}")
        
        # 加载源数据集
        src_dataset = LeRobotDataset(
            repo_id=f"local/{dataset_name}",
            root=src_root,
        )
        
        print(f"  源数据集总帧数: {src_dataset.meta.total_frames}")
        print(f"  源数据集总 episode 数: {src_dataset.meta.total_episodes}")
        
        # 使用 split_dataset 创建子数据集
        split_output = temp_root / dataset_name
        
        result = split_dataset(
            dataset=src_dataset,
            splits={
                "selected": selected_episodes,
            },
            output_dir=split_output,
        )
        
        subset_dataset = result["selected"]
        subset_datasets.append(subset_dataset)
        
        print(f"  子数据集总帧数: {subset_dataset.meta.total_frames}")
        print(f"  子数据集总 episode 数: {subset_dataset.meta.total_episodes}")
        
        # 验证子数据集
        indices = subset_dataset.hf_dataset["index"]
        assert indices == list(range(len(indices))), f"{dataset_name}: index 不连续"
        print(f"  ✓ index 连续且正确")
    
    # 第二步：使用 merge_datasets 合并子数据集
    print(f"\n{'=' * 60}")
    print(f"合并 {len(subset_datasets)} 个子数据集")
    print(f"{'=' * 60}")
    
    merged_dataset = merge_datasets(
        datasets=subset_datasets,
        output_repo_id=output_dir.name,
        output_dir=output_dir,
        concatenate_videos=False,  # 先保证正确，再考虑优化
        concatenate_data=False,
    )
    
    print(f"  合并后总帧数: {merged_dataset.meta.total_frames}")
    print(f"  合并后总 episode 数: {merged_dataset.meta.total_episodes}")
    
    # 第三步：重新计算 stats
    print(f"\n{'=' * 60}")
    print(f"重新计算 stats")
    print(f"{'=' * 60}")
    
    recompute_stats(
        merged_dataset,
        skip_image_video=True,
    )
    
    print(f"  ✓ stats 已重新计算")
    
    # 第四步：验证合并后的数据集
    print(f"\n{'=' * 60}")
    print(f"验证合并后的数据集")
    print(f"{'=' * 60}")
    
    # 验证 index 连续性
    assert len(merged_dataset) == merged_dataset.meta.total_frames, "总帧数不匹配"
    print(f"  ✓ 总帧数匹配: {len(merged_dataset)}")
    
    indices = merged_dataset.hf_dataset["index"]
    assert indices == list(range(len(indices))), "index 不连续"
    print(f"  ✓ index 连续: 0 到 {len(indices) - 1}")
    
    # 验证 episode_index 连续性
    episode_indices = merged_dataset.hf_dataset["episode_index"]
    # 转换为整数列表
    unique_ep_indices = sorted(set(int(x) if hasattr(x, 'item') else x for x in episode_indices))
    expected_ep_indices = list(range(merged_dataset.meta.total_episodes))
    
    if unique_ep_indices != expected_ep_indices:
        print(f"  ⚠️ episode_index 不连续！")
        print(f"  期望: {expected_ep_indices[:10]}... (共 {len(expected_ep_indices)} 个)")
        print(f"  实际: {unique_ep_indices[:10]}... (共 {len(unique_ep_indices)} 个)")
        print(f"  差异: {set(unique_ep_indices) - set(expected_ep_indices)}")
    
    assert unique_ep_indices == expected_ep_indices, "episode_index 不连续"
    print(f"  ✓ episode_index 连续: 0 到 {merged_dataset.meta.total_episodes - 1}")
    
    # 验证每个 episode 的 dataset_from_index 和 dataset_to_index
    for i in range(merged_dataset.meta.total_episodes):
        ep = merged_dataset.meta.episodes[i]
        ep_length = ep["length"]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]
        
        assert to_idx - from_idx == ep_length, f"Episode {i}: to_idx - from_idx != length"
    
    print(f"  ✓ 所有 episode 的 dataset_from_index/to_index 正确")
    
    # 验证 action 无 NaN/Inf
    import numpy as np
    for i in range(len(merged_dataset)):
        item = merged_dataset[i]
        if 'action' in item:
            action = item['action'].numpy()
            assert not np.isnan(action).any(), f"Frame {i}: action 包含 NaN"
            assert not np.isinf(action).any(), f"Frame {i}: action 包含 Inf"
    
    print(f"  ✓ action 无 NaN/Inf")
    
    # 验证 stats
    for feature_name, feature_stats in merged_dataset.meta.stats.items():
        if 'mean' in feature_stats:
            mean = np.array(feature_stats['mean'])
            std = np.array(feature_stats['std'])
            assert np.isfinite(mean).all(), f"{feature_name}: mean 包含非有限值"
            assert np.isfinite(std).all(), f"{feature_name}: std 包含非有限值"
    
    print(f"  ✓ stats 所有 mean/std 有限")
    
    # 第五步：处理 episode_initial_states.json 和生成 subset_file.json
    _process_episode_initial_states(
        dataset_root, dataset_names, selected_episode_lists, output_dir
    )
    
    # 清理临时目录
    print(f"\n{'=' * 60}")
    print(f"清理临时目录")
    print(f"{'=' * 60}")
    
    if temp_root.exists():
        shutil.rmtree(temp_root)
        print(f"  ✓ 已清理 {temp_root}")
    
    print(f"\n{'=' * 60}")
    print(f"合并完成！")
    print(f"{'=' * 60}")
    print(f"输出目录: {output_dir}")
    print(f"总 episode 数: {merged_dataset.meta.total_episodes}")
    print(f"总帧数: {merged_dataset.meta.total_frames}")


def merge_from_subset_file(
    subset_file,
    output_dir,
    dataset_base_dir,
):
    """
    从 subset file 合并数据集（兼容旧的接口）。
    
    Args:
        subset_file: merged subset JSON file 路径
        output_dir: 输出目录
        dataset_base_dir: 源数据集根目录
    """
    # 加载 subset file
    with open(subset_file, "r") as f:
        subset_data = json.load(f)
    
    datasets = subset_data["datasets"]
    episodes_per_dataset = subset_data["episodes_per_dataset"]
    all_indices = subset_data["selected_episode_indices"]
    
    # 将 all_indices 按数据集拆分
    selected_episode_lists = []
    for i, ds in enumerate(datasets):
        start_idx = i * episodes_per_dataset
        end_idx = (i + 1) * episodes_per_dataset
        ep_list = all_indices[start_idx:end_idx]
        selected_episode_lists.append(ep_list)
        print(f"数据集 {ds}: 选中 {len(ep_list)} episodes")
    
    # 调用 select_episodes
    select_episodes(
        dataset_root=dataset_base_dir,
        dataset_names=datasets,
        selected_episode_lists=selected_episode_lists,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge selected episodes from multiple datasets")
    
    # 新接口参数
    parser.add_argument("--dataset-root", type=str,
                        help="源数据集根目录")
    parser.add_argument("--dataset-names", type=str, nargs="+",
                        help="数据集名称列表")
    parser.add_argument("--episodes-per-dataset", type=int,
                        help="每个数据集选中的 episode 数")
    parser.add_argument("--selection-mode", type=str, default="random",
                        choices=["random", "first", "ours_v5"],
                        help="选择模式：random, first, 或 ours_v5")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（用于 random 模式）")
    parser.add_argument("--v5-output-dir", type=str,
                        help="ours_v5 模式的输出目录（包含 subset JSON 文件）")
    parser.add_argument("--output-dir", type=str,
                        help="输出合并数据集目录")
    
    # 旧接口参数（兼容）
    parser.add_argument("--subset-file", type=str,
                        help="Path to merged subset JSON file（旧接口）")
    parser.add_argument("--dataset-base-dir", type=str,
                        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view",
                        help="Base directory containing source datasets（旧接口）")
    
    args = parser.parse_args()
    
    # 判断使用新接口还是旧接口
    if args.subset_file:
        # 旧接口：从 subset file 读取
        print("使用旧接口：从 subset file 读取")
        merge_from_subset_file(
            subset_file=args.subset_file,
            output_dir=args.output_dir,
            dataset_base_dir=args.dataset_base_dir,
        )
    else:
        # 新接口：直接使用参数
        print("使用新接口：直接使用参数")
        
        # 生成选中的 episode 索引列表
        selected_episode_lists = []
        for dataset_name in args.dataset_names:
            # 加载源数据集获取总 episode 数
            src_root = Path(args.dataset_root) / dataset_name
            src_dataset = LeRobotDataset(
                repo_id=f"local/{dataset_name}",
                root=src_root,
            )
            total_episodes = src_dataset.meta.total_episodes
            
            if args.selection_mode == "random":
                random.seed(args.seed)
                # 确保不超过总 episode 数
                n_select = min(args.episodes_per_dataset, total_episodes)
                selected_episodes = sorted(random.sample(range(total_episodes), n_select))
                print(f"数据集 {dataset_name}: 随机选中 {n_select}/{total_episodes} 个 episodes (seed={args.seed})")
            elif args.selection_mode == "first":
                n_select = min(args.episodes_per_dataset, total_episodes)
                selected_episodes = list(range(n_select))
                print(f"数据集 {dataset_name}: 选中前 {n_select}/{total_episodes} 个 episodes")
            elif args.selection_mode == "ours_v5":
                # 从 ours_v5 输出目录读取 subset JSON 文件
                # 文件路径模式: {v5_output_dir}/our_v5_{episodes}_seed{seed}_{dataset_name}/subsets/our_v5_{episodes}_seed{seed}.json
                v5_dataset_dir = Path(args.v5_output_dir) / f"our_v5_{args.episodes_per_dataset}_seed{args.seed}_{dataset_name}"
                v5_subset_file = v5_dataset_dir / "subsets" / f"our_v5_{args.episodes_per_dataset}_seed{args.seed}.json"
                if not v5_subset_file.exists():
                    raise FileNotFoundError(
                        f"V5 subset file not found: {v5_subset_file}\n"
                        f"请先为数据集 {dataset_name} 运行V5 episode选择脚本生成subset文件"
                    )
                with open(v5_subset_file, "r") as f:
                    v5_data = json.load(f)
                selected_episodes = sorted(v5_data["selected_episode_indices"])
                print(f"数据集 {dataset_name}: V5 选中 {len(selected_episodes)} 个 episodes (seed={args.seed})")
            else:
                raise ValueError(f"Unknown selection mode: {args.selection_mode}")
            
            selected_episode_lists.append(selected_episodes)
        
        # 如果 output_dir 没有指定，自动生成包含 episode 数的目录名
        if args.output_dir is None:
            episodes_str = "x".join([str(len(ep_list)) for ep_list in selected_episode_lists])
            datasets_str = "+".join(args.dataset_names)
            mode_suffix = f"_{args.selection_mode}{args.seed}" if args.selection_mode == "random" else "_first"
            auto_dir = Path(args.dataset_root) / f"merged_{datasets_str}_{episodes_str}{mode_suffix}"
            args.output_dir = str(auto_dir)
            print(f"\n自动生成输出目录: {args.output_dir}")
        
        select_episodes(
            dataset_root=args.dataset_root,
            dataset_names=args.dataset_names,
            selected_episode_lists=selected_episode_lists,
            output_dir=args.output_dir,
        )