#!/usr/bin/env python3
"""
为多个数据集运行V5 episode选择，生成merge脚本需要的subset文件。

用法:
    python run_v5_for_all_datasets.py \
        --dataset-root personal/work2/dataset_view \
        --dataset-names disassemble-v3_corner pick_place-v3_corner coffee-button-v3_corner \
        --episodes-per-dataset 112 \
        --seed 42 \
        --output-dir personal/work2/duibi/our_v5_multi_336_seed42/subsets
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_CONFIGS = {
    "disassemble-v3_corner": {
        "dataset_root": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner",
        "env_task": "disassemble-v3",
    },
    "pick_place-v3_corner": {
        "dataset_root": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place-v3_corner",
        "env_task": "pick-place-v3",
    },
    "coffee-button-v3_corner": {
        "dataset_root": "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/coffee-button-v3_corner",
        "env_task": "coffee-button-v3",
    },
}


def run_v5_for_dataset(
    dataset_name: str,
    dataset_root: str,
    episodes_per_dataset: int,
    seed: int,
    v5_output_dir: str,
):
    """为单个数据集运行V5选择"""
    print(f"\n{'='*60}")
    print(f"为数据集 {dataset_name} 运行V5选择")
    print(f"{'='*60}")
    
    v5_dataset_dir = Path(v5_output_dir) / f"our_v5_{episodes_per_dataset}_seed{seed}_{dataset_name}"
    v5_subset_file = v5_dataset_dir / "subsets" / f"our_v5_{episodes_per_dataset}_seed{seed}.json"
    
    if v5_subset_file.exists():
        print(f"✓ V5 subset文件已存在: {v5_subset_file}")
        print(f"跳过，使用缓存结果")
        return str(v5_subset_file)
    
    select_script = PROJECT_ROOT / "personal" / "work2" / "our_v5" / "select_our_v5.py"
    if not select_script.exists():
        raise FileNotFoundError(f"V5选择脚本不存在: {select_script}")
    
    dataset_dir = Path(dataset_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    
    if not (dataset_dir / "episode_initial_states.json").exists():
        raise FileNotFoundError(
            f"数据集缺少episode_initial_states.json: {dataset_dir}\n"
            f"our_v5需要每个episode的rand_vec"
        )
    
    print(f"\n步骤1: 提取V5 embeddings...")
    
    embedding_script = PROJECT_ROOT / "personal" / "work2" / "embedding_utils" / "ensure_embeddings_v5.py"
    if not embedding_script.exists():
        raise FileNotFoundError(f"Embedding脚本不存在: {embedding_script}")
    
    cmd_embedding = [
        sys.executable, str(embedding_script),
        "--dataset-root", dataset_root,
        "--dataset-name", dataset_name,
        "--gpu-id", "0",
        "--pca-dim", "32",
    ]
    
    print(f"运行: {' '.join(cmd_embedding)}")
    sys.stdout.flush()
    
    result = subprocess.run(cmd_embedding, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Embedding提取失败，退出码 {result.returncode}")
    
    print(f"\n步骤2: 运行V5 episode选择...")
    
    embedding_dir = Path(PROJECT_ROOT) / "personal" / "work2" / "embedding_cache" / dataset_name / "smolvla2_500m" / "corner"
    action_dir = embedding_dir / "action_descriptors"
    
    if not embedding_dir.exists():
        embedding_dir = Path(PROJECT_ROOT) / "personal" / "work2" / "embedding_cache" / "smolvla2_500m" / dataset_name / "corner"
    
    if not embedding_dir.exists():
        print(f"查找embeddings目录...")
        for root, dirs, files in Path(PROJECT_ROOT).walk():
            if "episode_0.json" in files and "smolvla" in str(root).lower():
                print(f"找到: {root}")
                embedding_dir = root
                break
    
    cmd_v5 = [
        sys.executable, str(select_script),
        "--visual-embedding-dir", str(embedding_dir),
        "--action-descriptor-dir", str(action_dir),
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(v5_dataset_dir),
        "--num-selected", str(episodes_per_dataset),
        "--seed", str(seed),
        "--visual-weight", "0.5",
        "--action-weight", "0.5",
        "--region-ratio", "0.1",
        "--min-regions", "16",
        "--max-regions", "128",
        "--b0-region-ratio", "0.2",
        "--coverage-weight", "0.5",
        "--region-visual-weight", "0.3",
        "--region-action-weight", "0.2",
    ]
    
    print(f"运行: {' '.join(cmd_v5)}")
    sys.stdout.flush()
    
    result = subprocess.run(cmd_v5, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"V5选择失败，退出码 {result.returncode}")
    
    if not v5_subset_file.exists():
        raise FileNotFoundError(f"V5 subset文件未生成: {v5_subset_file}")
    
    print(f"✓ V5选择完成: {v5_subset_file}")
    return str(v5_subset_file)


def main():
    parser = argparse.ArgumentParser(description="为多个数据集运行V5 episode选择")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="数据集根目录")
    parser.add_argument("--dataset-names", type=str, nargs="+", required=True,
                       help="数据集名称列表")
    parser.add_argument("--episodes-per-dataset", type=int, required=True,
                       help="每个数据集选择的episode数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="V5输出目录（包含subset JSON文件）")
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# 为多个数据集运行V5 Episode选择")
    print(f"# 数据集: {', '.join(args.dataset_names)}")
    print(f"# 每个数据集: {args.episodes_per_dataset} episodes")
    print(f"# Seed: {args.seed}")
    print(f"# 输出目录: {args.output_dir}")
    print(f"{'#'*60}")
    
    subset_files = []
    for dataset_name in args.dataset_names:
        dataset_config = DATASET_CONFIGS.get(dataset_name)
        if dataset_config is None:
            raise ValueError(f"未知数据集: {dataset_name}。可用: {list(DATASET_CONFIGS.keys())}")
        
        dataset_root = dataset_config["dataset_root"]
        
        subset_file = run_v5_for_dataset(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            episodes_per_dataset=args.episodes_per_dataset,
            seed=args.seed,
            v5_output_dir=args.output_dir,
        )
        subset_files.append(subset_file)
    
    print(f"\n{'='*60}")
    print(f"所有数据集的V5选择完成！")
    print(f"{'='*60}")
    for i, (name, file) in enumerate(zip(args.dataset_names, subset_files)):
        print(f"{i+1}. {name}: {file}")
    
    print(f"\n现在可以运行merge脚本:")
    print(f"python personal/work2/duibi/train_and_eval_scripts/merge_selected_episodes.py \\")
    print(f"    --dataset-root {args.dataset_root} \\")
    print(f"    --dataset-names {' '.join(args.dataset_names)} \\")
    print(f"    --episodes-per-dataset {args.episodes_per_dataset} \\")
    print(f"    --selection-mode ours_v5 \\")
    print(f"    --seed {args.seed} \\")
    print(f"    --v5-output-dir {args.output_dir} \\")
    print(f"    --output-dir personal/work2/dataset_view/merged_3tasks_v5_336_seed42")


if __name__ == "__main__":
    main()