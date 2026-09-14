#!/usr/bin/env python3
"""
只运行V5 episode选择，不启动训练
为单个数据集生成subset文件
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORK2_ROOT = Path(__file__).resolve().parents[2]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from embedding_utils.ensure_embeddings_v5 import ensure_v5_embeddings_exist
from embedding_utils.config import DEFAULT_PCA_DIM


def run_v5_selection(dataset_name: str, dataset_root: str, num_episodes: int, seed: int, gpu_id: int, output_dir: str):
    """运行V5选择流程"""
    print(f"\n{'='*60}")
    print(f"V5 Episode Selection: {dataset_name}")
    print(f"{'='*60}")
    print(f"Dataset root: {dataset_root}")
    print(f"Num episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print(f"GPU: {gpu_id}")
    print(f"Output dir: {output_dir}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 1. 确保embeddings存在
    print(f"\n[1/3] 确保V5 embeddings存在...")
    visual_dir, action_dir = ensure_v5_embeddings_exist(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        gpu_id=gpu_id,
        pca_dim=DEFAULT_PCA_DIM,
    )
    print(f"Visual embeddings: {visual_dir}")
    print(f"Action descriptors: {action_dir}")
    
    # 2. 检查episode_initial_states.json
    dataset_dir = Path(dataset_root)
    if not (dataset_dir / "episode_initial_states.json").exists():
        raise FileNotFoundError(
            f"Dataset directory missing episode_initial_states.json: {dataset_dir}\n"
            f"our_v5 requires rand_vec for each episode."
        )
    
    # 3. 运行选择脚本
    print(f"\n[2/3] 运行V5选择脚本...")
    select_script = PROJECT_ROOT / "personal" / "work2" / "our_v5" / "select_our_v5.py"
    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")
    
    v5_output_dir = Path(output_dir) / f"our_v5_{num_episodes}_seed{seed}"
    
    cmd = [
        sys.executable, str(select_script),
        "--visual-embedding-dir", str(visual_dir),
        "--action-descriptor-dir", str(action_dir),
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(v5_output_dir),
        "--num-selected", str(num_episodes),
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
    
    print(f"Running: {' '.join(cmd)}")
    import subprocess
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        raise RuntimeError(f"V5 selection failed with exit code {result.returncode}")
    
    # 4. 检查输出
    subset_file = v5_output_dir / "subsets" / f"our_v5_{num_episodes}_seed{seed}.json"
    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")
    
    print(f"\n[3/3] V5选择完成!")
    print(f"Subset file: {subset_file}")
    
    return str(subset_file)


def main():
    parser = argparse.ArgumentParser(description="V5 Episode Selection Only")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--num-episodes", type=int, default=112, help="Number of episodes to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    run_v5_selection(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        num_episodes=args.num_episodes,
        seed=args.seed,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()