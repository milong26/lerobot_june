"""
Auto VLM Embedding Extraction Pipeline

Selects the appropriate VLM extractor based on config, checks embedding cache,
and runs extraction if needed.
"""

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig
from differentvlm.vlm_extract.base_extractor import BaseVLMExtractor


def get_extractor(cfg: VLMExperimentConfig) -> BaseVLMExtractor:
    """Get the appropriate VLM extractor based on config."""
    if cfg.is_prismatic:
        from differentvlm.vlm_extract.prismatic_qwen_extractor import PrismaticQwenExtractor
        return PrismaticQwenExtractor(
            model_id=cfg.hf_model_id,
            device="cuda",
            pca_dim=cfg.pca_dim,
            output_dir=cfg.embedding_cache_dir,
            base_model_id=cfg.prismatic_base_model_id,
        )
    else:
        from differentvlm.vlm_extract.llava_pythia_extractor import LlavaPythiaExtractor
        return LlavaPythiaExtractor(
            model_id=cfg.hf_model_id,
            device="cuda",
            pca_dim=cfg.pca_dim,
            output_dir=cfg.embedding_cache_dir,
        )


def check_embedding_cache(embedding_dir: str, expected_episodes: int, pca_dim: int) -> bool:
    """Check if embedding cache is complete."""
    cache_dir = Path(embedding_dir)
    if not cache_dir.exists():
        return False

    pca_dir = cache_dir / "pca_models"
    pca_g = pca_dir / f"pca_global_{pca_dim}.joblib"
    pca_w = pca_dir / f"pca_wrist_{pca_dim}.joblib"
    if not (pca_g.exists() and pca_w.exists()):
        return False

    count = 0
    for ep_idx in range(expected_episodes):
        ep_file = cache_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            count += 1

    return count >= expected_episodes


def auto_extract_embedding(cfg: VLMExperimentConfig) -> str:
    """
    Auto VLM selection and embedding generation:
    1. Check if embedding cache exists
    2. If not, load HuggingFace model and extract
    3. Save embeddings
    Returns the embedding directory path.
    """
    print(f"\n{'='*60}")
    print(f"Auto VLM Embedding Extraction")
    print(f"{'='*60}")
    print(f"VLM: {cfg.vlm_name}")
    print(f"Model: {cfg.hf_model_id}")
    print(f"Embedding dir: {cfg.embedding_cache_dir}")
    print(f"Dataset: {cfg.dataset_root}")
    print(f"PCA dim: {cfg.pca_dim}")
    print(f"GPU: {cfg.gpu_id}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    if check_embedding_cache(cfg.embedding_cache_dir, 2000, cfg.pca_dim):
        print(f"Embedding cache already exists and is complete: {cfg.embedding_cache_dir}")
        print(f"Skipping extraction.")
        sys.stdout.flush()
        return cfg.embedding_cache_dir

    print(f"Embedding cache not found or incomplete. Starting extraction...")
    sys.stdout.flush()

    extractor = get_extractor(cfg)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        repo_id=cfg.lerobot_repo_id,
        root=cfg.dataset_root,
    )
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    sys.stdout.flush()

    result = extractor.extract_and_save_all(dataset, output_dir=Path(cfg.embedding_cache_dir))

    print(f"\nExtraction summary:")
    print(f"  Total episodes: {result['total_episodes']}")
    print(f"  Success: {result['success_count']}")
    print(f"  Failed: {result['fail_count']}")
    print(f"  Output dir: {result['output_dir']}")
    sys.stdout.flush()

    return cfg.embedding_cache_dir