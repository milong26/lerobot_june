"""
Auto VLM Embedding Extraction Pipeline

Selects the appropriate VLM extractor based on config, checks embedding cache,
and runs extraction if needed.

Cache validation logic:
1. Check if embedding directory exists
2. Check if episode_0.json exists and contains valid metadata
3. Validate model_name matches expected selection VLM
4. Validate camera matches expected camera
5. If all valid -> reuse cache
6. If any mismatch -> re-extract
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
            model_id=cfg.selection_vlm_model_id,
            model_name=cfg.vlm_name,
            camera=cfg.camera,
            device="cuda",
            pca_dim=cfg.pca_dim,
            output_dir=cfg.embedding_cache_dir,
            base_model_id=cfg.prismatic_base_model_id,
            vision_model_id=cfg.prismatic_vision_model_id,
        )
    else:
        from differentvlm.vlm_extract.llava_pythia_extractor import LlavaPythiaExtractor
        return LlavaPythiaExtractor(
            model_id=cfg.selection_vlm_model_id,
            model_name=cfg.vlm_name,
            camera=cfg.camera,
            device="cuda",
            pca_dim=cfg.pca_dim,
            output_dir=cfg.embedding_cache_dir,
        )


def check_embedding_cache(embedding_dir: str, expected_model_name: str, expected_camera: str) -> tuple:
    """
    Check if embedding cache exists and matches expected model and camera.
    Returns (is_valid, reason, episode_count).
    """
    cache_dir = Path(embedding_dir)
    if not cache_dir.exists():
        return False, f"Directory does not exist: {embedding_dir}", 0

    ep0_file = cache_dir / "episode_0.json"
    if not ep0_file.exists():
        return False, f"episode_0.json not found in {embedding_dir}", 0

    import json
    with open(ep0_file, "r") as f:
        meta = json.load(f)

    cached_model = meta.get("model_name", "unknown")
    cached_camera = meta.get("camera", "unknown")

    if cached_model != expected_model_name:
        return False, f"Model mismatch: cached={cached_model}, expected={expected_model_name}", 0

    if cached_camera != expected_camera:
        return False, f"Camera mismatch: cached={cached_camera}, expected={expected_camera}", 0

    pca_dir = cache_dir / "pca_models"
    pca_g = pca_dir / f"pca_global_32.joblib"
    pca_w = pca_dir / f"pca_wrist_32.joblib"
    if not (pca_g.exists() and pca_w.exists()):
        return False, f"PCA models not found (need pca_global_32.joblib and pca_wrist_32.joblib)", 0

    count = 0
    for ep_file in cache_dir.glob("episode_*.json"):
        count += 1

    return True, f"Cache valid: model={cached_model}, camera={cached_camera}, episodes={count}", count


def auto_extract_embedding(cfg: VLMExperimentConfig) -> str:
    """
    Auto VLM selection and embedding generation:
    1. Check if embedding cache exists with correct model_name and camera
    2. If cache valid -> reuse
    3. If cache invalid/missing -> load VLM -> GPU inference -> save embedding
    All steps print detailed logs.
    Returns the embedding directory path.
    """
    print(f"\n{'='*60}")
    print(f"Auto VLM Embedding Extraction")
    print(f"{'='*60}")
    print(f"VLM name: {cfg.vlm_name}")
    print(f"Selection VLM model ID: {cfg.selection_vlm_model_id}")
    print(f"Embedding dir: {cfg.embedding_cache_dir}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"PCA dim: {cfg.pca_dim}")
    print(f"GPU: {cfg.gpu_id}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    is_valid, reason, episode_count = check_embedding_cache(
        cfg.embedding_cache_dir,
        cfg.vlm_name,
        cfg.camera,
    )

    if is_valid:
        print(f"[CACHE HIT] {reason}")
        print(f"Reusing existing embeddings. Skipping extraction.")
        sys.stdout.flush()
        return cfg.embedding_cache_dir

    print(f"[CACHE MISS] {reason}")
    print(f"Starting fresh embedding extraction...")
    sys.stdout.flush()

    extractor = get_extractor(cfg)

    print(f"\nLoading dataset...")
    print(f"  Repo: {cfg.lerobot_repo_id}")
    print(f"  Root: {cfg.dataset_root}")
    sys.stdout.flush()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        repo_id=cfg.lerobot_repo_id,
        root=cfg.dataset_root,
    )
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    sys.stdout.flush()

    result = extractor.extract_and_save_all(dataset, output_dir=Path(cfg.embedding_cache_dir))

    print(f"\n{'='*60}")
    print(f"Extraction summary:")
    print(f"  Total episodes: {result['total_episodes']}")
    print(f"  Success: {result['success_count']}")
    print(f"  Failed: {result['fail_count']}")
    print(f"  Output dir: {result['output_dir']}")
    print(f"  Model: {result['model_name']}")
    print(f"  Camera: {result['camera']}")
    print(f"  PCA dim: {result['pca_dim']}")
    print(f"{'='*60}")
    sys.stdout.flush()

    return cfg.embedding_cache_dir