"""
Base VLM Embedding Extractor Abstract Interface

Defines the unified interface for different VLM backends to extract episode-level
visual embeddings. All concrete extractors must implement these methods.

Unified output format (per episode):
- JSON metadata file: episode_{idx}.json
  {
    "episode_id": int,
    "global_embedding": list[float],
    "wrist_embedding": list[float],
    "embedding_dim": int,
    "model_name": str,
    "camera": str,
    "n_global_frames": int,
    "n_wrist_frames": int,
  }
- PT tensor file: episode_{idx}.pt
  {
    "phi_global": torch.Tensor (dim,),
    "phi_wrist": torch.Tensor (dim,),
    "episode_id": int,
    "model_name": str,
    "camera": str,
  }

This unified format ensures select_v4_wrapper does NOT need to check VLM type.
"""

import sys
import os
import time
import json
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

sys.stdout.reconfigure(line_buffering=True)


class BaseVLMExtractor(ABC):
    """Abstract base class for VLM embedding extraction."""

    def __init__(
        self,
        model_id: str,
        model_name: str,
        camera: str = "corner",
        device: str = "cuda",
        pca_dim: int = 32,
        output_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.camera = camera
        self.device = device
        self.pca_dim = pca_dim
        self.output_dir = Path(output_dir) if output_dir else None
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> Tuple[torch.nn.Module, object]:
        """Load the VLM model and processor. Returns (model, processor)."""
        pass

    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Encode a single image tensor to a 1-D embedding vector.
        MUST use the vision encoder to extract visual features.
        MUST NOT use text embedding or random projection.

        Args:
            image: torch.Tensor of shape (C, H, W) or (H, W, C)
        Returns:
            np.ndarray 1-D embedding vector
        """
        pass

    def encode_images_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encode a batch of image tensors to embedding vectors.
        Default implementation loops over single images (slow).
        Subclasses should override for true batch processing (fast).

        Args:
            images: torch.Tensor of shape (N, C, H, W)
        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        embs = []
        for i in range(len(images)):
            emb = self.encode_image(images[i])
            embs.append(emb)
        return np.stack(embs, axis=0)

    def extract_episode_embedding(
        self,
        global_frames: torch.Tensor,
        wrist_frames: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Extract episode-level visual embedding from global and wrist camera frames.
        Only uses visual information from camera images.
        Does NOT read obj_init_pos, goal_pose, environment_state, success, grasp_success, eval results.

        Args:
            global_frames: torch.Tensor of shape (N, C, H, W)
            wrist_frames: torch.Tensor of shape (M, C, H, W)
        Returns:
            {"phi_global": np.ndarray, "phi_wrist": np.ndarray}
        """
        n_global = len(global_frames)
        n_wrist = len(wrist_frames)

        global_start = 0
        global_end = min(5, n_global)
        global_selected = global_frames[global_start:global_end]

        wrist_start = int(n_wrist * 0.2)
        wrist_end = int(n_wrist * 0.7)
        wrist_selected = wrist_frames[wrist_start:wrist_end]

        print(f"  Encoding {len(global_selected)} global frames (from {n_global} total)...")
        sys.stdout.flush()
        global_embs = self.encode_images_batch(global_selected)
        phi_global = np.mean(global_embs, axis=0)

        print(f"  Encoding {len(wrist_selected)} wrist frames (from {n_wrist} total)...")
        sys.stdout.flush()
        wrist_embs = self.encode_images_batch(wrist_selected)
        phi_wrist = np.mean(wrist_embs, axis=0)

        return {
            "phi_global": phi_global,
            "phi_wrist": phi_wrist,
            "n_global_frames": n_global,
            "n_wrist_frames": n_wrist,
        }

    def save_embedding(
        self,
        episode_index: int,
        phi_global: np.ndarray,
        phi_wrist: np.ndarray,
        n_global_frames: int = 0,
        n_wrist_frames: int = 0,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save episode embedding in unified JSON + PT format.
        Format is identical regardless of VLM type.
        """
        out_dir = output_dir or self.output_dir
        if out_dir is None:
            raise ValueError("output_dir must be specified")
        out_dir.mkdir(parents=True, exist_ok=True)

        embedding_dim = phi_global.shape[0]

        json_data = {
            "episode_id": int(episode_index),
            "global_embedding": phi_global.tolist(),
            "wrist_embedding": phi_wrist.tolist(),
            "embedding_dim": int(embedding_dim),
            "model_name": self.model_name,
            "camera": self.camera,
            "n_global_frames": int(n_global_frames),
            "n_wrist_frames": int(n_wrist_frames),
        }

        json_file = out_dir / f"episode_{episode_index}.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f)

        pt_data = {
            "phi_global": torch.from_numpy(phi_global.astype(np.float32)),
            "phi_wrist": torch.from_numpy(phi_wrist.astype(np.float32)),
            "episode_id": int(episode_index),
            "model_name": self.model_name,
            "camera": self.camera,
        }
        pt_file = out_dir / f"episode_{episode_index}.pt"
        torch.save(pt_data, pt_file)

        return json_file

    @staticmethod
    def load_embedding(episode_index: int, embedding_dir: Path) -> Dict[str, np.ndarray]:
        """Load episode embedding from unified JSON format."""
        json_file = embedding_dir / f"episode_{episode_index}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Embedding not found: {json_file}")
        with open(json_file, "r") as f:
            data = json.load(f)
        return {
            "phi_global": np.array(data["global_embedding"], dtype=np.float32),
            "phi_wrist": np.array(data["wrist_embedding"], dtype=np.float32),
            "episode_id": data["episode_id"],
            "model_name": data["model_name"],
            "camera": data["camera"],
            "embedding_dim": data["embedding_dim"],
        }

    @staticmethod
    def check_embedding_metadata(embedding_dir: Path, expected_model_name: str, expected_camera: str) -> Tuple[bool, str]:
        """
        Check if embedding cache exists and matches the expected model and camera.
        Returns (is_valid, reason).
        """
        if not embedding_dir.exists():
            return False, f"Directory does not exist: {embedding_dir}"

        ep0_file = embedding_dir / "episode_0.json"
        if not ep0_file.exists():
            return False, f"episode_0.json not found in {embedding_dir}"

        with open(ep0_file, "r") as f:
            meta = json.load(f)

        cached_model = meta.get("model_name", "unknown")
        cached_camera = meta.get("camera", "unknown")

        if cached_model != expected_model_name:
            return False, f"Model mismatch: cached={cached_model}, expected={expected_model_name}"

        if cached_camera != expected_camera:
            return False, f"Camera mismatch: cached={cached_camera}, expected={expected_camera}"

        return True, f"Cache valid: model={cached_model}, camera={cached_camera}"

    def extract_and_save_all(
        self,
        dataset,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Extract embeddings for all episodes in the dataset and save them.
        Returns a summary dict.
        """
        out_dir = output_dir or self.output_dir
        if out_dir is None:
            raise ValueError("output_dir must be specified")
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            print("Loading model...")
            sys.stdout.flush()
            self.model, self.processor = self.load_model()

        total_episodes = dataset.num_episodes
        print(f"\nExtracting embeddings for {total_episodes} episodes...")
        print(f"  Model: {self.model_name}")
        print(f"  Model ID: {self.model_id}")
        print(f"  Camera: {self.camera}")
        print(f"  Device: {self.device}")
        print(f"  Output: {out_dir}")
        sys.stdout.flush()

        ep_indices_map = {}
        for ep_idx in range(total_episodes):
            from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
            to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
            ep_indices_map[ep_idx] = list(range(from_idx, to_idx))

        global_embs_list = []
        wrist_embs_list = []
        episode_coords = []
        success_count = 0
        fail_count = 0

        # Chunked batch processing: process episodes in chunks to avoid OOM
        # Each chunk collects frames, batch encodes, then frees memory
        chunk_size = 25  # episodes per chunk
        num_chunks = (total_episodes + chunk_size - 1) // chunk_size
        print(f"\nChunked batch processing: {total_episodes} episodes in {num_chunks} chunks of {chunk_size}")
        sys.stdout.flush()

        for chunk_idx in range(num_chunks):
            chunk_start_ep = chunk_idx * chunk_size
            chunk_end_ep = min(chunk_start_ep + chunk_size, total_episodes)
            print(f"\n{'='*40}")
            print(f"Chunk {chunk_idx + 1}/{num_chunks}: episodes {chunk_start_ep}-{chunk_end_ep - 1}")
            print(f"{'='*40}")
            sys.stdout.flush()

            # Collect frames for this chunk
            all_global_frames = []
            all_wrist_frames = []
            ep_frame_counts = []  # Track (n_global, n_wrist) per episode

            for ep_idx in range(chunk_start_ep, chunk_end_ep):
                episode_indices = ep_indices_map.get(ep_idx, [])
                if not episode_indices:
                    ep_frame_counts.append((0, 0))
                    continue

                episode_frames_global = []
                episode_frames_wrist = []
                for idx in episode_indices:
                    frame = dataset[idx]
                    episode_frames_global.append(frame["observation.images.top"])
                    episode_frames_wrist.append(frame["observation.images.wrist"])

                global_frames = torch.stack(episode_frames_global)
                wrist_frames = torch.stack(episode_frames_wrist)
                all_global_frames.append(global_frames)
                all_wrist_frames.append(wrist_frames)
                ep_frame_counts.append((len(global_frames), len(wrist_frames)))

            total_global = sum(f[0] for f in ep_frame_counts)
            total_wrist = sum(f[1] for f in ep_frame_counts)
            print(f"  Chunk frames: {total_global} global, {total_wrist} wrist")
            print(f"  Batch encoding chunk frames...")
            sys.stdout.flush()

            # Batch encode chunk's global frames
            if total_global > 0:
                all_global_stacked = torch.cat([f for f in all_global_frames if len(f) > 0], dim=0)
                all_global_embs = self.encode_images_batch(all_global_stacked)
                del all_global_stacked
            else:
                all_global_embs = np.array([])

            # Batch encode chunk's wrist frames
            if total_wrist > 0:
                all_wrist_stacked = torch.cat([f for f in all_wrist_frames if len(f) > 0], dim=0)
                all_wrist_embs = self.encode_images_batch(all_wrist_stacked)
                del all_wrist_stacked
            else:
                all_wrist_embs = np.array([])

            # Free collected frames memory
            del all_global_frames
            del all_wrist_frames

            print(f"  Chunk encoding complete. Computing per-episode embeddings...")
            sys.stdout.flush()

            # Split results per episode and save
            global_offset = 0
            wrist_offset = 0

            for i, ep_idx in enumerate(range(chunk_start_ep, chunk_end_ep)):
                ep_start = time.time()
                print(f"\nEpisode {ep_idx + 1}/{total_episodes} (index {ep_idx})")
                sys.stdout.flush()

                n_global, n_wrist = ep_frame_counts[i]
                if n_global == 0 and n_wrist == 0:
                    print(f"  SKIP: no frames for episode {ep_idx}")
                    fail_count += 1
                    continue

                try:
                    # Get embeddings for this episode from the batch results
                    if n_global > 0:
                        ep_global_embs = all_global_embs[global_offset:global_offset + n_global]
                        phi_global = np.mean(ep_global_embs, axis=0)
                    else:
                        phi_global = np.zeros(all_global_embs.shape[1] if len(all_global_embs) > 0 else 0)

                    if n_wrist > 0:
                        ep_wrist_embs = all_wrist_embs[wrist_offset:wrist_offset + n_wrist]
                        phi_wrist = np.mean(ep_wrist_embs, axis=0)
                    else:
                        phi_wrist = np.zeros(all_wrist_embs.shape[1] if len(all_wrist_embs) > 0 else 0)

                    self.save_embedding(
                        ep_idx,
                        phi_global,
                        phi_wrist,
                        n_global,
                        n_wrist,
                        out_dir,
                    )

                    global_embs_list.append(phi_global)
                    wrist_embs_list.append(phi_wrist)
                    episode_coords.append(ep_idx)
                    success_count += 1

                    global_offset += n_global
                    wrist_offset += n_wrist

                    ep_time = time.time() - ep_start
                    progress = success_count / total_episodes * 100
                    print(f"  DONE: ep={ep_idx}, time={ep_time:.1f}s, progress={progress:.1f}%")
                    sys.stdout.flush()

                except Exception as e:
                    print(f"  FAIL: ep={ep_idx}, error={e}")
                    sys.stdout.flush()
                    fail_count += 1

            # Free chunk encoding results
            del all_global_embs
            del all_wrist_embs

            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            print(f"\nChunk {chunk_idx + 1} complete. Memory freed.")
            sys.stdout.flush()

        print(f"\nExtraction complete: {success_count} succeeded, {fail_count} failed")
        sys.stdout.flush()

        if global_embs_list:
            from sklearn.decomposition import PCA
            global_embs_array = np.array(global_embs_list)
            wrist_embs_array = np.array(wrist_embs_list)

            pca_global = PCA(n_components=self.pca_dim, random_state=42)
            pca_global.fit(global_embs_array)
            explained_var_g = pca_global.explained_variance_ratio_.sum()

            pca_wrist = PCA(n_components=self.pca_dim, random_state=42)
            pca_wrist.fit(wrist_embs_array)
            explained_var_w = pca_wrist.explained_variance_ratio_.sum()

            pca_dir = out_dir / "pca_models"
            pca_dir.mkdir(parents=True, exist_ok=True)

            import joblib
            joblib.dump(pca_global, pca_dir / f"pca_global_{self.pca_dim}.joblib")
            joblib.dump(pca_wrist, pca_dir / f"pca_wrist_{self.pca_dim}.joblib")

            for i, ep_idx in enumerate(episode_coords):
                phi_g = pca_global.transform(global_embs_array[i:i+1])[0]
                phi_w = pca_wrist.transform(wrist_embs_array[i:i+1])[0]
                self.save_embedding(
                    ep_idx,
                    phi_g,
                    phi_w,
                    output_dir=out_dir,
                )

            print(f"PCA applied: global_var={explained_var_g:.4f}, wrist_var={explained_var_w:.4f}")
            sys.stdout.flush()

        return {
            "total_episodes": total_episodes,
            "success_count": success_count,
            "fail_count": fail_count,
            "output_dir": str(out_dir),
            "pca_dim": self.pca_dim,
            "model_name": self.model_name,
            "camera": self.camera,
        }