"""
Base VLM Embedding Extractor Abstract Interface

Defines the unified interface for different VLM backends to extract episode-level
visual embeddings. All concrete extractors must implement these methods.
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
        device: str = "cuda",
        pca_dim: int = 32,
        output_dir: Optional[str] = None,
    ):
        self.model_id = model_id
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
        Args:
            image: torch.Tensor of shape (C, H, W) or (H, W, C)
        Returns:
            np.ndarray 1-D embedding vector
        """
        pass

    def extract_episode_embedding(
        self,
        global_frames: torch.Tensor,
        wrist_frames: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Extract episode-level visual embedding from global and wrist camera frames.
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

        print(f"  Encoding {len(global_selected)} global frames...")
        sys.stdout.flush()
        global_embs = []
        for i in range(len(global_selected)):
            emb = self.encode_image(global_selected[i])
            global_embs.append(emb)
        phi_global = np.mean(global_embs, axis=0)

        print(f"  Encoding {len(wrist_selected)} wrist frames...")
        sys.stdout.flush()
        wrist_embs = []
        for i in range(len(wrist_selected)):
            emb = self.encode_image(wrist_selected[i])
            wrist_embs.append(emb)
        phi_wrist = np.mean(wrist_embs, axis=0)

        return {"phi_global": phi_global, "phi_wrist": phi_wrist}

    def save_embedding(
        self,
        episode_index: int,
        phi_global: np.ndarray,
        phi_wrist: np.ndarray,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save episode embedding to .npy file compatible with our_v4 format."""
        out_dir = output_dir or self.output_dir
        if out_dir is None:
            raise ValueError("output_dir must be specified")
        out_dir.mkdir(parents=True, exist_ok=True)

        coord_key = f"({episode_index})"
        output_file = out_dir / f"{coord_key}.npy"

        np.save(output_file, {
            "phi_global": phi_global,
            "phi_wrist": phi_wrist,
            "episode_index": episode_index,
        }, allow_pickle=True)
        return output_file

    @staticmethod
    def load_embedding(episode_index: int, embedding_dir: Path) -> Dict[str, np.ndarray]:
        """Load episode embedding from .npy file."""
        coord_key = f"({episode_index})"
        embedding_file = embedding_dir / f"{coord_key}.npy"
        if not embedding_file.exists():
            raise FileNotFoundError(f"Embedding not found: {embedding_file}")
        data = np.load(embedding_file, allow_pickle=True).item()
        return {
            "phi_global": data["phi_global"],
            "phi_wrist": data["phi_wrist"],
        }

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

        for ep_idx in range(total_episodes):
            ep_start = time.time()
            print(f"\nEpisode {ep_idx + 1}/{total_episodes} (index {ep_idx})")
            sys.stdout.flush()

            try:
                episode_indices = ep_indices_map.get(ep_idx, [])
                if not episode_indices:
                    print(f"  SKIP: no frames for episode {ep_idx}")
                    fail_count += 1
                    continue

                episode_frames_global = []
                episode_frames_wrist = []
                for idx in episode_indices:
                    frame = dataset[idx]
                    episode_frames_global.append(frame["observation.images.top"])
                    episode_frames_wrist.append(frame["observation.images.wrist"])

                global_frames = torch.stack(episode_frames_global)
                wrist_frames = torch.stack(episode_frames_wrist)

                emb = self.extract_episode_embedding(global_frames, wrist_frames)

                self.save_embedding(ep_idx, emb["phi_global"], emb["phi_wrist"], out_dir)

                global_embs_list.append(emb["phi_global"])
                wrist_embs_list.append(emb["phi_wrist"])
                episode_coords.append(ep_idx)
                success_count += 1

                ep_time = time.time() - ep_start
                progress = success_count / total_episodes * 100
                print(f"  DONE: ep={ep_idx}, time={ep_time:.1f}s, progress={progress:.1f}%")
                sys.stdout.flush()

            except Exception as e:
                print(f"  FAIL: ep={ep_idx}, error={e}")
                sys.stdout.flush()
                fail_count += 1

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
                self.save_embedding(ep_idx, phi_g, phi_w, out_dir)

            print(f"PCA applied: global_var={explained_var_g:.4f}, wrist_var={explained_var_w:.4f}")
            sys.stdout.flush()

        return {
            "total_episodes": total_episodes,
            "success_count": success_count,
            "fail_count": fail_count,
            "output_dir": str(out_dir),
            "pca_dim": self.pca_dim,
        }