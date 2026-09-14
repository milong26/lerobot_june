"""Causal visual representation for our_v6.

Two variants share the same frozen SmolVLM raw episode features:
  l2: branch-wise L2 normalization -> concatenate -> final L2.
  online_pca: fit branch-wise PCA using ONLY currently acquired raw features,
              re-project ALL acquired episodes after every acquisition, branch
              L2 normalize, concatenate, final L2.

No visual feature of an unacquired episode can be requested through this module.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple
import numpy as np

from our_v6.config import (
    DEFAULT_VISUAL_VARIANT,
    VISUAL_VARIANTS,
    PCA_DIM,
    VISUAL_GLOBAL_WEIGHT,
    VISUAL_WRIST_WEIGHT,
    EPS,
)


def _l2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec if norm < EPS else vec / norm


def _pick_image_keys(dataset) -> Tuple[str, str]:
    keys = list(dataset.meta.features.keys())
    image_keys = [k for k in keys if "image" in k.lower()]
    if not image_keys:
        raise ValueError(f"No image feature found. Available={keys}")

    global_candidates = [
        "observation.images.top",
        "observation.images.camera1",
        "observation.images.corner",
    ]
    wrist_candidates = [
        "observation.images.wrist",
        "observation.images.camera2",
        "observation.images.gripperPOV",
    ]
    global_key = next((k for k in global_candidates if k in keys), None)
    wrist_key = next((k for k in wrist_candidates if k in keys), None)

    if global_key is None:
        non_wrist = [k for k in image_keys if "wrist" not in k.lower() and "gripper" not in k.lower()]
        global_key = non_wrist[0] if non_wrist else image_keys[0]
    if wrist_key is None:
        wrist_like = [k for k in image_keys if "wrist" in k.lower() or "gripper" in k.lower()]
        wrist_key = wrist_like[0] if wrist_like else image_keys[-1]
    return global_key, wrist_key


class FrozenVLMEpisodeEncoder:
    """Extract frozen SmolVLM raw features only for officially acquired episodes."""

    def __init__(self, dataset, dataset_name: str, device: str = "cuda"):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.device = device
        self.global_key, self.wrist_key = _pick_image_keys(dataset)
        self._model = None
        self._processor = None
        self._raw_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _ensure_model(self) -> None:
        if self._model is None:
            from our.embeddings.extract_embeddings import load_vlm_model
            self._model, self._processor = load_vlm_model(self.device)

    def extract(self, ep_idx: int, acquired_indices: Set[int]) -> Dict[str, np.ndarray]:
        if ep_idx not in acquired_indices:
            raise RuntimeError(
                f"CAUSAL VIOLATION: visual content of episode {ep_idx} requested before acquisition"
            )
        if ep_idx in self._raw_cache:
            return self._raw_cache[ep_idx]
        if ep_idx < 0 or ep_idx >= self.dataset.num_episodes:
            raise IndexError(f"episode {ep_idx} outside [0, {self.dataset.num_episodes})")

        self._ensure_model()
        from torchvision.transforms import ToPILImage
        from our.embeddings.extract_embeddings import extract_frame_embeddings

        from_idx = int(self.dataset.meta.episodes["dataset_from_index"][ep_idx])
        to_idx = int(self.dataset.meta.episodes["dataset_to_index"][ep_idx])
        if to_idx <= from_idx:
            raise ValueError(f"Episode {ep_idx} has no frames")

        to_pil = ToPILImage()
        global_frames = []
        wrist_frames = []
        for frame_idx in range(from_idx, to_idx):
            frame = self.dataset[frame_idx]
            g = frame[self.global_key]
            w = frame[self.wrist_key]
            global_frames.append(g if hasattr(g, "mode") else to_pil(g))
            wrist_frames.append(w if hasattr(w, "mode") else to_pil(w))

        n = len(global_frames)
        global_selected = global_frames[: min(5, n)]
        wrist_start = int(n * 0.20)
        wrist_end = max(wrist_start + 1, int(n * 0.70))
        wrist_end = min(wrist_end, n)
        wrist_selected = wrist_frames[wrist_start:wrist_end]

        global_frame_features = extract_frame_embeddings(
            self._model,
            self._processor,
            global_selected,
            device=self.device,
            dataset_name=self.dataset_name,
        )
        wrist_frame_features = extract_frame_embeddings(
            self._model,
            self._processor,
            wrist_selected,
            device=self.device,
            dataset_name=self.dataset_name,
        )
        if global_frame_features.ndim == 1:
            phi_global = global_frame_features.astype(np.float32)
        else:
            phi_global = global_frame_features.mean(axis=0).astype(np.float32)
        if wrist_frame_features.ndim == 1:
            phi_wrist = wrist_frame_features.astype(np.float32)
        else:
            phi_wrist = wrist_frame_features.mean(axis=0).astype(np.float32)

        if not np.all(np.isfinite(phi_global)) or not np.all(np.isfinite(phi_wrist)):
            raise ValueError(f"Non-finite VLM feature in episode {ep_idx}")

        result = {"phi_global": phi_global, "phi_wrist": phi_wrist}
        self._raw_cache[ep_idx] = result
        return result


class CausalVisualProjector:
    """Maintain current visual coordinates for the acquired set only."""

    def __init__(
        self,
        variant: str = DEFAULT_VISUAL_VARIANT,
        pca_dim: int = PCA_DIM,
        global_weight: float = VISUAL_GLOBAL_WEIGHT,
        wrist_weight: float = VISUAL_WRIST_WEIGHT,
    ):
        if variant not in VISUAL_VARIANTS:
            raise ValueError(f"Unknown visual variant {variant!r}; choose from {VISUAL_VARIANTS}")
        self.variant = variant
        self.pca_dim = int(pca_dim)
        self.global_weight = float(global_weight)
        self.wrist_weight = float(wrist_weight)
        self.raw: Dict[int, Dict[str, np.ndarray]] = {}
        self.projected: Dict[int, np.ndarray] = {}
        self.last_pca_components: Dict[str, int] = {"global": 0, "wrist": 0}

    def add(self, ep_idx: int, raw_feature: Dict[str, np.ndarray]) -> None:
        self.raw[int(ep_idx)] = {
            "phi_global": np.asarray(raw_feature["phi_global"], dtype=np.float32).reshape(-1),
            "phi_wrist": np.asarray(raw_feature["phi_wrist"], dtype=np.float32).reshape(-1),
        }
        self.recompute()

    def _combine(self, global_feature: np.ndarray, wrist_feature: np.ndarray) -> np.ndarray:
        combined = np.concatenate([
            self.global_weight * _l2(global_feature),
            self.wrist_weight * _l2(wrist_feature),
        ]).astype(np.float32)
        return _l2(combined)

    def _fit_transform_branch(self, matrix: np.ndarray, branch: str) -> np.ndarray:
        n_samples, raw_dim = matrix.shape
        if n_samples < 2:
            self.last_pca_components[branch] = 0
            return np.zeros((n_samples, self.pca_dim), dtype=np.float32)

        # Normalize each frozen-VLM branch before fitting PCA. PCA sees acquired rows only.
        normalized = np.stack([_l2(row) for row in matrix], axis=0)
        n_components = min(self.pca_dim, n_samples - 1, raw_dim)
        if n_components < 1:
            self.last_pca_components[branch] = 0
            return np.zeros((n_samples, self.pca_dim), dtype=np.float32)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, svd_solver="full")
        reduced = pca.fit_transform(normalized).astype(np.float32)
        padded = np.zeros((n_samples, self.pca_dim), dtype=np.float32)
        padded[:, :n_components] = reduced
        self.last_pca_components[branch] = int(n_components)
        return padded

    def recompute(self) -> None:
        """Recompute ALL acquired visual coordinates in one current coordinate system."""
        ids = sorted(self.raw.keys())
        if not ids:
            self.projected = {}
            return

        if self.variant == "l2":
            self.projected = {
                ep: self._combine(self.raw[ep]["phi_global"], self.raw[ep]["phi_wrist"])
                for ep in ids
            }
            self.last_pca_components = {"global": 0, "wrist": 0}
            return

        global_matrix = np.stack([self.raw[ep]["phi_global"] for ep in ids], axis=0)
        wrist_matrix = np.stack([self.raw[ep]["phi_wrist"] for ep in ids], axis=0)
        global_pca = self._fit_transform_branch(global_matrix, "global")
        wrist_pca = self._fit_transform_branch(wrist_matrix, "wrist")

        self.projected = {}
        for row, ep in enumerate(ids):
            self.projected[ep] = self._combine(global_pca[row], wrist_pca[row])

    def get(self, ep_idx: int) -> np.ndarray:
        if ep_idx not in self.projected:
            raise KeyError(f"No acquired visual representation for episode {ep_idx}")
        return self.projected[ep_idx]

    def as_dict(self) -> Dict[int, np.ndarray]:
        return dict(self.projected)
