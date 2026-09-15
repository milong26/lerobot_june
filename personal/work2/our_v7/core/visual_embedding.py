"""Our-V7 visual representation.

The actual frozen-VLM extraction and the two visual projectors are intentionally
reused from Our-V6 so V6/V7 differ only in selection logic, not in visual
feature computation. A disk cache is added for raw acquired features. The
causal gate is checked before any cache lookup: an unacquired episode cannot be
loaded merely because another run has already cached it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

from our_v6.core.visual_embedding import (
    CausalVisualProjector,
    FrozenVLMEpisodeEncoder,
)
from our_v7.config import RAW_VISUAL_CACHE_VERSION


class CachedFrozenVLMEpisodeEncoder(FrozenVLMEpisodeEncoder):
    """V6-equivalent extractor with acquired-gated persistent raw caching."""

    def __init__(
        self,
        dataset,
        dataset_name: str,
        device: str = "cuda",
        cache_root: str | Path | None = None,
    ) -> None:
        super().__init__(dataset=dataset, dataset_name=dataset_name, device=device)
        if cache_root is None:
            work2_root = Path(__file__).resolve().parents[2]
            cache_root = work2_root / "our_v7" / "shared_raw_embeddings"
        cache_root = Path(cache_root)

        source_signature = "|".join(
            [
                str(Path(dataset.root).resolve()),
                str(dataset.num_episodes),
                str(getattr(dataset.meta, "total_frames", "unknown")),
                str(dataset.meta.fps),
                self.global_key,
                self.wrist_key,
                RAW_VISUAL_CACHE_VERSION,
            ]
        )
        short_hash = hashlib.sha1(source_signature.encode("utf-8")).hexdigest()[:12]
        safe_name = dataset_name.replace("/", "_")
        self.cache_dir = cache_root / safe_name / f"{RAW_VISUAL_CACHE_VERSION}_{short_hash}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, ep_idx: int) -> Path:
        return self.cache_dir / f"episode_{int(ep_idx):06d}.npz"

    @staticmethod
    def _validate_cached(payload: Dict[str, np.ndarray], ep_idx: int) -> Dict[str, np.ndarray]:
        result = {}
        for key in ("phi_global", "phi_wrist"):
            if key not in payload:
                raise ValueError(f"Cached VLM feature for episode {ep_idx} missing {key}")
            arr = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                raise ValueError(f"Invalid cached {key} for episode {ep_idx}")
            result[key] = arr
        return result

    def extract(self, ep_idx: int, acquired_indices: Set[int]) -> Dict[str, np.ndarray]:
        # Gate FIRST. A cache created by another run never grants algorithmic
        # access to an episode that has not yet been acquired in this run.
        if ep_idx not in acquired_indices:
            raise RuntimeError(
                f"CAUSAL VIOLATION: visual content of episode {ep_idx} requested before acquisition"
            )

        if ep_idx in self._raw_cache:
            return self._raw_cache[ep_idx]

        path = self._cache_file(ep_idx)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    result = self._validate_cached(
                        {"phi_global": data["phi_global"], "phi_wrist": data["phi_wrist"]}, ep_idx
                    )
                self._raw_cache[ep_idx] = result
                print(f"  [V7 raw visual cache hit] episode={ep_idx} path={path}")
                return result
            except Exception as exc:
                print(f"  [V7 raw visual cache invalid] episode={ep_idx}: {exc}; recomputing")

        # Exact V6 extraction path: same frame selection, frozen VLM, and pooling.
        result = super().extract(ep_idx, acquired_indices)
        result = self._validate_cached(result, ep_idx)

        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        try:
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f,
                    phi_global=result["phi_global"],
                    phi_wrist=result["phi_wrist"],
                )
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return result


__all__ = ["CachedFrozenVLMEpisodeEncoder", "CausalVisualProjector"]
