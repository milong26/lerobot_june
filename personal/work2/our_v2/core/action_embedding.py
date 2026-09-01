"""
Action Trajectory Embedding Extraction Module

Reads trace.pt files produced by the attention analysis pipeline and extracts
fixed-length action representations from x_t, v_t, and hidden representations.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_action_trace(trace_path: Path) -> Optional[Dict]:
    """
    Load a single episode's action trace from trace.pt.

    Args:
        trace_path: path to trace.pt file.

    Returns:
        Dict with keys: x_t, v_t, suffix_hidden, timesteps, final_action, etc.
        or None if loading fails.
    """
    if not trace_path.exists():
        return None

    try:
        trace = torch.load(trace_path, weights_only=False)
        return trace
    except Exception as e:
        print(f"  Failed to load trace {trace_path}: {e}")
        return None


def extract_action_embedding(
    trace: Dict,
    method: str = "combined",
) -> np.ndarray:
    """
    Extract a fixed-length action representation from a trace dict.

    Supported methods:
        "xt_stats": mean and std of x_t over denoising steps.
        "vt_stats": mean and std of v_t trajectory.
        "hidden_mean": mean-pooled suffix hidden representation.
        "combined": concatenate xt_stats + vt_stats + hidden_mean (default).

    Args:
        trace: dict from load_action_trace.
        method: extraction method.

    Returns:
        1-D numpy array representing the action embedding.
    """
    parts = []

    x_t = trace.get("x_t")
    v_t = trace.get("v_t")
    suffix_hidden = trace.get("suffix_hidden")
    final_action = trace.get("final_action")

    if method in ("xt_stats", "combined") and x_t is not None and x_t.numel() > 0:
        x_t_flat = x_t.reshape(x_t.shape[0], -1)
        x_mean = x_t_flat.mean(dim=0).numpy()
        x_std = x_t_flat.std(dim=0).numpy()
        parts.append(np.concatenate([x_mean, x_std]))

    if method in ("vt_stats", "combined") and v_t is not None and v_t.numel() > 0:
        v_t_flat = v_t.reshape(v_t.shape[0], -1)
        v_mean = v_t_flat.mean(dim=0).numpy()
        v_std = v_t_flat.std(dim=0).numpy()
        parts.append(np.concatenate([v_mean, v_std]))

    if method in ("hidden_mean", "combined") and suffix_hidden is not None and suffix_hidden.numel() > 0:
        h_flat = suffix_hidden.reshape(suffix_hidden.shape[0], -1)
        h_mean = h_flat.mean(dim=0).numpy()
        parts.append(h_mean)

    if method == "final_action" and final_action is not None and final_action.numel() > 0:
        parts.append(final_action.reshape(-1).numpy())

    if not parts:
        raise ValueError("No valid action data found in trace for extraction.")

    return np.concatenate(parts)


def build_action_embeddings(
    trace_dir: Path,
    episode_indices: Optional[List[int]] = None,
    method: str = "combined",
) -> Dict[int, np.ndarray]:
    """
    Batch-generate action embeddings for episodes.

    Looks for trace.pt files under trace_dir/<episode_index>/ or directly
    under trace_dir if structured differently.

    Args:
        trace_dir: directory containing episode trace files.
        episode_indices: optional list of episode indices to process.
                         If None, auto-detect from directory structure.
        method: extraction method passed to extract_action_embedding.

    Returns:
        Dict[episode_index, action_embedding_array]
    """
    action_embeddings = {}

    if episode_indices is None:
        episode_indices = []
        for d in sorted(trace_dir.iterdir()):
            if d.is_dir():
                try:
                    ep_idx = int(d.name)
                    episode_indices.append(ep_idx)
                except ValueError:
                    pass

    for ep_idx in episode_indices:
        trace_path = trace_dir / str(ep_idx) / "trace.pt"
        if not trace_path.exists():
            trace_path = trace_dir / f"episode_{ep_idx}" / "trace.pt"
        if not trace_path.exists():
            continue

        trace = load_action_trace(trace_path)
        if trace is None:
            continue

        try:
            emb = extract_action_embedding(trace, method=method)
            action_embeddings[ep_idx] = emb
        except ValueError as e:
            print(f"  Skip episode {ep_idx}: {e}")

    print(f"Built action embeddings for {len(action_embeddings)} episodes")
    return action_embeddings