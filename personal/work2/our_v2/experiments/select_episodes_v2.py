#!/usr/bin/env python
"""
Dynamic Anchor v2 Episode Selection

Command-line entry point for sequential greedy episode selection
on the state-action manifold.

Usage:
    python select_episodes_v2.py \
        --dataset-root /path/to/dataset \
        --embedding-dir /path/to/embeddings \
        --output-dir /path/to/output \
        --num-selected 112 \
        --seed 42 \
        [--action-trace-dir /path/to/traces] \
        [--use-action-embedding]
"""

import sys
import json
import warnings
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v2.core.embedding import (
    load_episode_embeddings,
    build_combined_embedding,
    build_state_action_embedding,
    normalize_embedding,
)
from our_v2.core.action_embedding import build_action_embeddings
from our_v2.core.anchors_v2 import (
    initialize_seed_set,
    sequential_greedy_selection,
    save_subset,
)
from our_v2.config import TARGET_SIZE, SEED, SEED_SIZE, KNN_K, USE_ACTION_EMBEDDING


def main():
    parser = argparse.ArgumentParser(description="Dynamic Anchor v2 Episode Selection")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--embedding-dir", type=str, required=True, help="Visual embedding cache directory")
    parser.add_argument("--action-trace-dir", type=str, default=None, help="Action trace directory (optional)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-selected", type=int, default=TARGET_SIZE, help="Target number of episodes")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--seed-size", type=int, default=SEED_SIZE, help="Initial seed set size")
    parser.add_argument("--knn-k", type=int, default=KNN_K, help="kNN parameter for redundancy")
    parser.add_argument("--use-action-embedding", action="store_true", help="Include action trajectory embedding")

    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    action_trace_dir = Path(args.action_trace_dir) if args.action_trace_dir else None

    if not embedding_dir.exists():
        print(f"Error: Embedding directory does not exist: {embedding_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dynamic Anchor v2 - Episode Selection")
    print(f"{'='*60}")
    print(f"Embedding dir: {embedding_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target episodes: {args.num_selected}")
    print(f"Seed: {args.seed}")
    print(f"Use action embedding: {args.use_action_embedding}")

    # Step 1: Load visual embeddings
    print(f"\n[Step 1] Loading visual embeddings...")
    visual_embeddings = load_episode_embeddings(embedding_dir)

    if not visual_embeddings:
        print("Error: No visual embeddings found.")
        return

    n_visual = len(visual_embeddings)
    print(f"  Visual embedding episodes: {n_visual}")

    # Step 2: Load action embeddings (optional)
    action_embeddings = None
    if args.use_action_embedding:
        if action_trace_dir and action_trace_dir.exists():
            print(f"\n[Step 2] Loading action embeddings from {action_trace_dir}...")
            action_embeddings = build_action_embeddings(
                action_trace_dir,
                episode_indices=list(visual_embeddings.keys()),
            )
            n_action = len(action_embeddings)
            print(f"  Action embedding episodes: {n_action}")
            if n_action == 0:
                warnings.warn(
                    "No action embeddings loaded. Falling back to visual-only."
                )
        else:
            trace_info = action_trace_dir if action_trace_dir else "not specified"
            warnings.warn(
                f"--use-action-embedding set but action trace dir not found: {trace_info}. "
                "Falling back to visual-only."
            )

    # Step 3: Build state-action embedding
    print(f"\n[Step 3] Building state-action embeddings...")
    sa_embeddings = build_state_action_embedding(
        visual_embeddings,
        action_embeddings=action_embeddings,
        use_action=args.use_action_embedding,
    )

    # Step 4: Normalize embeddings
    print(f"\n[Step 4] Normalizing embeddings...")
    sa_embeddings = normalize_embedding(sa_embeddings)

    all_indices = sorted(sa_embeddings.keys())
    sample_emb = sa_embeddings[all_indices[0]]
    print(f"  Final embedding dimension: {len(sample_emb)}")
    print(f"  Total episodes with embeddings: {len(all_indices)}")

    # Step 5: Initialize seed set
    print(f"\n[Step 5] Initializing seed set ({args.seed_size} episodes)...")
    seed_set = initialize_seed_set(all_indices, args.seed_size, args.seed)
    print(f"  Seed episodes: {seed_set}")

    # Step 6: Sequential greedy selection
    print(f"\n[Step 6] Running sequential greedy selection...")
    start_time = time.time()

    result = sequential_greedy_selection(
        embeddings=sa_embeddings,
        initial_selected=seed_set,
        target_size=args.num_selected,
        k=args.knn_k,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\nSelection completed in {elapsed:.2f}s")

    # Step 7: Save results
    print(f"\n[Step 7] Saving results...")

    subset_file = output_dir / f"dynamicanchor_v2_{args.num_selected}_seed{args.seed}.json"
    save_subset(result, subset_file)

    # Save full selection log
    log_file = output_dir / f"selection_log_v2_{args.num_selected}_seed{args.seed}.json"
    with open(log_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Selection log saved to: {log_file}")

    # Validation
    selected = result["selected_episode_indices"]
    print(f"\n{'='*60}")
    print(f"Validation")
    print(f"{'='*60}")
    print(f"Selected episodes: {len(selected)}")
    print(f"Unique episodes: {len(set(selected))}")
    print(f"No duplicates: {len(selected) == len(set(selected))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()