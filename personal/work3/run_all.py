#!/usr/bin/env python3
"""
SIC Framework for MetaWorld — One-Click Runner
Run all analyses and generate all paper figures.

Usage:
  # Full run (requires dataset and success rates)
  python personal/work3/run_all.py \
    --dataset_root /path/to/lerobot_datasets \
    --config_map personal/work3/data/config_map.json \
    --success_rates personal/work3/data/success_rates.json

  # Analysis only (no success rates, skips comparison figures)
  python personal/work3/run_all.py \
    --dataset_root /path/to/lerobot_datasets \
    --config_map personal/work3/data/config_map.json \
    --mode analysis_only

  # Figures only (all data pre-computed, just regenerate figures)
  python personal/work3/run_all.py \
    --mode figures_only
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Set HF_LEROBOT_HOME to project-local data directory
_project_root = Path(__file__).parent.parent.parent
os.environ["HF_LEROBOT_HOME"] = str(_project_root / "personal" / "work3" / "data")

sys.path.insert(0, '.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--config_map', type=str,
                        default='personal/work3/data/config_map.json')
    parser.add_argument('--success_rates', type=str, default='')
    parser.add_argument('--model_paths', type=str, default='',
                        help='JSON file: {dataset_name: checkpoint_path}')
    parser.add_argument('--mode', choices=['full', 'analysis_only', 'figures_only'],
                        default='full')
    parser.add_argument('--budget_B', type=int, default=144)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip embedding extraction if cached')
    return parser.parse_args()


def main():
    args = parse_args()

    from configs import CFG
    CFG.dataset_root = args.dataset_root
    CFG.device = args.device
    CFG.budget_B = args.budget_B
    CFG.setup()

    print("=" * 60)
    print("SIC Framework (MetaWorld) — Full Analysis Pipeline")
    print("=" * 60)

    # ============================================================
    # PHASE 1: Embedding Extraction & Anchor Construction
    # ============================================================

    if args.mode == 'figures_only':
        print("\n[Mode: figures_only] Loading pre-computed data...")
        anchor_ref = pickle.load(open(os.path.join(CFG.results_dir, 'anchor_ref.pkl'), 'rb'))
        global_embs = pickle.load(open(os.path.join(CFG.cache_dir, 'global_embs.pkl'), 'rb'))
        wrist_embs = pickle.load(open(os.path.join(CFG.cache_dir, 'wrist_embs.pkl'), 'rb'))
        with open(args.config_map) as f:
            config_map_raw = json.load(f)
        config_map = {int(k): tuple(v) for k, v in config_map_raw.items()}

    else:
        print("\n[Phase 1] Loading dataset and extracting embeddings...")

        if not args.dataset_root:
            print("ERROR: --dataset_root required for modes other than 'figures_only'")
            sys.exit(1)

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from sic.embeddings import load_frozen_vlm, extract_and_cache_all_embeddings
        from sic.anchor import build_anchor_reference

        with open(args.config_map) as f:
            config_map_raw = json.load(f)
        config_map = {int(k): tuple(v) for k, v in config_map_raw.items()}

        dataset = LeRobotDataset(args.dataset_root)

        model, processor = load_frozen_vlm(CFG.vlm_model_id, CFG.device)

        global_embs = extract_and_cache_all_embeddings(
            model, processor, dataset, config_map,
            CFG.global_cam_key, CFG.device,
            os.path.join(CFG.cache_dir, 'global_embs.pkl'),
            CFG.batch_size
        )
        wrist_embs = extract_and_cache_all_embeddings(
            model, processor, dataset, config_map,
            CFG.wrist_cam_key, CFG.device,
            os.path.join(CFG.cache_dir, 'wrist_embs.pkl'),
            CFG.batch_size
        )

        anchor_ref = build_anchor_reference(
            global_embs, wrist_embs, config_map,
            d_pca=CFG.d_pca,
            save_path=os.path.join(CFG.results_dir, 'anchor_ref.pkl')
        )

    # ============================================================
    # PHASE 2: Research Analyses (no training required)
    # ============================================================

    print("\n[Phase 2] Research analyses (embedding space)...")

    from research.embedding_analysis import (
        compute_cross_config_distance_curve,
        compute_pca_variance_curve,
        compute_tsne_for_configs
    )

    distance_curve = compute_cross_config_distance_curve(wrist_embs, config_map)
    np.save(os.path.join(CFG.results_dir, 'cross_config_distance.npy'), distance_curve)

    global_variance = compute_pca_variance_curve(global_embs, max_components=128)
    wrist_variance = compute_pca_variance_curve(wrist_embs, max_components=128)

    tsne_result = compute_tsne_for_configs(wrist_embs, config_map)

    # ============================================================
    # PHASE 3: SIC Computation
    # ============================================================

    print("\n[Phase 3] Computing SIC scores...")

    from sic.score import compute_sic
    from sic.greedy import greedy_plan

    b0_plan = {cfg: 1 for cfg in anchor_ref['anchors'].keys()}
    b0_sic = compute_sic(b0_plan, anchor_ref, CFG.alpha, CFG.lambda_weight)
    print(f"  B0 SIC: {b0_sic['sic']:.4f}")

    print(f"  Running greedy planning (budget={CFG.budget_B})...")
    greedy_result = greedy_plan(anchor_ref, CFG.budget_B, CFG.t_max,
                                CFG.alpha, CFG.lambda_weight)

    plan = greedy_result['final_plan']
    all_pos = sorted(set(k[0] for k in plan.keys()))
    all_rot = sorted(set(k[1] for k in plan.keys()))
    plan_matrix = np.zeros((len(all_pos), len(all_rot)), dtype=int)
    for (p, r), n in plan.items():
        plan_matrix[all_pos.index(p), all_rot.index(r)] = n

    rot_labels = [f"{r*45}°" for r in all_rot]
    plan_df = pd.DataFrame(plan_matrix,
                           index=[f"Pos {p}" for p in all_pos],
                           columns=rot_labels)

    print(f"  Greedy plan: {sum(plan.values())} total demos, SIC={greedy_result['sic_history'][-1]:.4f}")

    # ============================================================
    # PHASE 4: Correlation Analysis (loads success rates if available)
    # ============================================================

    print("\n[Phase 4] Correlation analysis...")

    success_rates = None
    if args.success_rates and os.path.exists(args.success_rates):
        with open(args.success_rates) as f:
            success_rates = json.load(f)
        print(f"  Loaded success rates for {len(success_rates)} subsets")
    else:
        print("  [INFO] No success rates found. Correlation figures will be skipped.")
        print("  To add results: create personal/work3/data/success_rates.json")
        print("  Format: {subset_name: {success_rate: float, std: float, n_demos: int, sic_score: float}}")

    spearman_result = None
    df_correlation = None

    if success_rates:
        from scipy.stats import spearmanr

        df_correlation = pd.DataFrame([
            {
                'subset_name': name,
                'sic_score': data['sic_score'],
                'success_rate': data['success_rate'],
                'std': data.get('std', 0),
                'n_demos': data['n_demos']
            }
            for name, data in success_rates.items()
        ])

        rho, p_val = spearmanr(df_correlation['sic_score'], df_correlation['success_rate'])
        spearman_result = {'rho': rho, 'p_value': p_val}

        print(f"  Spearman rho = {rho:.4f}, p = {p_val:.6f}")

    # ============================================================
    # PHASE 5: Attention Map Analysis (if model paths provided)
    # ============================================================

    model_comparison_result = None
    if args.model_paths and os.path.exists(args.model_paths):
        print("\n[Phase 5] Attention map analysis...")
        from research.attention_analysis import compare_attention_across_datasets
        from sic.embeddings import load_frozen_vlm

        with open(args.model_paths) as f:
            model_paths = json.load(f)

        _, processor = load_frozen_vlm(CFG.vlm_model_id, CFG.device)

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from PIL import Image
        dataset = LeRobotDataset(args.dataset_root)

        test_images = []
        for ep_idx in list(config_map.keys())[:3]:
            frame = dataset[dataset.episode_data_index['from'][ep_idx]]
            img_t = frame[CFG.global_cam_key]
            img_pil = Image.fromarray(
                (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            test_images.append((img_pil, "Grasp the object and lift it out"))

        model_comparison_result = compare_attention_across_datasets(
            model_paths, test_images, processor, CFG.device, CFG.figures_dir
        )
    else:
        print("\n[Phase 5] Attention analysis skipped (no model paths provided)")

    # ============================================================
    # PHASE 6: Generate All Figures
    # ============================================================

    print("\n[Phase 6] Generating all figures...")

    from visualize.all_figures import (
        fig1_tsne_embeddings,
        fig2_cross_config_distance,
        fig3_pca_variance,
        fig4_sic_correlation,
        fig5_greedy_sic_curve,
        fig6_collection_heatmap,
        fig7_baseline_comparison,
        fig8_efficiency_curve,
        fig9_ablation_components
    )

    print("\n  Generating fig1: t-SNE embeddings...")
    fig1_tsne_embeddings(tsne_result, CFG.figures_dir)

    print("  Generating fig2: cross-config distance curve...")
    fig2_cross_config_distance(distance_curve, CFG.figures_dir)

    print("  Generating fig3: PCA variance curves...")
    fig3_pca_variance(global_variance, wrist_variance, CFG.figures_dir)

    if df_correlation is not None and spearman_result is not None:
        print("  Generating fig4: SIC correlation scatter...")
        fig4_sic_correlation(df_correlation, spearman_result, CFG.figures_dir)
    else:
        print("  [SKIP] fig4: waiting for success rate data")

    print("  Generating fig5: greedy SIC curve...")
    fig5_greedy_sic_curve(greedy_result, CFG.budget_B, len(b0_plan), CFG.figures_dir)

    print("  Generating fig6: collection heatmap...")
    fig6_collection_heatmap(plan_df, CFG.figures_dir)

    if success_rates:
        comparison_data = {
            name: {'success_rate': d['success_rate'],
                   'std': d.get('std', 0),
                   'n_demos': d['n_demos']}
            for name, d in success_rates.items()
        }
        print("  Generating fig7: baseline comparison...")
        fig7_baseline_comparison(comparison_data, CFG.figures_dir)
    else:
        print("  [SKIP] fig7: waiting for success rate data")

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 60)
    print(f"\nFigures saved to: {CFG.figures_dir}/")
    print("\nAvailable figures:")
    for f in sorted(os.listdir(CFG.figures_dir)):
        if f.endswith('.pdf'):
            print(f"  {f}")

    print("\nKey results:")
    print(f"  B0 SIC score: {b0_sic['sic']:.4f}")
    print(f"  Greedy plan ({CFG.budget_B} demos) SIC: {greedy_result['sic_history'][-1]:.4f}")

    if spearman_result:
        print(f"  Spearman correlation: rho={spearman_result['rho']:.4f}, p={spearman_result['p_value']:.6f}")

    print(f"\nStopping criterion: step {greedy_result.get('stopping_step', 'N/A')}")
    print("\nCollect plan printout:")
    for (pos, rot), n in sorted(plan.items(), key=lambda x: -x[1]):
        if n > 1:
            print(f"  Position {pos}, Rotation {rot*45}°: {n} collections (add {n-1} to B0)")


if __name__ == '__main__':
    main()