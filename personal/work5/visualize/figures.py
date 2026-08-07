"""All figure generation functions with English labels."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150
})
COLORS = sns.color_palette("colorblind")


def save_figure(fig, save_path, dpi_preview=150, dpi_high=300):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path + ".png", dpi=dpi_preview, bbox_inches='tight')
    fig.savefig(save_path + "_high.png", dpi=dpi_high, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}.png")


def fig1_dataset_coverage(pca_results, save_path):
    """2D scatter of PCA embeddings for all 4 datasets."""
    n_datasets = len(pca_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(pca_results.items()):
        ax = axes[idx]
        embeddings = data['embeddings']
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(len(embeddings), -1)

        if len(embeddings) > 2:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                               c=range(len(reduced)), cmap='viridis', alpha=0.7, s=30)
            ax.set_title(f"{name}\n(PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, "
                        f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)")
        else:
            ax.scatter([0], [0], c='red', s=50)
            ax.set_title(name)

        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.grid(True, alpha=0.3)

    plt.suptitle("PCA Embedding Coverage Comparison (dim 1 vs dim 2)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, save_path)


def fig2_sic_scores(scores_dict, save_path):
    """Bar chart: SIC score for each of 4 dataset configurations."""
    strategies = list(scores_dict.keys())
    scores = [scores_dict[s] for s in strategies]

    colors = []
    for s in strategies:
        if 'SIC' in s or 'Ours' in s:
            colors.append(COLORS[2])
        else:
            colors.append(COLORS[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(strategies))
    bars = ax.bar(x, scores, color=colors, edgecolor='black', linewidth=0.7, alpha=0.85)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("Collection Strategy")
    ax.set_ylabel("SIC Score")
    ax.set_title("State Information Coverage (SIC) Score by Collection Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in strategies], rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, save_path)


def fig3_training_curves(results_dict, save_path):
    """Line plot: eval success rate vs training step for all 4 methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in results_dict.items():
        steps = data.get('steps', [])
        success_rates = data.get('success_rates', [])
        stds = data.get('stds', [0] * len(success_rates))

        color_idx = list(results_dict.keys()).index(name) % len(COLORS)
        ax.plot(steps, success_rates, marker='o', linewidth=2,
                color=COLORS[color_idx], label=name, markersize=6)
        if stds and len(stds) == len(success_rates):
            ax.fill_between(steps,
                          [max(0, s - e) for s, e in zip(success_rates, stds)],
                          [min(100, s + e) for s, e in zip(success_rates, stds)],
                          alpha=0.15, color=COLORS[color_idx])

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Policy Success Rate During Training (MetaWorld Pick-Place)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    ax.set_xlim([0, 8500])

    plt.tight_layout()
    save_figure(fig, save_path)


def fig4_data_efficiency(efficiency_data, save_path):
    """X-axis: Number of demonstrations, Y-axis: Final success rate."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy, data in efficiency_data.items():
        demos = data['demos']
        srs = data['success_rates']
        color_idx = list(efficiency_data.keys()).index(strategy) % len(COLORS)
        ax.plot(demos, srs, marker='o', linewidth=2, color=COLORS[color_idx],
                label=strategy, markersize=7)

    ax.set_xlabel("Number of Demonstrations")
    ax.set_ylabel("Final Success Rate (%)")
    ax.set_title("Data Efficiency: Success Rate vs. Demonstration Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    save_figure(fig, save_path)


def fig5_attention_evolution(attention_data, save_path):
    """Grid: rows=training checkpoints, columns=3 test images."""
    checkpoints = attention_data.get('checkpoints', [])
    n_images = len(attention_data.get('test_images', [1, 2, 3]))
    n_checkpoints = len(checkpoints)

    if n_checkpoints == 0:
        return

    fig, axes = plt.subplots(n_checkpoints, n_images * 2,
                            figsize=(4 * n_images * 2, 3 * n_checkpoints))
    if n_checkpoints == 1:
        axes = [axes]

    for cp_idx, cp_data in enumerate(checkpoints):
        for img_idx in range(n_images):
            base_col = img_idx * 2
            if cp_idx < len(axes) and base_col < len(axes[cp_idx]):
                ax_orig = axes[cp_idx][base_col]
                ax_attn = axes[cp_idx][base_col + 1] if base_col + 1 < len(axes[cp_idx]) else None

                if 'original' in cp_data and img_idx < len(cp_data['original']):
                    ax_orig.imshow(cp_data['original'][img_idx])
                    ax_orig.set_title(f"Step {cp_data.get('step', '?')}: Original")
                    ax_orig.axis('off')

                if ax_attn and 'attention' in cp_data and img_idx < len(cp_data['attention']):
                    ax_attn.imshow(cp_data['attention'][img_idx])
                    ax_attn.set_title(f"Step {cp_data.get('step', '?')}: Attention")
                    ax_attn.axis('off')

    plt.suptitle("Attention Map Evolution During Training (SIC-Noise vs Uniform)")
    plt.tight_layout()
    save_figure(fig, save_path)


def fig6_noise_analysis(b0_positions, noise_positions, sic_heatmap, save_path):
    """Show original B0 demo positions and noise-augmented positions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if b0_positions:
        b0_x = [p[0][0] for p in b0_positions]
        b0_y = [p[0][1] for p in b0_positions]
        ax.scatter(b0_x, b0_y, c='blue', s=50, alpha=0.7, label='B0 Demos', zorder=5)

    if noise_positions:
        noise_x = [p[0][0] for p in noise_positions]
        noise_y = [p[0][1] for p in noise_positions]
        ax.scatter(noise_x, noise_y, c='orange', s=40, alpha=0.7,
                  label='Noise-Augmented', zorder=4)

    if sic_heatmap is not None:
        cax = ax.imshow(sic_heatmap, cmap='RdYlBu_r', alpha=0.3,
                       extent=[-0.15, 0.15, 0.55, 0.75], aspect='auto')
        plt.colorbar(cax, ax=ax, label='SIC Support (low=red, high=blue)')

    ax.set_xlabel("Object X Position")
    ax.set_ylabel("Object Y Position")
    ax.set_title("SIC-Guided Noise Augmentation: Targeting Under-Covered Regions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, save_path)


def fig7_embedding_tsne(embeddings_dict, save_path):
    """t-SNE of all demos from all 4 datasets."""
    all_embeddings = []
    all_labels = []

    for name, embeddings in embeddings_dict.items():
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(len(embeddings), -1)
        if len(embeddings) > 0:
            all_embeddings.append(embeddings)
            all_labels.extend([name] * len(embeddings))

    if len(all_embeddings) == 0:
        return

    combined = np.vstack(all_embeddings)

    if len(combined) > 2:
        perplexity = min(30, max(1, len(combined) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 7))
        unique_labels = list(set(all_labels))
        for label in unique_labels:
            mask = np.array(all_labels) == label
            color_idx = unique_labels.index(label) % len(COLORS)
            ax.scatter(reduced[mask, 0], reduced[mask, 1],
                      c=[COLORS[color_idx]], label=label, alpha=0.7, s=30)

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title("t-SNE Visualization of Demo Embeddings by Collection Strategy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, save_path)