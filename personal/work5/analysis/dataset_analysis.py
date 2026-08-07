"""Dataset analysis figures."""

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
})
COLORS = sns.color_palette("colorblind")


def plot_pca_coverage(embeddings_dict, save_path, title="PCA Embedding Coverage"):
    fig, axes = plt.subplots(1, len(embeddings_dict), figsize=(6 * len(embeddings_dict), 5))
    if len(embeddings_dict) == 1:
        axes = [axes]

    for idx, (name, embeddings) in enumerate(embeddings_dict.items()):
        ax = axes[idx]
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
            ax.set_title(f"{name} (too few points)")

        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(save_path + "_high.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.png")


def plot_sic_scores(scores_dict, save_path):
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
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(save_path + "_high.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.png")


def plot_training_curves(results_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in results_dict.items():
        steps = data.get('steps', [])
        success_rates = data.get('success_rates', [])
        stds = data.get('stds', [0] * len(success_rates))

        color = COLORS[list(results_dict.keys()).index(name) % len(COLORS)]
        ax.plot(steps, success_rates, marker='o', linewidth=2,
                color=color, label=name, markersize=6)
        if stds and len(stds) == len(success_rates):
            ax.fill_between(steps,
                          [s - e for s, e in zip(success_rates, stds)],
                          [s + e for s, e in zip(success_rates, stds)],
                          alpha=0.15, color=color)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Policy Success Rate During Training (MetaWorld Pick-Place)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(save_path + "_high.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.png")


def plot_data_efficiency(efficiency_data, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy, data in efficiency_data.items():
        demos = data['demos']
        srs = data['success_rates']
        color = COLORS[list(efficiency_data.keys()).index(strategy) % len(COLORS)]
        ax.plot(demos, srs, marker='o', linewidth=2, color=color,
                label=strategy, markersize=7)

    ax.set_xlabel("Number of Demonstrations")
    ax.set_ylabel("Final Success Rate (%)")
    ax.set_title("Data Efficiency: Success Rate vs. Demonstration Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(save_path + "_high.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.png")


def plot_attention_evolution(attention_data, save_path):
    n_checkpoints = len(attention_data.get('checkpoints', []))
    n_images = len(attention_data.get('test_images', [1, 2, 3]))

    fig, axes = plt.subplots(n_checkpoints, n_images * 2,
                            figsize=(4 * n_images * 2, 3 * n_checkpoints))

    if n_checkpoints == 1:
        axes = [axes]

    for cp_idx, cp_data in enumerate(attention_data.get('checkpoints', [])):
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
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(save_path + "_high.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.png")


def plot_noise_analysis(b0_positions, noise_positions, sic_heatmap, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))

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
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(save_path + "_high.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.png")


def plot_tsne_embeddings(embeddings_dict, save_path):
    all_embeddings = []
    all_labels = []

    for name, embeddings in embeddings_dict.items():
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(len(embeddings), -1)
        all_embeddings.append(embeddings)
        all_labels.extend([name] * len(embeddings))

    if len(all_embeddings) == 0:
        return

    combined = np.vstack(all_embeddings)

    if len(combined) > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
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
        plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
        plt.savefig(save_path + "_high.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}.png")