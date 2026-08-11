"""
Compare different pooling strategies for episode embeddings.
Goal: Find which pooling gives the best d_bar (B0 separation in embedding space).

Uses pre-extracted frame embeddings from all_frames/ directory.
"""
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from selection_strategies import B0Initializer
from extract_embeddings import build_episode_features, load_episode_embeddings

def analyze_pooling(features, n_components=32, b0_size=9):
    """Analyze one pooling method."""
    # PCA
    pca = PCA(n_components=n_components)
    pca_emb = pca.fit_transform(features)
    
    # Pairwise distances
    N = len(pca_emb)
    dist_matrix = squareform(pdist(pca_emb, metric='euclidean'))
    upper_tri = dist_matrix[np.triu_indices(N, k=1)]
    
    # B0: FPS in PCA space
    b0_indices = B0Initializer.fps(pca_emb, n=b0_size)
    
    # d_bar: median distance among B0 points
    b0_dist = dist_matrix[np.ix_(b0_indices, b0_indices)]
    b0_upper = b0_dist[np.triu_indices(b0_size, k=1)]
    d_bar = np.median(b0_upper)
    
    # Other metrics
    global_mean = upper_tri.mean()
    global_std = upper_tri.std()
    cv = global_std / global_mean  # coefficient of variation
    
    # Nearest neighbor distances
    np.fill_diagonal(dist_matrix, np.inf)
    nn_dists = dist_matrix.min(axis=1)
    
    return {
        'd_bar': d_bar,
        'global_mean': global_mean,
        'global_std': global_std,
        'cv': cv,
        'nn_mean': nn_dists.mean(),
        'nn_min': nn_dists.min(),
        'pca_explained': pca.explained_variance_ratio_.sum(),
        'b0_indices': b0_indices.tolist(),
        'output_dim': features.shape[-1],
    }

def main(num_keyframes=5, n_components=32, num_episodes=50):
    """Main analysis."""
    # Load pool metadata
    pool_dir = Path(__file__).parent.parent / "pool"
    import pandas as pd
    df = pd.read_csv(pool_dir / "episode_metadata.csv")
    episode_indices = df["episode_index"].tolist()[:num_episodes]
    
    print(f"Using {len(episode_indices)} episodes for testing")
    
    methods = ['max', 'mean', 'concat']
    results = []
    
    print(f"\n{'='*70}")
    print(f"Comparing pooling methods (k={num_keyframes}, episodes={len(episode_indices)})")
    print(f"{'='*70}\n")
    
    for method in methods:
        print(f"Testing: {method}...")
        features = build_episode_features(episode_indices, num_keyframes, method)
        print(f"  Features shape: {features.shape}")
        
        result = analyze_pooling(features, n_components, b0_size=9)
        result['method'] = method
        results.append(result)
        
        print(f"  d_bar: {result['d_bar']:.4f}")
        print(f"  Global dist: mean={result['global_mean']:.4f}, std={result['global_std']:.4f}")
        print(f"  CV (diversity): {result['cv']:.4f}")
        print(f"  NN dist: mean={result['nn_mean']:.4f}, min={result['nn_min']:.4f}")
        print(f"  PCA explained: {result['pca_explained']:.4f}")
        print(f"  B0 indices: {result['b0_indices']}")
        print()
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'d_bar':>10} {'CV':>10} {'NN_mean':>10} {'NN_min':>10} {'Dim':>6}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['method']:<12} {r['d_bar']:>10.4f} {r['cv']:>10.4f} {r['nn_mean']:>10.4f} {r['nn_min']:>10.4f} {r['output_dim']:>6}")
    
    # Find best method
    best = max(results, key=lambda x: x['d_bar'])
    print(f"\nBest method by d_bar: {best['method']} (d_bar={best['d_bar']:.4f})")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to test (default: 50)")
    args = parser.parse_args()
    
    main(num_keyframes=args.k, n_components=args.pca_dim, num_episodes=args.num_episodes)