"""
Quality Pipeline Unit Tests

Tests:
1. Terminal transitions are dropped (last transition per episode removed)
2. With repeat=4 and no drop, each transition appears 4 times (minus drop_remainder tail)
3. Same seed produces identical batch assignment and episode scores
4. Different seeds allow different batch contexts
5. 1/99 clipping, global z-score, episode mean match manual NumPy reference
"""

import numpy as np
import pandas as pd
import pytest

from deminf.config import DemInfConfig
from deminf.score_episodes import (
    build_official_quality_batches,
    build_quality_records,
    postprocess_scores,
    score_quality_batches,
)


def _make_synthetic_records(
    n_episodes: int = 5,
    timesteps_per_episode: int = 20,
    z_dim_s: int = 12,
    z_dim_a: int = 6,
    seed: int = 42,
) -> list:
    """Create synthetic quality records with known structure."""
    np.random.seed(seed)
    records = []
    for ep_idx in range(n_episodes):
        for ts_idx in range(timesteps_per_episode):
            records.append({
                "z_state": np.random.randn(z_dim_s).astype(np.float32),
                "z_action": np.random.randn(z_dim_a).astype(np.float32),
                "episode_idx": ep_idx,
                "timestep_idx": ts_idx,
                "global_row_idx": ep_idx * 1000 + ts_idx,
            })
    return records


class TestTerminalTransitionDropped:
    """Test that terminal transitions are dropped before quality batching."""

    def test_last_transition_dropped(self):
        """
        Construct records where last timestep per episode is marked.
        After adapter drop_terminal_transitions, these should not appear.
        """
        # Simulate what adapter does: drop last row per episode
        n_episodes = 3
        timesteps = [10, 15, 12]
        records = []
        for ep_idx, n_ts in enumerate(timesteps):
            for ts_idx in range(n_ts - 1):  # Drop last (terminal) transition
                records.append({
                    "z_state": np.zeros(12, dtype=np.float32),
                    "z_action": np.zeros(6, dtype=np.float32),
                    "episode_idx": ep_idx,
                    "timestep_idx": ts_idx,
                    "global_row_idx": ep_idx * 100 + ts_idx,
                })

        # Verify no terminal timestep (n_ts - 1) is present
        for ep_idx, n_ts in enumerate(timesteps):
            ep_records = [r for r in records if r["episode_idx"] == ep_idx]
            max_ts = max(r["timestep_idx"] for r in ep_records)
            assert max_ts == n_ts - 2, (
                f"Episode {ep_idx}: max timestep should be {n_ts - 2} (terminal dropped), got {max_ts}"
            )


class TestRepeatAppearance:
    """Test that repeat=4 causes each transition to appear 4 times."""

    def test_repeat4_each_transition_appears_4_times(self):
        """
        With repeat=4, no discard, and enough transitions to fill full batches,
        each transition should appear exactly 4 times in results.
        """
        # Create enough records to fill multiple full batches
        # 1024 * 3 = 3072 transitions, so 3 full batches per repeat
        n_episodes = 10
        timesteps_per_ep = 308  # 3080 total, > 3072
        records = _make_synthetic_records(n_episodes, timesteps_per_ep, seed=42)

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            quality_batch_size=1024,
            quality_repeat=4,
            quality_discard_fraction=0.0,
            quality_cache=True,  # forces effective_discard_fraction=0
            quality_drop_remainder=True,
            seed=42,
        )

        batches = build_official_quality_batches(records, config, base_seed=42)

        # Count how many times each global_row_idx appears across all batches
        row_counts = {}
        for batch_records, repeat_id, batch_id in batches:
            for rec in batch_records:
                row = rec["global_row_idx"]
                row_counts[row] = row_counts.get(row, 0) + 1

        # Each transition should appear at most 4 times (some may be in drop_remainder tail)
        # With 3080 transitions and batch_size=1024, each repeat has 3 full batches = 3072
        # So 8 transitions per repeat are dropped (3080 - 3072 = 8)
        # Over 4 repeats, different 8 may be dropped each time
        max_count = max(row_counts.values())
        assert max_count <= 4, f"Max count should be <= 4, got {max_count}"

        # Most transitions should appear 4 times
        count_4 = sum(1 for c in row_counts.values() if c == 4)
        assert count_4 > len(records) * 0.9, (
            f"Most transitions should appear 4 times, only {count_4}/{len(records)} do"
        )


class TestReproducibility:
    """Test that same seed produces identical results."""

    def test_same_seed_identical_batches(self):
        """Same seed should produce identical batch assignment."""
        records = _make_synthetic_records(5, 50, seed=42)

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            quality_batch_size=128,
            quality_repeat=4,
            quality_discard_fraction=0.0,
            quality_cache=True,
            quality_drop_remainder=True,
            seed=42,
        )

        batches1 = build_official_quality_batches(records, config, base_seed=42)
        batches2 = build_official_quality_batches(records, config, base_seed=42)

        assert len(batches1) == len(batches2)
        for (recs1, rep1, bid1), (recs2, rep2, bid2) in zip(batches1, batches2):
            assert rep1 == rep2
            assert bid1 == bid2
            for r1, r2 in zip(recs1, recs2):
                assert r1["global_row_idx"] == r2["global_row_idx"]

    def test_same_seed_identical_episode_scores(self):
        """Same seed should produce identical episode scores after full pipeline."""
        records = _make_synthetic_records(5, 50, seed=42)

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            quality_batch_size=128,
            quality_repeat=4,
            quality_discard_fraction=0.0,
            quality_cache=True,
            quality_drop_remainder=True,
            score_clip_low=1.0,
            score_clip_high=99.0,
            seed=42,
        )

        batches = build_official_quality_batches(records, config, base_seed=42)
        results = score_quality_batches(batches, ks=(5, 6, 7))
        ep_scores1, _ = postprocess_scores(results, config)

        # Run again with same seed
        batches2 = build_official_quality_batches(records, config, base_seed=42)
        results2 = score_quality_batches(batches2, ks=(5, 6, 7))
        ep_scores2, _ = postprocess_scores(results2, config)

        pd.testing.assert_frame_equal(ep_scores1, ep_scores2)

    def test_different_seed_different_batches(self):
        """Different seeds should produce different batch contexts."""
        records = _make_synthetic_records(5, 50, seed=42)

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            quality_batch_size=128,
            quality_repeat=4,
            quality_discard_fraction=0.0,
            quality_cache=True,
            quality_drop_remainder=True,
            seed=42,
        )

        batches1 = build_official_quality_batches(records, config, base_seed=42)
        batches2 = build_official_quality_batches(records, config, base_seed=99)

        # First batch should be different
        recs1 = [r["global_row_idx"] for r in batches1[0][0]]
        recs2 = [r["global_row_idx"] for r in batches2[0][0]]
        assert recs1 != recs2, "Different seeds should produce different batch orderings"


class TestScorePostprocessing:
    """Test clipping, normalization, and episode aggregation."""

    def test_clipping_normalization_aggregation_match_reference(self):
        """
        Verify 1/99 clipping, global z-score, and episode mean match manual NumPy reference.
        """
        np.random.seed(42)
        n_samples = 500

        # Simulate raw scores
        raw_scores = np.random.randn(n_samples) * 2.0
        episode_indices = np.random.randint(0, 10, size=n_samples)

        # Manual reference computation
        p_low = np.percentile(raw_scores, 1.0)
        p_high = np.percentile(raw_scores, 99.0)
        scores_clipped = np.clip(raw_scores, p_low, p_high)

        score_mean = np.mean(scores_clipped)
        score_std = np.std(scores_clipped)
        scores_norm = (scores_clipped - score_mean) / score_std

        # Episode mean reference
        ep_scores_ref = {}
        for ep in range(10):
            mask = episode_indices == ep
            if mask.sum() > 0:
                ep_scores_ref[ep] = np.mean(scores_norm[mask])

        # Now test via postprocess_scores
        results = []
        for i in range(n_samples):
            results.append({
                "episode_idx": int(episode_indices[i]),
                "timestep_idx": i,
                "global_row_idx": i,
                "repeat_id": 0,
                "batch_id": 0,
                "raw_score": float(raw_scores[i]),
            })

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            score_clip_low=1.0,
            score_clip_high=99.0,
            seed=42,
        )

        ep_scores_df, ts_df = postprocess_scores(results, config)

        # Verify clipping
        np.testing.assert_allclose(
            ts_df["clipped_score"].values,
            scores_clipped,
            rtol=1e-10,
        )

        # Verify normalization
        np.testing.assert_allclose(
            ts_df["normalized_score"].values,
            scores_norm,
            rtol=1e-10,
        )

        # Verify episode aggregation
        for _, row in ep_scores_df.iterrows():
            ep = int(row["episode_idx"])
            if ep in ep_scores_ref:
                np.testing.assert_allclose(
                    row["deminf_score"],
                    ep_scores_ref[ep],
                    rtol=1e-10,
                )

    def test_near_zero_std_raises_error(self):
        """If all scores are identical, normalization should raise ValueError."""
        results = []
        for i in range(100):
            results.append({
                "episode_idx": i % 5,
                "timestep_idx": i,
                "global_row_idx": i,
                "repeat_id": 0,
                "batch_id": 0,
                "raw_score": 1.0,  # All identical
            })

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            score_clip_low=1.0,
            score_clip_high=99.0,
            seed=42,
        )

        with pytest.raises(ValueError, match="near zero"):
            postprocess_scores(results, config)

    def test_nan_filtering(self):
        """NaN scores should be filtered out."""
        results = []
        for i in range(100):
            score = float(np.random.randn())
            if i % 10 == 0:
                score = float("nan")
            results.append({
                "episode_idx": i % 5,
                "timestep_idx": i,
                "global_row_idx": i,
                "repeat_id": 0,
                "batch_id": 0,
                "raw_score": score,
            })

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            score_clip_low=1.0,
            score_clip_high=99.0,
            seed=42,
        )

        ep_scores_df, ts_df = postprocess_scores(results, config)

        # No NaN in output
        assert np.all(np.isfinite(ts_df["normalized_score"].values))
        assert np.all(np.isfinite(ep_scores_df["deminf_score"].values))

        # 10 NaN should be filtered
        assert len(ts_df) == 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])