"""
Dataset Adapter Unit Tests

Tests:
1. Global row indices are correct (not parquet-local indices)
2. Episode index validation passes
3. Terminal transitions are dropped correctly
4. State dimension is 39 (not 43)
5. Cache fingerprint changes when key fields change
"""

import hashlib
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from deminf.dataset_adapter import (
    drop_terminal_transitions,
    DemInfNormalizer,
    _array_hash,
)
from deminf.config import DemInfConfig


class TestDropTerminalTransitions:
    """Test that terminal transitions are correctly dropped."""

    def test_drop_last_row_per_episode(self):
        """Each episode should lose its last row."""
        episode_map = {
            0: [10, 11, 12, 13, 14],
            1: [20, 21, 22],
            2: [30, 31, 32, 33],
        }

        result = drop_terminal_transitions(episode_map)

        assert result[0] == [10, 11, 12, 13]
        assert result[1] == [20, 21]
        assert result[2] == [30, 31, 32]

    def test_single_row_episode_not_dropped(self):
        """Episode with only 1 row should not be dropped (with warning)."""
        episode_map = {
            0: [10],
            1: [20, 21, 22],
        }

        result = drop_terminal_transitions(episode_map)

        assert result[0] == [10]  # Cannot drop
        assert result[1] == [20, 21]


class TestDemInfNormalizer:
    """Test DemInf normalization."""

    def test_gripper_not_normalized(self):
        """Gripper dims (3 and 21) should not be normalized."""
        np.random.seed(42)
        states = np.random.randn(100, 39).astype(np.float32)
        actions = np.random.randn(100, 4).astype(np.float32)

        normalizer = DemInfNormalizer(state_dim=39, action_dim=4)
        normalizer.fit(states, actions)

        norm_states = normalizer.normalize_state(states)

        # Gripper dims should be unchanged
        np.testing.assert_allclose(
            norm_states[:, 3], states[:, 3], rtol=1e-6
        )
        np.testing.assert_allclose(
            norm_states[:, 21], states[:, 21], rtol=1e-6
        )

        # Non-gripper dims should be normalized
        non_gripper = np.ones(39, dtype=bool)
        non_gripper[[3, 21]] = False
        normalized_means = np.mean(norm_states[:, non_gripper], axis=0)
        normalized_stds = np.std(norm_states[:, non_gripper], axis=0)
        np.testing.assert_allclose(normalized_means, 0.0, atol=1e-5)
        np.testing.assert_allclose(normalized_stds, 1.0, atol=1e-5)

    def test_action_gripper_bounds_normalization(self):
        """Action gripper should use bounds normalization if not in [-1,1]."""
        np.random.seed(42)
        states = np.random.randn(100, 39).astype(np.float32)
        # Action gripper in [0, 255] (not normalized)
        actions = np.random.randn(100, 4).astype(np.float32)
        actions[:, 3] = np.random.uniform(0, 255, size=100).astype(np.float32)

        normalizer = DemInfNormalizer(state_dim=39, action_dim=4)
        normalizer.fit(states, actions)
        norm_actions = normalizer.normalize_action(actions)

        # Gripper should be in [-1, 1]
        assert norm_actions[:, 3].min() >= -1.0 - 1e-6
        assert norm_actions[:, 3].max() <= 1.0 + 1e-6

    def test_action_gripper_already_normalized(self):
        """Action gripper already in [-1,1] should not be changed."""
        np.random.seed(42)
        states = np.random.randn(100, 39).astype(np.float32)
        actions = np.random.randn(100, 4).astype(np.float32)
        actions[:, 3] = np.random.uniform(-0.9, 0.9, size=100).astype(np.float32)

        normalizer = DemInfNormalizer(state_dim=39, action_dim=4)
        normalizer.fit(states, actions)

        assert normalizer.action_gripper_normalized is True

        norm_actions = normalizer.normalize_action(actions)
        np.testing.assert_allclose(
            norm_actions[:, 3], actions[:, 3], rtol=1e-6
        )

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve all normalization stats."""
        np.random.seed(42)
        states = np.random.randn(100, 39).astype(np.float32)
        actions = np.random.randn(100, 4).astype(np.float32)

        normalizer = DemInfNormalizer(state_dim=39, action_dim=4)
        normalizer.fit(states, actions)

        save_path = tmp_path / "norm_stats.npz"
        normalizer.save(str(save_path))

        loaded = DemInfNormalizer.load(str(save_path))

        np.testing.assert_allclose(loaded.state_mean, normalizer.state_mean)
        np.testing.assert_allclose(loaded.state_std, normalizer.state_std)
        np.testing.assert_allclose(loaded.action_mean, normalizer.action_mean)
        np.testing.assert_allclose(loaded.action_std, normalizer.action_std)
        assert loaded.state_gripper_indices == normalizer.state_gripper_indices
        assert loaded.action_gripper_normalized == normalizer.action_gripper_normalized


class TestCacheFingerprint:
    """Test cache fingerprint changes when key fields change."""

    def test_different_latent_dim_changes_fingerprint(self):
        """Changing latent_dim should change fingerprint."""
        config1 = DemInfConfig(
            dataset_path="/tmp/data",
            output_dir="/tmp/out",
            state_latent_dim=12,
            action_latent_dim=6,
        )
        config2 = DemInfConfig(
            dataset_path="/tmp/data",
            output_dir="/tmp/out",
            state_latent_dim=16,
            action_latent_dim=6,
        )

        assert config1.config_fingerprint() != config2.config_fingerprint()

    def test_different_state_source_changes_fingerprint(self):
        """Changing state_source should change fingerprint."""
        config1 = DemInfConfig(
            dataset_path="/tmp/data",
            output_dir="/tmp/out",
            state_source="observation.environment_state",
        )
        config2 = DemInfConfig(
            dataset_path="/tmp/data",
            output_dir="/tmp/out",
            state_source="observation.state",
        )

        assert config1.config_fingerprint() != config2.config_fingerprint()

    def test_same_config_same_fingerprint(self):
        """Same config should produce same fingerprint."""
        config1 = DemInfConfig(
            dataset_path="/tmp/data",
            output_dir="/tmp/out",
            state_latent_dim=12,
            action_latent_dim=6,
            vae_steps=50000,
            vae_lr=1e-4,
        )
        config2 = DemInfConfig(
            dataset_path="/tmp/data",
            output_dir="/tmp/out",
            state_latent_dim=12,
            action_latent_dim=6,
            vae_steps=50000,
            vae_lr=1e-4,
        )

        assert config1.config_fingerprint() == config2.config_fingerprint()


class TestArrayHash:
    """Test array hash function."""

    def test_same_array_same_hash(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _array_hash(arr) == _array_hash(arr)

    def test_different_array_different_hash(self):
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        assert _array_hash(arr1) != _array_hash(arr2)


class TestEffectiveDiscardFraction:
    """Test effective_discard_fraction logic."""

    def test_cache_true_forces_zero(self):
        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            quality_cache=True,
            quality_discard_fraction=0.5,
        )
        assert config.effective_discard_fraction() == 0.0

    def test_cache_false_uses_requested(self):
        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir="/tmp",
            quality_cache=False,
            quality_discard_fraction=0.5,
        )
        assert config.effective_discard_fraction() == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])