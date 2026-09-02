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

        # Bounds normalization is ALWAYS applied (official BOUNDS semantics)
        assert normalizer.action_gripper_bounds_applied is True

        norm_actions = normalizer.normalize_action(actions)
        # Even if values are already in [-1,1], bounds transform is still applied
        # If min=-1, max=1, the result is identical to original
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
        assert loaded.action_gripper_bounds_applied == normalizer.action_gripper_bounds_applied


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


class TestBuildEpisodeIndexGlobalHfRows:
    """Test that build_episode_index_from_lerobot uses global HF dataset rows."""

    def test_build_episode_index_uses_global_hf_rows(self):
        """
        Verify that build_episode_index_from_lerobot builds the mapping
        using global HF dataset row indices, not parquet-local indices.

        Construct a fake hf_dataset where global rows 0..8 belong to:
        - episode0: rows 0,1,2 (frame_index 0,1,2)
        - episode1: rows 3,4,5 (frame_index 0,1,2)
        - episode2: rows 6,7,8 (frame_index 0,1,2)

        The result must be {0:[0,1,2], 1:[3,4,5], 2:[6,7,8]}.
        """
        from deminf.dataset_adapter import build_episode_index_from_lerobot

        # Create fake hf_dataset rows
        fake_rows = []
        for global_idx in range(9):
            ep_idx = global_idx // 3
            frame_idx = global_idx % 3
            fake_rows.append({
                "episode_index": ep_idx,
                "frame_index": frame_idx,
            })

        # Mock LeRobotDataset
        class FakeHFDataset:
            def __init__(self, rows):
                self._rows = rows

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, idx):
                return self._rows[int(idx)]

        class FakeLeRobotDataset:
            def __init__(self, repo_id, root):
                self.hf_dataset = FakeHFDataset(fake_rows)
                self.num_episodes = 3
                self.num_frames = 9

        with patch(
            "deminf.dataset_adapter.LeRobotDataset",
            FakeLeRobotDataset,
        ):
            result = build_episode_index_from_lerobot("/tmp/fake", "fake_repo")

        assert result == {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}, (
            f"Expected global row mapping, got {result}"
        )

    def test_build_episode_index_with_shuffled_physical_order(self):
        """
        Test that even if the HF dataset physical order is shuffled,
        the mapping still uses the correct global row IDs and sorts
        by frame_index within each episode.

        Construct a shuffled case:
        - Physical row 0: episode1, frame_index=1 (global row 4)
        - Physical row 1: episode0, frame_index=0 (global row 0)
        - Physical row 2: episode2, frame_index=2 (global row 8)
        - Physical row 3: episode0, frame_index=1 (global row 1)
        - Physical row 4: episode1, frame_index=0 (global row 3)
        - Physical row 5: episode2, frame_index=0 (global row 6)
        - Physical row 6: episode0, frame_index=2 (global row 2)
        - Physical row 7: episode1, frame_index=2 (global row 5)
        - Physical row 8: episode2, frame_index=1 (global row 7)

        Expected result (sorted by frame_index within each episode):
        {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        """
        from deminf.dataset_adapter import build_episode_index_from_lerobot

        shuffled_rows = [
            {"episode_index": 1, "frame_index": 1},  # global 4
            {"episode_index": 0, "frame_index": 0},  # global 0
            {"episode_index": 2, "frame_index": 2},  # global 8
            {"episode_index": 0, "frame_index": 1},  # global 1
            {"episode_index": 1, "frame_index": 0},  # global 3
            {"episode_index": 2, "frame_index": 0},  # global 6
            {"episode_index": 0, "frame_index": 2},  # global 2
            {"episode_index": 1, "frame_index": 2},  # global 5
            {"episode_index": 2, "frame_index": 1},  # global 7
        ]

        class FakeHFDataset:
            def __init__(self, rows):
                self._rows = rows

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, idx):
                return self._rows[int(idx)]

        class FakeLeRobotDataset:
            def __init__(self, repo_id, root):
                self.hf_dataset = FakeHFDataset(shuffled_rows)
                self.num_episodes = 3
                self.num_frames = 9

        with patch(
            "deminf.dataset_adapter.LeRobotDataset",
            FakeLeRobotDataset,
        ):
            result = build_episode_index_from_lerobot("/tmp/fake", "fake_repo")

        # Should still get global row IDs sorted by frame_index
        assert result[0] == [0, 1, 2], f"Episode 0: expected [0,1,2], got {result[0]}"
        assert result[1] == [3, 4, 5], f"Episode 1: expected [3,4,5], got {result[1]}"
        assert result[2] == [6, 7, 8], f"Episode 2: expected [6,7,8], got {result[2]}"

    def test_validate_episode_index_detects_wrong_global_row(self):
        """
        Test that validate_episode_index raises an assertion error
        when given an incorrect episode mapping.
        """
        from deminf.dataset_adapter import validate_episode_index

        # Create fake hf_dataset rows
        fake_rows = []
        for global_idx in range(9):
            ep_idx = global_idx // 3
            frame_idx = global_idx % 3
            fake_rows.append({
                "episode_index": ep_idx,
                "frame_index": frame_idx,
            })

        class FakeHFDataset:
            def __init__(self, rows):
                self._rows = rows

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, idx):
                return self._rows[int(idx)]

        class FakeLeRobotDataset:
            def __init__(self, repo_id, root):
                self.hf_dataset = FakeHFDataset(fake_rows)
                self.num_episodes = 3
                self.num_frames = 9

        # Construct WRONG mapping: swap some rows between episodes
        wrong_mapping = {
            0: [3, 1, 2],  # row 3 actually belongs to episode 1
            1: [0, 4, 5],  # row 0 actually belongs to episode 0
            2: [6, 7, 8],
        }

        with patch(
            "deminf.dataset_adapter.LeRobotDataset",
            FakeLeRobotDataset,
        ):
            with pytest.raises(AssertionError, match="episode_index"):
                validate_episode_index("/tmp/fake", "fake_repo", wrong_mapping)


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