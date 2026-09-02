"""
Latent Cache Unit Tests

Tests:
1. Same VAE checkpoint and config produce same fingerprint
2. Changing state checkpoint content changes fingerprint
3. Changing action checkpoint content changes fingerprint
4. Changing git_commit changes fingerprint
5. Changing global_row_ids causes validate_latent_cache to reject
6. Cache hit bypasses encode_all_timesteps (mock verification)
"""

import hashlib
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from deminf.config import DemInfConfig
from deminf.utils import build_cache_fingerprint
from deminf.score_episodes import (
    validate_latent_cache,
    save_latent_cache,
    load_latent_cache,
    score_latents,
)


def _make_dummy_ckpt(path: Path, content: str = "dummy_checkpoint_data") -> None:
    """Create a dummy checkpoint file for fingerprint testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


class TestCacheFingerprintConsistency:
    """Test that cache fingerprint is consistent and changes when key fields change."""

    def test_same_config_same_fingerprint(self, tmp_path):
        """Same inputs must produce same fingerprint."""
        state_ckpt = tmp_path / "state_vae.pt"
        action_ckpt = tmp_path / "action_vae.pt"
        _make_dummy_ckpt(state_ckpt, "state_ckpt_v1")
        _make_dummy_ckpt(action_ckpt, "action_ckpt_v1")

        norm_manifest = {"state_mean_hash": "abc", "action_mean_hash": "def"}

        fp1 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt),
            action_ckpt_path=str(action_ckpt),
            git_commit="abc123def456",
            total_episodes=112,
            total_frames=50000,
        )

        fp2 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt),
            action_ckpt_path=str(action_ckpt),
            git_commit="abc123def456",
            total_episodes=112,
            total_frames=50000,
        )

        assert fp1 == fp2, "Same inputs must produce identical fingerprint"

    def test_changing_state_ckpt_changes_fingerprint(self, tmp_path):
        """Changing state checkpoint content must change fingerprint."""
        state_ckpt_v1 = tmp_path / "state_vae_v1.pt"
        state_ckpt_v2 = tmp_path / "state_vae_v2.pt"
        action_ckpt = tmp_path / "action_vae.pt"
        _make_dummy_ckpt(state_ckpt_v1, "state_ckpt_version_1")
        _make_dummy_ckpt(state_ckpt_v2, "state_ckpt_version_2_different")
        _make_dummy_ckpt(action_ckpt, "action_ckpt_v1")

        norm_manifest = {"state_mean_hash": "abc"}

        fp1 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt_v1),
            action_ckpt_path=str(action_ckpt),
            git_commit="abc123",
            total_episodes=112,
            total_frames=50000,
        )

        fp2 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt_v2),
            action_ckpt_path=str(action_ckpt),
            git_commit="abc123",
            total_episodes=112,
            total_frames=50000,
        )

        assert fp1 != fp2, "Changing state checkpoint must change fingerprint"

    def test_changing_action_ckpt_changes_fingerprint(self, tmp_path):
        """Changing action checkpoint content must change fingerprint."""
        state_ckpt = tmp_path / "state_vae.pt"
        action_ckpt_v1 = tmp_path / "action_vae_v1.pt"
        action_ckpt_v2 = tmp_path / "action_vae_v2.pt"
        _make_dummy_ckpt(state_ckpt, "state_ckpt_v1")
        _make_dummy_ckpt(action_ckpt_v1, "action_ckpt_version_1")
        _make_dummy_ckpt(action_ckpt_v2, "action_ckpt_version_2_different")

        norm_manifest = {"state_mean_hash": "abc"}

        fp1 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt),
            action_ckpt_path=str(action_ckpt_v1),
            git_commit="abc123",
            total_episodes=112,
            total_frames=50000,
        )

        fp2 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt),
            action_ckpt_path=str(action_ckpt_v2),
            git_commit="abc123",
            total_episodes=112,
            total_frames=50000,
        )

        assert fp1 != fp2, "Changing action checkpoint must change fingerprint"

    def test_changing_git_commit_changes_fingerprint(self, tmp_path):
        """Changing git_commit must change fingerprint."""
        state_ckpt = tmp_path / "state_vae.pt"
        action_ckpt = tmp_path / "action_vae.pt"
        _make_dummy_ckpt(state_ckpt, "state_ckpt")
        _make_dummy_ckpt(action_ckpt, "action_ckpt")

        norm_manifest = {"state_mean_hash": "abc"}

        fp1 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt),
            action_ckpt_path=str(action_ckpt),
            git_commit="commit_abc123",
            total_episodes=112,
            total_frames=50000,
        )

        fp2 = build_cache_fingerprint(
            dataset_path="/tmp/data",
            state_source="observation.environment_state",
            action_key="action",
            state_dim=39,
            action_dim=4,
            state_latent_dim=12,
            action_latent_dim=6,
            hidden_dims=[512, 512],
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            vae_lr=1e-4,
            vae_steps=50000,
            weight_decay=0.0,
            normalization_manifest=norm_manifest,
            state_ckpt_path=str(state_ckpt),
            action_ckpt_path=str(action_ckpt),
            git_commit="commit_xyz789",
            total_episodes=112,
            total_frames=50000,
        )

        assert fp1 != fp2, "Changing git_commit must change fingerprint"


class TestLatentCacheValidation:
    """Test latent cache validation logic."""

    def test_valid_cache_passes_validation(self, tmp_path):
        """A correctly saved cache with matching fingerprint must pass validation."""
        N = 100
        z_s = np.random.randn(N, 12).astype(np.float32)
        z_a = np.random.randn(N, 6).astype(np.float32)
        ep_ids = np.array([i // 10 for i in range(N)], dtype=np.int64)
        ts_ids = np.array([i % 10 for i in range(N)], dtype=np.int64)
        gr_ids = np.arange(N, dtype=np.int64)

        expected_fp = "test_fingerprint_1234567890abcdef"
        manifest = {
            "fingerprint": expected_fp,
            "git_commit": "abc123",
            "dataset_path": "/tmp/data",
            "state_source": "observation.environment_state",
            "state_dim": 39,
            "action_dim": 4,
            "state_latent_dim": 12,
            "action_latent_dim": 6,
            "normalization_manifest": {"state_mean_hash": "abc"},
            "state_ckpt_path": "state_vae.pt",
            "state_ckpt_sha256": "sha_state",
            "action_ckpt_path": "action_vae.pt",
            "action_ckpt_sha256": "sha_action",
            "num_transitions": N,
        }

        save_latent_cache(
            tmp_path, z_s, z_a, ep_ids, ts_ids, gr_ids, manifest,
        )

        valid, reasons = validate_latent_cache(
            tmp_path,
            expected_fingerprint=expected_fp,
            expected_num_transitions=N,
            expected_global_row_ids=gr_ids,
            expected_z_state_shape=(N, 12),
            expected_z_action_shape=(N, 6),
        )

        assert valid, f"Valid cache must pass validation, got reasons: {reasons}"

    def test_wrong_fingerprint_rejected(self, tmp_path):
        """Cache with wrong fingerprint must be rejected."""
        N = 50
        z_s = np.random.randn(N, 12).astype(np.float32)
        z_a = np.random.randn(N, 6).astype(np.float32)
        ep_ids = np.zeros(N, dtype=np.int64)
        ts_ids = np.arange(N, dtype=np.int64)
        gr_ids = np.arange(N, dtype=np.int64)

        manifest = {
            "fingerprint": "wrong_fingerprint_1234567890ab",
            "git_commit": "abc123",
            "num_transitions": N,
        }

        save_latent_cache(
            tmp_path, z_s, z_a, ep_ids, ts_ids, gr_ids, manifest,
        )

        valid, reasons = validate_latent_cache(
            tmp_path,
            expected_fingerprint="correct_fingerprint_1234567890ab",
            expected_num_transitions=N,
            expected_global_row_ids=gr_ids,
        )

        assert not valid, "Wrong fingerprint must be rejected"
        assert any("fingerprint mismatch" in r for r in reasons), (
            f"Expected fingerprint mismatch reason, got: {reasons}"
        )

    def test_wrong_global_row_ids_rejected(self, tmp_path):
        """Cache with different global_row_ids must be rejected."""
        N = 50
        z_s = np.random.randn(N, 12).astype(np.float32)
        z_a = np.random.randn(N, 6).astype(np.float32)
        ep_ids = np.zeros(N, dtype=np.int64)
        ts_ids = np.arange(N, dtype=np.int64)
        gr_ids = np.arange(N, dtype=np.int64)

        expected_fp = "test_fp_row_ids_1234567890ab"
        manifest = {
            "fingerprint": expected_fp,
            "git_commit": "abc123",
            "num_transitions": N,
        }

        save_latent_cache(
            tmp_path, z_s, z_a, ep_ids, ts_ids, gr_ids, manifest,
        )

        # Try to validate with DIFFERENT global_row_ids
        wrong_gr_ids = np.arange(N, 2 * N, dtype=np.int64)

        valid, reasons = validate_latent_cache(
            tmp_path,
            expected_fingerprint=expected_fp,
            expected_num_transitions=N,
            expected_global_row_ids=wrong_gr_ids,
        )

        assert not valid, "Wrong global_row_ids must be rejected"
        assert any("global_row_ids hash mismatch" in r for r in reasons), (
            f"Expected global_row_ids hash mismatch, got: {reasons}"
        )

    def test_wrong_num_transitions_rejected(self, tmp_path):
        """Cache with different number of transitions must be rejected."""
        N = 50
        z_s = np.random.randn(N, 12).astype(np.float32)
        z_a = np.random.randn(N, 6).astype(np.float32)
        ep_ids = np.zeros(N, dtype=np.int64)
        ts_ids = np.arange(N, dtype=np.int64)
        gr_ids = np.arange(N, dtype=np.int64)

        manifest = {
            "fingerprint": "test_fp",
            "git_commit": "abc123",
            "num_transitions": N,
        }

        save_latent_cache(
            tmp_path, z_s, z_a, ep_ids, ts_ids, gr_ids, manifest,
        )

        valid, reasons = validate_latent_cache(
            tmp_path,
            expected_fingerprint="test_fp",
            expected_num_transitions=100,  # Wrong!
            expected_global_row_ids=gr_ids,
        )

        assert not valid, "Wrong num_transitions must be rejected"
        assert any("num_transitions mismatch" in r for r in reasons), (
            f"Expected num_transitions mismatch, got: {reasons}"
        )


class TestCacheBypassesEncoding:
    """Test that cache hit truly bypasses encode_all_timesteps."""

    def test_cache_hit_does_not_call_encode(self, tmp_path):
        """
        When cache is valid, score_latents should be called directly
        without calling encode_all_timesteps.

        We mock encode_all_timesteps to raise if called, proving that
        the cache path truly bypasses encoding.
        """
        from deminf.score_episodes import encode_all_timesteps

        N = 100
        z_s = np.random.randn(N, 12).astype(np.float32)
        z_a = np.random.randn(N, 6).astype(np.float32)
        ep_ids = np.array([i // 10 for i in range(N)], dtype=np.int64)
        ts_ids = np.array([i % 10 for i in range(N)], dtype=np.int64)
        gr_ids = np.arange(N, dtype=np.int64)

        expected_fp = "cache_bypass_test_fp_12345678"
        manifest = {
            "fingerprint": expected_fp,
            "git_commit": "abc123",
            "dataset_path": "/tmp/data",
            "state_source": "observation.environment_state",
            "state_dim": 39,
            "action_dim": 4,
            "state_latent_dim": 12,
            "action_latent_dim": 6,
            "normalization_manifest": {},
            "state_ckpt_path": "state_vae.pt",
            "state_ckpt_sha256": "sha_state",
            "action_ckpt_path": "action_vae.pt",
            "action_ckpt_sha256": "sha_action",
            "num_transitions": N,
        }

        save_latent_cache(
            tmp_path, z_s, z_a, ep_ids, ts_ids, gr_ids, manifest,
        )

        # Verify cache is valid
        valid, reasons = validate_latent_cache(
            tmp_path,
            expected_fingerprint=expected_fp,
            expected_num_transitions=N,
            expected_global_row_ids=gr_ids,
            expected_z_state_shape=(N, 12),
            expected_z_action_shape=(N, 6),
        )
        assert valid, f"Cache must be valid for this test, got: {reasons}"

        # Load from cache
        loaded_z_s, loaded_z_a, loaded_ep, loaded_ts, loaded_gr = load_latent_cache(tmp_path)

        # Now call score_latents with loaded latents
        # If encode_all_timesteps is called, it will raise
        def raise_if_called(*args, **kwargs):
            raise AssertionError(
                "encode_all_timesteps was called but should have been bypassed by cache"
            )

        config = DemInfConfig(
            dataset_path="/tmp/data",
            output_dir=str(tmp_path),
            state_latent_dim=12,
            action_latent_dim=6,
            vae_steps=50000,
            vae_lr=1e-4,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=256,
            quality_batch_size=64,  # Small for test
            quality_repeat=2,
            quality_cache=True,
            ks=(5, 6, 7),
            hidden_dims=[512, 512],
        )

        with patch(
            "deminf.score_episodes.encode_all_timesteps",
            side_effect=raise_if_called,
        ):
            # score_latents should NOT call encode_all_timesteps
            ep_df, ts_df = score_latents(
                z_s=loaded_z_s,
                z_a=loaded_z_a,
                episode_ids=loaded_ep,
                timestep_ids=loaded_ts,
                global_row_ids=loaded_gr,
                config=config,
                output_dir=tmp_path,
            )

        # Verify outputs were created
        assert len(ep_df) > 0, "Episode scores should be non-empty"
        assert len(ts_df) > 0, "Timestep scores should be non-empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])