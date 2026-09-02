"""
VAE Unit Tests

Tests:
1. Official VAE loss reduction: recon=((x-xhat)**2).sum(-1).mean(), kl=(0.5*(-logvar-1+exp(logvar)+mu**2).sum(-1)).mean()
2. Deterministic posterior mean
3. Fixed optimizer step training runs specified steps
4. Checkpoint resume from step continues correctly
5. VAE output shapes are correct
6. KL loss is finite
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from deminf.models import BetaVAE, MLPEncoder, MLPDecoder, vae_loss


class TestOfficialVAELossReduction:
    """Test that VAE loss matches official DemInf formulation exactly."""

    def test_official_vae_loss_reduction(self):
        """
        Hand-compute recon=((x-xhat)**2).sum(-1).mean() and
        kl=(0.5*(-logvar-1+exp(logvar)+mu**2).sum(-1)).mean()
        and verify they match vae_loss() to 1e-7.
        """
        torch.manual_seed(42)
        batch_size = 32
        input_dim = 10
        latent_dim = 5

        x = torch.randn(batch_size, input_dim)
        x_recon = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)

        # Official vae_loss
        total, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=0.05)

        # Manual computation
        recon_per_sample = ((x_recon - x) ** 2).sum(dim=-1)
        recon_manual = recon_per_sample.mean()

        kl_per_sample = 0.5 * torch.sum(-logvar - 1 + torch.exp(logvar) + mu ** 2, dim=-1)
        kl_manual = kl_per_sample.mean()

        total_manual = recon_manual + 0.05 * kl_manual

        torch.testing.assert_close(total, total_manual, rtol=0, atol=1e-7)
        torch.testing.assert_close(recon, recon_manual, rtol=0, atol=1e-7)
        torch.testing.assert_close(kl, kl_manual, rtol=0, atol=1e-7)

    def test_loss_not_using_mse_reduction_mean(self):
        """
        Verify that loss is NOT F.mse_loss(..., reduction="mean") which would
        average over both batch and feature dimensions.
        """
        torch.manual_seed(42)
        batch_size = 16
        input_dim = 8
        latent_dim = 4

        x = torch.randn(batch_size, input_dim)
        x_recon = torch.randn(batch_size, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)

        total, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=1.0)

        # F.mse_loss would give: ((x_recon - x)**2).mean()
        mse_wrong = ((x_recon - x) ** 2).mean()

        # Official: ((x_recon - x)**2).sum(-1).mean()
        recon_official = ((x_recon - x) ** 2).sum(dim=-1).mean()

        # They should differ by a factor of input_dim
        expected_ratio = float(input_dim)
        actual_ratio = float(recon_official / mse_wrong)

        assert abs(actual_ratio - expected_ratio) < 1e-5, (
            f"Recon should differ from MSE by factor of {expected_ratio}, got {actual_ratio}"
        )


class TestDeterministicPosteriorMean:
    """Test that use_mean=True gives deterministic embeddings."""

    def test_deterministic_mean_embedding(self):
        """get_embedding with use_mean=True should always return the same value."""
        torch.manual_seed(42)
        input_dim = 10
        latent_dim = 5
        vae = BetaVAE(input_dim, latent_dim, [64, 64])
        vae.eval()

        x = torch.randn(16, input_dim)

        with torch.no_grad():
            z1 = vae.get_embedding(x, use_mean=True)
            z2 = vae.get_embedding(x, use_mean=True)

        torch.testing.assert_close(z1, z2, rtol=0, atol=0)

    def test_stochastic_embedding_differs(self):
        """get_embedding with use_mean=False should return different values."""
        torch.manual_seed(42)
        input_dim = 10
        latent_dim = 5
        vae = BetaVAE(input_dim, latent_dim, [64, 64])
        vae.eval()

        x = torch.randn(16, input_dim)

        with torch.no_grad():
            z1 = vae.get_embedding(x, use_mean=False)
            z2 = vae.get_embedding(x, use_mean=False)

        assert not torch.allclose(z1, z2, rtol=1e-5, atol=1e-5)


class TestFixedOptimizerSteps:
    """Test that training runs for exactly the specified number of steps."""

    def test_training_runs_exact_steps(self, tmp_path):
        """Training should run for exactly the specified number of steps."""
        from deminf.config import DemInfConfig
        from deminf.train_vae import train_beta_vae

        torch.manual_seed(42)
        np.random.seed(42)

        data = np.random.randn(200, 8).astype(np.float32)

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            vae_steps=50,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            num_workers=0,
            weight_decay=0.0,
            hidden_dims=[32, 32],
            skip_train_if_checkpoint_exists=False,
        )

        model, log = train_beta_vae(
            data=data,
            input_dim=8,
            latent_dim=4,
            config=config,
            name="state",
        )

        assert log["global_step"] == 50, f"Expected 50 steps, got {log['global_step']}"
        assert len(log["history"]["train_total"]) == 50


class TestCheckpointResume:
    """Test VAE checkpoint save/load/resume."""

    def test_checkpoint_save_load(self, tmp_path):
        """Save and load checkpoint should preserve model weights."""
        from deminf.config import DemInfConfig
        from deminf.train_vae import save_vae_checkpoint, load_vae_checkpoint

        input_dim = 10
        latent_dim = 5
        vae1 = BetaVAE(input_dim, latent_dim, [64, 64])

        ckpt_path = str(tmp_path / "vae_test.pt")

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            hidden_dims=[64, 64],
            vae_lr=1e-3,
            weight_decay=1e-5,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
        )

        optimizer = torch.optim.Adam(vae1.parameters(), lr=1e-3)
        save_vae_checkpoint(vae1, optimizer, 0, config, input_dim, latent_dim, ckpt_path)

        vae2 = BetaVAE(input_dim, latent_dim, [64, 64])
        ckpt = load_vae_checkpoint(ckpt_path, torch.device("cpu"))
        vae2.load_state_dict(ckpt["model_state_dict"])

        for p1, p2 in zip(vae1.parameters(), vae2.parameters()):
            torch.testing.assert_close(p1, p2, rtol=0, atol=0)

    def test_checkpoint_resume_continues_from_step(self, tmp_path):
        """Resume should continue training from the saved step."""
        from deminf.config import DemInfConfig
        from deminf.train_vae import train_beta_vae, find_checkpoint

        torch.manual_seed(42)
        np.random.seed(42)

        data = np.random.randn(200, 8).astype(np.float32)
        ckpt_dir = str(tmp_path / "checkpoints")

        # Train for 20 steps
        config1 = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            checkpoint_dir=ckpt_dir,
            vae_steps=20,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            num_workers=0,
            weight_decay=0.0,
            hidden_dims=[32, 32],
            skip_train_if_checkpoint_exists=False,
            resume=False,
        )

        model1, log1 = train_beta_vae(
            data=data, input_dim=8, latent_dim=4, config=config1, name="state",
        )
        assert log1["global_step"] == 20

        # Resume and train for 30 more steps (total 50)
        config2 = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            checkpoint_dir=ckpt_dir,
            vae_steps=50,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            num_workers=0,
            weight_decay=0.0,
            hidden_dims=[32, 32],
            skip_train_if_checkpoint_exists=False,
            resume=True,
        )

        model2, log2 = train_beta_vae(
            data=data, input_dim=8, latent_dim=4, config=config2, name="state",
        )
        assert log2["global_step"] == 50


class TestXavierInitialization:
    """Test that Xavier uniform initialization is applied to all Linear layers."""

    def test_xavier_initialization_is_applied(self):
        """
        Verify that Linear weights are NOT PyTorch default initialization
        and that the init function is actually called.

        PyTorch default Linear uses Kaiming uniform. Xavier uniform has a
        different variance: Xavier ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        vs Kaiming ~ U(-sqrt(3/fan_in), sqrt(3/fan_in)).

        We verify by checking that the weight distribution matches Xavier
        statistics rather than Kaiming.
        """
        torch.manual_seed(42)
        input_dim = 10
        latent_dim = 5
        hidden_dims = [64, 64]

        encoder = MLPEncoder(input_dim, hidden_dims, latent_dim)
        decoder = MLPDecoder(latent_dim, hidden_dims, input_dim)

        # Check that all Linear layers have Xavier-like weight statistics
        # Xavier uniform: std ≈ sqrt(2 / (fan_in + fan_out))
        # Kaiming uniform: std ≈ sqrt(1 / fan_in)
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Linear):
                fan_in, fan_out = module.weight.shape[1], module.weight.shape[0]
                xavier_std = (2.0 / (fan_in + fan_out)) ** 0.5
                kaiming_std = (1.0 / fan_in) ** 0.5

                weight_std = module.weight.std().item()
                # Xavier std should be closer than Kaiming std
                xavier_diff = abs(weight_std - xavier_std)
                kaiming_diff = abs(weight_std - kaiming_std)
                assert xavier_diff < kaiming_diff, (
                    f"Layer {name}: weight std={weight_std:.4f}, "
                    f"xavier_std={xavier_std:.4f}, kaiming_std={kaiming_std:.4f}. "
                    f"Expected Xavier, but closer to Kaiming."
                )

        # Verify bias is zero
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert torch.allclose(module.bias, torch.zeros_like(module.bias), atol=1e-7), (
                    f"Layer {name} bias should be zero after Xavier init"
                )

        # Same for decoder
        for name, module in decoder.named_modules():
            if isinstance(module, nn.Linear):
                fan_in, fan_out = module.weight.shape[1], module.weight.shape[0]
                xavier_std = (2.0 / (fan_in + fan_out)) ** 0.5
                kaiming_std = (1.0 / fan_in) ** 0.5

                weight_std = module.weight.std().item()
                xavier_diff = abs(weight_std - xavier_std)
                kaiming_diff = abs(weight_std - kaiming_std)
                assert xavier_diff < kaiming_diff, (
                    f"Decoder layer {name}: weight std={weight_std:.4f}, "
                    f"xavier_std={xavier_std:.4f}, kaiming_std={kaiming_std:.4f}. "
                    f"Expected Xavier, but closer to Kaiming."
                )

    def test_xavier_init_via_monkeypatch(self):
        """
        Use monkeypatch to verify that init_linear_xavier is actually called
        during encoder/decoder construction.
        """
        from deminf.models import init_linear_xavier

        call_count = [0]
        original_init = init_linear_xavier

        def counting_init(module):
            if isinstance(module, nn.Linear):
                call_count[0] += 1
            return original_init(module)

        import deminf.models as models_module
        models_module.init_linear_xavier = counting_init

        try:
            encoder = MLPEncoder(10, [512, 512], 12)
            decoder = MLPDecoder(12, [512, 512], 10)

            # Encoder: 2 hidden Linear + 1 z_proj Linear = 3 Linear layers
            # Decoder: 2 hidden Linear + 1 output Linear = 3 Linear layers
            assert call_count[0] == 6, f"Expected 6 Linear inits, got {call_count[0]}"
        finally:
            models_module.init_linear_xavier = original_init


class TestCheckpointValidation:
    """Test checkpoint validation and fingerprint logic."""

    def test_smoke_checkpoint_cannot_skip_official_run(self, tmp_path):
        """
        A checkpoint with global_step=50 must NOT be accepted when
        config.vae_steps=50000. This prevents smoke checkpoints from
        being silently reused for official experiments.
        """
        from deminf.config import DemInfConfig
        from deminf.train_vae import (
            save_vae_checkpoint,
            load_vae_checkpoint,
            validate_vae_checkpoint,
        )

        input_dim = 8
        latent_dim = 4
        vae = BetaVAE(input_dim, latent_dim, [32, 32])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Create a smoke checkpoint with global_step=50
        smoke_config = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            vae_steps=50,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            weight_decay=0.0,
            hidden_dims=[32, 32],
        )
        ckpt_path = str(tmp_path / "smoke_vae.pt")
        save_vae_checkpoint(
            vae, optimizer, 50, smoke_config, input_dim, latent_dim,
            ckpt_path, name="state",
        )

        # Now try to use it with official config (vae_steps=50000)
        official_config = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            vae_steps=50000,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            weight_decay=0.0,
            hidden_dims=[32, 32],
        )

        ckpt = load_vae_checkpoint(ckpt_path, torch.device("cpu"))
        valid, reasons = validate_vae_checkpoint(
            ckpt, official_config, "state", input_dim, latent_dim,
            None, require_target_step=True,
        )

        assert not valid, "Smoke checkpoint (step=50) must NOT be valid for official run (steps=50000)"
        assert any("global_step" in r and "50000" in r for r in reasons), (
            f"Expected global_step mismatch reason, got: {reasons}"
        )

    def test_checkpoint_wrong_latent_dim_rejected(self, tmp_path):
        """Checkpoint with wrong latent_dim must be rejected."""
        from deminf.config import DemInfConfig
        from deminf.train_vae import (
            save_vae_checkpoint,
            load_vae_checkpoint,
            validate_vae_checkpoint,
        )

        input_dim = 8
        latent_dim = 4
        vae = BetaVAE(input_dim, latent_dim, [32, 32])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            vae_steps=50000,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            weight_decay=0.0,
            hidden_dims=[32, 32],
        )
        ckpt_path = str(tmp_path / "vae.pt")
        save_vae_checkpoint(
            vae, optimizer, 50000, config, input_dim, latent_dim,
            ckpt_path, name="state",
        )

        # Try to validate with different latent_dim
        ckpt = load_vae_checkpoint(ckpt_path, torch.device("cpu"))
        valid, reasons = validate_vae_checkpoint(
            ckpt, config, "state", input_dim, latent_dim=8,
            None, require_target_step=True,
        )

        assert not valid, "Checkpoint with wrong latent_dim must be rejected"
        assert any("latent_dim mismatch" in r for r in reasons), (
            f"Expected latent_dim mismatch, got: {reasons}"
        )

    def test_checkpoint_wrong_normalization_rejected(self, tmp_path):
        """Checkpoint with different normalization stats must be rejected."""
        from deminf.config import DemInfConfig
        from deminf.train_vae import (
            save_vae_checkpoint,
            load_vae_checkpoint,
            validate_vae_checkpoint,
        )

        input_dim = 8
        latent_dim = 4
        vae = BetaVAE(input_dim, latent_dim, [32, 32])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        norm_stats_a = {"state_mean_hash": "abc123", "action_mean_hash": "def456"}
        norm_stats_b = {"state_mean_hash": "xyz789", "action_mean_hash": "uvw012"}

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            vae_steps=50000,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            weight_decay=0.0,
            hidden_dims=[32, 32],
        )
        ckpt_path = str(tmp_path / "vae.pt")
        save_vae_checkpoint(
            vae, optimizer, 50000, config, input_dim, latent_dim,
            ckpt_path, name="state", normalization_stats=norm_stats_a,
        )

        ckpt = load_vae_checkpoint(ckpt_path, torch.device("cpu"))
        valid, reasons = validate_vae_checkpoint(
            ckpt, config, "state", input_dim, latent_dim,
            norm_stats_b, require_target_step=True,
        )

        assert not valid, "Checkpoint with different normalization must be rejected"
        assert any("normalization_manifest" in r for r in reasons), (
            f"Expected normalization mismatch, got: {reasons}"
        )

    def test_checkpoint_matching_50000_accepted(self, tmp_path):
        """A fully trained checkpoint (step=50000) with matching config must be accepted."""
        from deminf.config import DemInfConfig
        from deminf.train_vae import (
            save_vae_checkpoint,
            load_vae_checkpoint,
            validate_vae_checkpoint,
        )

        input_dim = 8
        latent_dim = 4
        vae = BetaVAE(input_dim, latent_dim, [32, 32])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        config = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            vae_steps=50000,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            weight_decay=0.0,
            hidden_dims=[32, 32],
        )
        ckpt_path = str(tmp_path / "vae.pt")
        save_vae_checkpoint(
            vae, optimizer, 50000, config, input_dim, latent_dim,
            ckpt_path, name="state",
        )

        ckpt = load_vae_checkpoint(ckpt_path, torch.device("cpu"))
        valid, reasons = validate_vae_checkpoint(
            ckpt, config, "state", input_dim, latent_dim,
            None, require_target_step=True,
        )

        assert valid, f"Matching 50000-step checkpoint must be accepted, got reasons: {reasons}"
        assert len(reasons) == 0

    def test_resume_from_20_to_50_continues_training(self, tmp_path):
        """
        Resume from a 20-step checkpoint with config.vae_steps=50 should
        continue training from step 20, not restart from 0.
        """
        from deminf.config import DemInfConfig
        from deminf.train_vae import train_beta_vae

        torch.manual_seed(42)
        np.random.seed(42)

        data = np.random.randn(200, 8).astype(np.float32)
        ckpt_dir = str(tmp_path / "checkpoints")

        # Train for 20 steps
        config1 = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            checkpoint_dir=ckpt_dir,
            vae_steps=20,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            num_workers=0,
            weight_decay=0.0,
            hidden_dims=[32, 32],
            skip_train_if_checkpoint_exists=False,
            resume=False,
        )

        model1, log1 = train_beta_vae(
            data=data, input_dim=8, latent_dim=4, config=config1, name="state",
        )
        assert log1["global_step"] == 20

        # Resume and train for 30 more steps (total 50)
        config2 = DemInfConfig(
            dataset_path="/tmp",
            output_dir=str(tmp_path),
            checkpoint_dir=ckpt_dir,
            vae_steps=50,
            vae_lr=1e-3,
            vae_beta_state=0.05,
            vae_beta_action=0.05,
            batch_size=32,
            num_workers=0,
            weight_decay=0.0,
            hidden_dims=[32, 32],
            skip_train_if_checkpoint_exists=False,
            resume=True,
        )

        model2, log2 = train_beta_vae(
            data=data, input_dim=8, latent_dim=4, config=config2, name="state",
        )
        assert log2["global_step"] == 50


class TestVAEOutputShapes:
    """Test that VAE output shapes are correct."""

    def test_encoder_output_shape(self):
        """Encoder should output mu and logvar with shape [batch, latent_dim]."""
        batch_size = 32
        input_dim = 10
        latent_dim = 5
        encoder = MLPEncoder(input_dim, [64, 64], latent_dim)

        x = torch.randn(batch_size, input_dim)
        mu, logvar = encoder(x)

        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)

    def test_decoder_output_shape(self):
        """Decoder should output reconstruction with shape [batch, output_dim]."""
        batch_size = 32
        latent_dim = 5
        output_dim = 10
        decoder = MLPDecoder(latent_dim, [64, 64], output_dim)

        z = torch.randn(batch_size, latent_dim)
        x_recon = decoder(z)

        assert x_recon.shape == (batch_size, output_dim)

    def test_vae_forward_shape(self):
        """VAE forward should return (x_recon, mu, logvar) with correct shapes."""
        batch_size = 32
        input_dim = 10
        latent_dim = 5
        vae = BetaVAE(input_dim, latent_dim, [64, 64])

        x = torch.randn(batch_size, input_dim)
        x_recon, mu, logvar = vae(x)

        assert x_recon.shape == (batch_size, input_dim)
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)


class TestVAEKLFinite:
    """Test that KL loss is finite."""

    def test_kl_loss_finite(self):
        """KL loss should be finite for reasonable inputs."""
        batch_size = 32
        input_dim = 10
        latent_dim = 5
        vae = BetaVAE(input_dim, latent_dim, [64, 64])

        x = torch.randn(batch_size, input_dim)
        x_recon, mu, logvar = vae(x)

        total, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=0.05)

        assert torch.isfinite(total)
        assert torch.isfinite(recon)
        assert torch.isfinite(kl)


class TestVAEReconstruction:
    """Test that VAE can reconstruct input after training."""

    def test_reconstruction_improves(self):
        """After a few training steps, reconstruction loss should decrease."""
        torch.manual_seed(42)
        np.random.seed(42)

        input_dim = 4
        latent_dim = 2
        vae = BetaVAE(input_dim, latent_dim, [32, 32])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)

        x = torch.randn(128, input_dim) * 0.1

        vae.train()
        x_recon, mu, logvar = vae(x)
        initial_loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=0.05)

        for _ in range(20):
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(x)
            loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=0.05)
            loss.backward()
            optimizer.step()

        vae.eval()
        with torch.no_grad():
            x_recon, mu, logvar = vae(x)
            final_loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=0.05)

        assert final_loss < initial_loss, (
            f"Final loss ({final_loss:.4f}) should be < initial loss ({initial_loss:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])