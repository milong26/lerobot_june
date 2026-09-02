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