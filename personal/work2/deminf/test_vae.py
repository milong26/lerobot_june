"""
VAE Smoke Tests

Tests:
1. VAE output shape matches input shape
2. KL loss is finite
3. Embedding is deterministic when use_mean=True
4. Checkpoint save/load works correctly
"""

import numpy as np
import pytest
import torch

from deminf.models import BetaVAE, MLPEncoder, MLPDecoder, vae_loss


class TestVAEOutputShape:
    """Test that VAE output shapes are correct."""

    def test_encoder_output_shape(self):
        """Encoder should output mu and logvar with shape [batch, latent_dim]."""
        batch_size = 32
        input_dim = 10
        latent_dim = 5
        encoder = MLPEncoder(input_dim, [64, 64], latent_dim)

        x = torch.randn(batch_size, input_dim)
        mu, logvar = encoder(x)

        assert mu.shape == (batch_size, latent_dim), f"mu shape: {mu.shape}"
        assert logvar.shape == (batch_size, latent_dim), f"logvar shape: {logvar.shape}"

    def test_decoder_output_shape(self):
        """Decoder should output reconstruction with shape [batch, output_dim]."""
        batch_size = 32
        latent_dim = 5
        output_dim = 10
        decoder = MLPDecoder(latent_dim, [64, 64], output_dim)

        z = torch.randn(batch_size, latent_dim)
        x_recon = decoder(z)

        assert x_recon.shape == (batch_size, output_dim), f"x_recon shape: {x_recon.shape}"

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

        assert torch.isfinite(total), f"Total loss is not finite: {total}"
        assert torch.isfinite(recon), f"Recon loss is not finite: {recon}"
        assert torch.isfinite(kl), f"KL loss is not finite: {kl}"


class TestVAEDeterministicEmbedding:
    """Test that use_mean=True gives deterministic embeddings."""

    def test_deterministic_mean_embedding(self):
        """get_embedding with use_mean=True should always return the same value."""
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
        input_dim = 10
        latent_dim = 5
        vae = BetaVAE(input_dim, latent_dim, [64, 64])
        vae.eval()

        x = torch.randn(16, input_dim)

        with torch.no_grad():
            z1 = vae.get_embedding(x, use_mean=False)
            z2 = vae.get_embedding(x, use_mean=False)

        # Should be different (with very high probability)
        assert not torch.allclose(z1, z2, rtol=1e-5, atol=1e-5)


class TestVAECheckpoint:
    """Test VAE checkpoint save/load."""

    def test_checkpoint_save_load(self, tmp_path):
        """Save and load checkpoint should preserve model weights."""
        input_dim = 10
        latent_dim = 5
        vae1 = BetaVAE(input_dim, latent_dim, [64, 64])

        ckpt_path = str(tmp_path / "vae_test.pt")

        from deminf.train_vae import save_vae_checkpoint, load_vae_checkpoint
        from deminf.config import DemInfConfig

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

        # Load into new model
        vae2 = BetaVAE(input_dim, latent_dim, [64, 64])
        ckpt = load_vae_checkpoint(ckpt_path, torch.device("cpu"))
        vae2.load_state_dict(ckpt["model_state_dict"])

        # Verify weights match
        for p1, p2 in zip(vae1.parameters(), vae2.parameters()):
            torch.testing.assert_close(p1, p2, rtol=0, atol=0)


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

        # Simple data: all points near origin
        x = torch.randn(128, input_dim) * 0.1

        # Initial loss
        vae.train()
        x_recon, mu, logvar = vae(x)
        initial_loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=0.05)

        # Train for 20 steps
        for _ in range(20):
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(x)
            loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=0.05)
            loss.backward()
            optimizer.step()

        # Final loss
        vae.eval()
        with torch.no_grad():
            x_recon, mu, logvar = vae(x)
            final_loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=0.05)

        assert final_loss < initial_loss, (
            f"Final loss ({final_loss:.4f}) should be < initial loss ({initial_loss:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])