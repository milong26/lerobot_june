"""
KSG Estimator Unit Tests

Tests:
1. Independent vs correlated variables: correlated group should have higher MI score
2. Input samples < max(k)+2 should raise clear exception
3. Chunked and full backend should give approximately consistent results on small data
4. Same seed should produce identical subset selection
"""

import numpy as np
import pytest

from deminf.ksg import ksg_local_scores, validate_ksg_inputs


class TestKSGIndependentVsCorrelated:
    """Test that KSG correctly distinguishes independent from correlated variables."""

    def test_correlated_has_higher_score(self):
        """
        Construct x ~ N(0,1), independent y ~ N(0,1) vs correlated y = x + 0.05*noise.
        Correlated group average KSG MI should be higher than independent group.
        """
        np.random.seed(42)
        N = 500

        # Independent case
        x_indep = np.random.randn(N, 1).astype(np.float32)
        y_indep = np.random.randn(N, 1).astype(np.float32)

        # Correlated case
        x_corr = np.random.randn(N, 1).astype(np.float32)
        noise = np.random.randn(N, 1).astype(np.float32) * 0.05
        y_corr = x_corr + noise

        ks = (5, 6, 7)

        scores_indep = ksg_local_scores(x_indep, y_indep, ks=ks, backend="full")
        scores_corr = ksg_local_scores(x_corr, y_corr, ks=ks, backend="full")

        mean_indep = np.mean(scores_indep)
        mean_corr = np.mean(scores_corr)

        # Correlated should have higher score (less negative in deminf_rank mode)
        assert mean_corr > mean_indep, (
            f"Correlated mean ({mean_corr:.4f}) should be > independent mean ({mean_indep:.4f})"
        )


class TestKSGInsufficientSamples:
    """Test that insufficient samples raise a clear exception."""

    def test_too_few_samples_raises_error(self):
        """Input samples < max(k)+2 should raise ValueError."""
        np.random.seed(42)
        N = 5  # max(k)=7, need at least 9 samples
        z_s = np.random.randn(N, 3).astype(np.float32)
        z_a = np.random.randn(N, 3).astype(np.float32)
        ks = (5, 6, 7)

        with pytest.raises(ValueError, match="must be greater than"):
            validate_ksg_inputs(z_s, z_a, ks)

    def test_just_enough_samples(self):
        """N = max(k)+2 should not raise."""
        np.random.seed(42)
        N = 9  # max(k)=7, need at least 8+1=9
        z_s = np.random.randn(N, 3).astype(np.float32)
        z_a = np.random.randn(N, 3).astype(np.float32)
        ks = (5, 6, 7)

        # Should not raise
        validate_ksg_inputs(z_s, z_a, ks)


class TestKSGBackendConsistency:
    """Test that chunked and full backend give approximately consistent results."""

    def test_chunked_vs_full(self):
        """On small data, chunked and full backend should give similar results."""
        np.random.seed(42)
        N = 200
        z_s = np.random.randn(N, 3).astype(np.float32)
        z_a = np.random.randn(N, 3).astype(np.float32)
        ks = (5, 6, 7)

        scores_full = ksg_local_scores(z_s, z_a, ks=ks, backend="full")
        scores_chunked = ksg_local_scores(z_s, z_a, ks=ks, backend="chunked", chunk_size=64)

        # Should be very close (may differ slightly due to floating point)
        max_diff = np.max(np.abs(scores_full - scores_chunked))
        assert max_diff < 1e-4, f"Max difference between backends: {max_diff}"


class TestKSGReproducibility:
    """Test that same seed produces identical results."""

    def test_same_seed_same_subset(self):
        """Running KSG with same input should produce identical scores."""
        np.random.seed(123)
        N = 300
        z_s = np.random.randn(N, 4).astype(np.float32)
        z_a = np.random.randn(N, 4).astype(np.float32)
        ks = (5, 6, 7)

        scores1 = ksg_local_scores(z_s, z_a, ks=ks, backend="full")
        scores2 = ksg_local_scores(z_s, z_a, ks=ks, backend="full")

        np.testing.assert_array_equal(scores1, scores2)


class TestKSGNaNInput:
    """Test that NaN/Inf inputs are caught."""

    def test_nan_in_zs_raises_error(self):
        z_s = np.random.randn(50, 3).astype(np.float32)
        z_s[10, 0] = np.nan
        z_a = np.random.randn(50, 3).astype(np.float32)

        with pytest.raises(ValueError, match="NaN"):
            validate_ksg_inputs(z_s, z_a, (5, 6, 7))

    def test_inf_in_za_raises_error(self):
        z_s = np.random.randn(50, 3).astype(np.float32)
        z_a = np.random.randn(50, 3).astype(np.float32)
        z_a[5, 1] = np.inf

        with pytest.raises(ValueError, match="Inf"):
            validate_ksg_inputs(z_s, z_a, (5, 6, 7))


class TestKSGShapeMismatch:
    """Test that mismatched sample counts are caught."""

    def test_shape_mismatch_raises_error(self):
        z_s = np.random.randn(50, 3).astype(np.float32)
        z_a = np.random.randn(48, 3).astype(np.float32)

        with pytest.raises(ValueError, match="same number of samples"):
            validate_ksg_inputs(z_s, z_a, (5, 6, 7))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])