"""Unit tests for stiefel_stitch.align."""

import pytest
import torch

from stiefel_stitch.align import (
    _zero_pad_cols,
    iterative_procrustes,
    procrustes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_orthogonal(n: int) -> torch.Tensor:
    """Return a random n×n orthogonal matrix via QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    return Q


# ---------------------------------------------------------------------------
# _zero_pad_cols
# ---------------------------------------------------------------------------


class TestZeroPadCols:
    def test_no_padding_needed(self):
        t = torch.ones(4, 6)
        out = _zero_pad_cols(t, 6)
        assert out is t  # same object – no copy

    def test_pads_correctly(self):
        t = torch.ones(3, 4)
        out = _zero_pad_cols(t, 7)
        assert out.shape == (3, 7)
        assert out[:, :4].allclose(t)
        assert out[:, 4:].allclose(torch.zeros(3, 3))


# ---------------------------------------------------------------------------
# procrustes
# ---------------------------------------------------------------------------


class TestProcrustes:
    def test_output_shape(self):
        d = 8
        Y = torch.randn(20, d)
        X = torch.randn(20, d)
        W = procrustes(Y, X)
        assert W.shape == (d, d)

    def test_orthogonality(self):
        """W^T W should be close to I."""
        d = 16
        Y = torch.randn(50, d).double()
        X = torch.randn(50, d).double()
        W = procrustes(Y, X)
        eye = torch.eye(d, dtype=torch.float64)
        assert (W.T @ W).allclose(eye, atol=1e-10)

    def test_known_rotation(self):
        """Given Y @ W_true ≈ X, procrustes should recover W_true."""
        d = 8
        W_true = random_orthogonal(d).double()
        Y = torch.randn(100, d).double()
        X = Y @ W_true

        W_hat = procrustes(Y, X)
        # W_hat and W_true may differ by a sign flip on rows;
        # check that Y @ W_hat ≈ X
        assert (Y @ W_hat).allclose(X, atol=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            procrustes(torch.randn(10, 4), torch.randn(10, 5))

    def test_reduces_frobenius_norm(self):
        """||Y @ W* - X||_F should be ≤ ||Y - X||_F (W=I baseline)."""
        d = 12
        Y = torch.randn(40, d).double()
        X = torch.randn(40, d).double()
        W = procrustes(Y, X)
        loss_procrustes = (Y @ W - X).norm().item()
        loss_identity = (Y - X).norm().item()
        assert loss_procrustes <= loss_identity + 1e-8


# ---------------------------------------------------------------------------
# iterative_procrustes
# ---------------------------------------------------------------------------


class TestIterativeProcrustes:
    def test_output_shape(self):
        d = 10
        W = iterative_procrustes(torch.randn(30, d), torch.randn(30, d))
        assert W.shape == (d, d)

    def test_orthogonality(self):
        d = 12
        Y = torch.randn(40, d).double()
        X = torch.randn(40, d).double()
        W = iterative_procrustes(Y, X, alpha=0.5, n_iter=4)
        eye = torch.eye(d, dtype=torch.float64)
        assert (W.T @ W).allclose(eye, atol=1e-9)

    def test_alpha_one_matches_single_shot(self):
        """With alpha=1, iterative result should equal single-shot result."""
        d = 8
        Y = torch.randn(30, d).double()
        X = torch.randn(30, d).double()
        W_single = procrustes(Y, X)
        W_iter = iterative_procrustes(Y, X, alpha=1.0, n_iter=5)
        assert W_single.allclose(W_iter, atol=1e-9)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            iterative_procrustes(torch.randn(10, 4), torch.randn(10, 4), alpha=1.5)

    def test_n_iter_one_matches_single_shot(self):
        """n_iter=1 should produce the same result as procrustes()."""
        d = 6
        Y = torch.randn(20, d).double()
        X = torch.randn(20, d).double()
        W_single = procrustes(Y, X)
        W_iter = iterative_procrustes(Y, X, alpha=0.5, n_iter=1)
        assert W_single.allclose(W_iter, atol=1e-9)
