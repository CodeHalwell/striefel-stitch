"""Unit tests for stiefel_stitch.stitch."""

import pytest
import torch
import torch.nn as nn

from stiefel_stitch.stitch import (
    _pad_to_shape,
    transplant_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_orthogonal(n: int) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    return Q


class TinyBlock(nn.Module):
    """Minimal two-layer block for testing."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=True)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


# ---------------------------------------------------------------------------
# _pad_to_shape
# ---------------------------------------------------------------------------


class TestPadToShape:
    def test_no_op_when_equal(self):
        t = torch.randn(3, 4)
        out = _pad_to_shape(t, t.shape)
        assert out.shape == t.shape
        assert out.allclose(t)

    def test_pads_1d(self):
        t = torch.ones(5)
        out = _pad_to_shape(t, torch.Size([8]))
        assert out.shape == (8,)
        assert out[:5].allclose(t)
        assert out[5:].allclose(torch.zeros(3))

    def test_pads_2d(self):
        t = torch.ones(3, 4)
        out = _pad_to_shape(t, torch.Size([3, 7]))
        assert out.shape == (3, 7)
        assert out[:, :4].allclose(t)
        assert out[:, 4:].allclose(torch.zeros(3, 3))

    def test_rank_mismatch_raises(self):
        with pytest.raises(ValueError, match="Rank mismatch"):
            _pad_to_shape(torch.randn(3, 4), torch.Size([3, 4, 5]))


# ---------------------------------------------------------------------------
# transplant_block
# ---------------------------------------------------------------------------


class TestTransplantBlock:
    def _make_blocks(self, d_host=8, d_guest=6):
        torch.manual_seed(0)
        host = TinyBlock(d_host, d_host)
        guest = TinyBlock(d_guest, d_guest)
        return host, guest

    def test_output_is_module(self):
        host, guest = self._make_blocks()
        W = random_orthogonal(8)
        merged = transplant_block(host, guest, W, alpha=0.5)
        assert isinstance(merged, nn.Module)

    def test_host_not_mutated(self):
        host, guest = self._make_blocks()
        W = random_orthogonal(8)
        host_w_before = host.linear1.weight.data.clone()
        transplant_block(host, guest, W, alpha=0.5)
        assert host.linear1.weight.data.allclose(host_w_before)

    def test_alpha_one_recovers_host(self):
        """With alpha=1.0 the merged block should have the same weights as host."""
        host, guest = self._make_blocks()
        W = random_orthogonal(8)
        merged = transplant_block(host, guest, W, alpha=1.0)
        for (name, hp), mp in zip(
            host.named_parameters(), merged.parameters()
        ):
            assert hp.allclose(mp), f"Mismatch at {name}"

    def test_merged_weights_shape_matches_host(self):
        host, guest = self._make_blocks(d_host=8, d_guest=6)
        W = random_orthogonal(8)
        merged = transplant_block(host, guest, W, alpha=0.5)
        for (hn, hp), (mn, mp) in zip(
            host.named_parameters(), merged.named_parameters()
        ):
            assert hp.shape == mp.shape, (
                f"Shape mismatch at {hn}: host {hp.shape} vs merged {mp.shape}"
            )

    def test_invalid_alpha_raises(self):
        host, guest = self._make_blocks()
        W = random_orthogonal(8)
        with pytest.raises(ValueError, match="alpha"):
            transplant_block(host, guest, W, alpha=1.5)

    def test_forward_pass_after_stitch(self):
        """Merged block should accept host-shaped inputs."""
        d = 8
        host, guest = self._make_blocks(d_host=d, d_guest=6)
        W = random_orthogonal(d)
        merged = transplant_block(host, guest, W, alpha=0.5)
        x = torch.randn(2, d)
        out = merged(x)
        assert out.shape == (2, d)
