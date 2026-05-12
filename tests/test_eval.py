"""Unit tests for stiefel_stitch.eval."""

import math

import torch
import torch.nn as nn

from stiefel_stitch.eval import compute_perplexity, evaluate_hellaswag


# ---------------------------------------------------------------------------
# Minimal stub language model
# ---------------------------------------------------------------------------


class StubLM(nn.Module):
    """Tiny language model stub that returns a fixed loss."""

    def __init__(self, fixed_loss: float = 2.0):
        super().__init__()
        self._loss = torch.tensor(fixed_loss)
        # Dummy parameter so that next(model.parameters()) works
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, labels=None):
        # Return an object with a .loss attribute
        class _Out:
            pass
        out = _Out()
        out.loss = self._loss.clone()
        return out


class StubTokenizer:
    def __call__(self, text, return_tensors="pt"):
        # Return a dict-like encoding regardless of text
        ids = torch.zeros(1, 10, dtype=torch.long)
        return {"input_ids": ids}


# ---------------------------------------------------------------------------
# compute_perplexity
# ---------------------------------------------------------------------------


class TestComputePerplexity:
    def _token_ids(self, n=600):
        return torch.zeros(n, dtype=torch.long)

    def test_returns_float(self):
        model = StubLM(fixed_loss=1.0)
        ppl = compute_perplexity(model, self._token_ids(), device="cpu")
        assert isinstance(ppl, float)

    def test_fixed_loss_gives_expected_ppl(self):
        """For a fixed NLL of L, perplexity should be exp(L)."""
        fixed_loss = 2.0
        model = StubLM(fixed_loss=fixed_loss)
        ppl = compute_perplexity(
            model, self._token_ids(n=600), stride=256, max_length=512, device="cpu"
        )
        # ppl should be close to exp(2.0)
        expected = math.exp(fixed_loss)
        assert abs(ppl - expected) / expected < 0.5  # within 50 % (stub artefact)

    def test_short_sequence(self):
        """Sequence shorter than max_length should still work."""
        model = StubLM(fixed_loss=1.5)
        ppl = compute_perplexity(
            model, torch.zeros(100, dtype=torch.long),
            stride=50, max_length=200, device="cpu",
        )
        assert ppl > 0


# ---------------------------------------------------------------------------
# evaluate_hellaswag
# ---------------------------------------------------------------------------


class TestEvaluateHellaSwag:
    def _examples(self, n=4, correct_label=0):
        return [
            {
                "ctx": "The sky is",
                "endings": ["blue.", "green.", "red.", "purple."],
                "label": correct_label,
            }
        ] * n

    def test_returns_float_in_range(self):
        model = StubLM(fixed_loss=1.0)
        tok = StubTokenizer()
        acc = evaluate_hellaswag(model, tok, self._examples(), device="cpu")
        assert 0.0 <= acc <= 1.0

    def test_empty_examples_returns_zero(self):
        model = StubLM()
        tok = StubTokenizer()
        acc = evaluate_hellaswag(model, tok, [], device="cpu")
        assert acc == 0.0

    def test_all_same_loss_first_wins(self):
        """When all endings have identical loss the first is chosen (index 0)."""
        model = StubLM(fixed_loss=1.0)
        tok = StubTokenizer()
        # label=0 means the correct ending is index 0
        examples = self._examples(n=10, correct_label=0)
        acc = evaluate_hellaswag(model, tok, examples, device="cpu")
        assert acc == 1.0
