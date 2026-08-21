"""
eval.py – Evaluation utilities (§7).

Implements:
  - compute_perplexity:  sliding-window perplexity on WikiText-2
  - evaluate_hellaswag:  accuracy on a HellaSwag subset
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perplexity (sliding-window)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    token_ids: Tensor,
    stride: int = 256,
    max_length: int = 512,
    device: Optional[str] = None,
) -> float:
    """Compute sliding-window perplexity on a long token sequence.

    Follows the standard HuggingFace sliding-window evaluation:
    the context window of size *max_length* is shifted by *stride* tokens
    each step.  Only the tokens in the non-overlapping "new" portion of each
    window contribute to the loss, preventing double-counting.

    Args:
        model: Language model with a ``forward(input_ids, labels=…)`` interface
               that returns an object with a ``.loss`` attribute.
        token_ids: Flat 1-D tensor of integer token IDs (the full test corpus).
        stride: Number of new tokens per sliding step.
        max_length: Context window size.
        device: Evaluation device.  Defaults to model's current device.

    Returns:
        ppl: Perplexity (scalar float).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    token_ids = token_ids.to(device)
    seq_len = token_ids.numel()

    nlls: List[Tensor] = []
    prev_end = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end  # tokens that are genuinely "new"

        input_ids = token_ids[begin_loc:end_loc].unsqueeze(0)
        target_ids = input_ids.clone()
        # Mask out context-only tokens so they don't contribute to the loss
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids=input_ids, labels=target_ids)
        nll = outputs.loss * trg_len
        nlls.append(nll.cpu())

        prev_end = end_loc
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = math.exp(total_nll.item() / prev_end)
    logger.info("Perplexity: %.2f", ppl)
    return ppl


# ---------------------------------------------------------------------------
# HellaSwag accuracy
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_hellaswag(
    model: nn.Module,
    tokenizer,
    examples: List[dict],
    device: Optional[str] = None,
) -> float:
    """Evaluate zero-shot HellaSwag accuracy via likelihood ranking.

    For each example the model scores each of the four candidate continuations
    by computing the average per-token negative log-likelihood over the
    continuation tokens only.  The candidate with the lowest NLL is chosen.

    Args:
        model: Language model compatible with ``forward(input_ids, labels=…)``.
        tokenizer: HuggingFace tokenizer paired with *model*.
        examples: List of HellaSwag example dicts with keys
                  ``"ctx"`` (context string) and ``"endings"`` (list of four
                  continuation strings).  The correct answer index is stored
                  in ``"label"`` (int or str).
        device: Evaluation device.

    Returns:
        accuracy: Fraction of examples answered correctly.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct = 0

    for ex in examples:
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])

        scores: List[float] = []
        for ending in endings:
            full_text = ctx + " " + ending
            enc = tokenizer(full_text, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            # Tokenise context only to find where the continuation starts
            ctx_enc = tokenizer(ctx, return_tensors="pt")
            ctx_len = ctx_enc["input_ids"].shape[1]

            target_ids = input_ids.clone()
            target_ids[:, :ctx_len] = -100  # mask context tokens

            outputs = model(input_ids=input_ids, labels=target_ids)
            # outputs.loss is mean NLL over non-masked tokens
            scores.append(outputs.loss.item())

        if scores.index(min(scores)) == label:
            correct += 1

    accuracy = correct / len(examples) if examples else 0.0
    logger.info("HellaSwag accuracy: %.4f  (%d/%d)", accuracy, correct, len(examples))
    return accuracy
