"""
stitch.py – Block transplantation.

Implements §2.5 of the paper:
  For each weight tensor θ_ℓ in f_θ with matching guest tensor φ_ℓ,
  zero-pad φ_ℓ to the shape of θ_ℓ, apply W* along the hidden axis where
  applicable, and form the merged tensor

      θ'_ℓ = α · θ_ℓ + (1 - α) · φ_ℓ @ W*

  Replace f_θ in M_h with h_{θ'} constructed from {θ'_ℓ}.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level weight merging
# ---------------------------------------------------------------------------


def _pad_to_shape(tensor: Tensor, target_shape: torch.Size) -> Tensor:
    """Zero-pad *tensor* so that it matches *target_shape*.

    Only trailing dimensions (right-padding) are supported.
    """
    if tensor.shape == target_shape:
        return tensor
    if len(tensor.shape) != len(target_shape):
        raise ValueError(
            f"Rank mismatch: source {tensor.shape} vs target {target_shape}"
        )
    pad: list[int] = []
    for src, tgt in zip(reversed(tensor.shape), reversed(target_shape)):
        pad += [0, tgt - src]
    return torch.nn.functional.pad(tensor, pad)


def _apply_alignment(weight: Tensor, W: Tensor) -> Tensor:
    """Apply the orthogonal alignment *W* to a guest weight tensor.

    For 2-D weight matrices the alignment is applied along the hidden
    (last) dimension:
        weight_aligned = weight @ W^T

    This ensures that when the merged block is applied to a host hidden
    state h the transformation is equivalent to first rotating h by W
    and then applying the original guest projection.
    """
    if weight.dim() == 1:
        # Bias vectors: align along the single dimension
        return weight @ W.T
    if weight.dim() == 2:
        return weight @ W.T
    # For higher-rank tensors (e.g. conv weights) we apply to the last dim
    original_shape = weight.shape
    w2d = weight.reshape(-1, original_shape[-1])
    aligned = w2d @ W.T
    return aligned.reshape(original_shape)


def transplant_block(
    host_block: nn.Module,
    guest_block: nn.Module,
    W: Tensor,
    alpha: float = 0.5,
) -> nn.Module:
    """Merge a guest transformer block into a host block via §2.5.

    For each named parameter in *host_block* that has a counterpart in
    *guest_block* (same name), the guest parameter is:
      1. Zero-padded to match the host parameter shape.
      2. Aligned with *W*.
      3. Linearly interpolated with the host parameter at ratio *alpha*.

    Parameters that exist only in the host are left unchanged.

    Args:
        host_block: The host transformer block (will NOT be modified in place;
                    a deep copy is returned).
        guest_block: The guest transformer block (read-only).
        W: Orthogonal alignment matrix of shape (d_h, d_h).
        alpha: Host weight in the convex combination.  alpha=1 → pure host;
               alpha=0 → pure (aligned) guest.

    Returns:
        merged_block: New nn.Module with merged parameters.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    merged = copy.deepcopy(host_block)

    guest_params = dict(guest_block.named_parameters())
    host_params = dict(host_block.named_parameters())

    with torch.no_grad():
        for name, host_param in merged.named_parameters():
            if name not in guest_params:
                logger.debug("No guest counterpart for %s – keeping host.", name)
                continue

            guest_param = guest_params[name]

            # Pad guest weight to host shape
            padded = _pad_to_shape(guest_param.data, host_param.shape)

            # Align padded guest weight with W
            aligned = _apply_alignment(padded, W)

            # Convex combination
            host_param.data.copy_(alpha * host_param.data + (1.0 - alpha) * aligned)
            logger.debug(
                "Merged %s: host %s  guest %s → merged %s",
                name, host_params[name].shape, guest_param.shape, host_param.shape,
            )

    return merged


# ---------------------------------------------------------------------------
# Full-model stitching
# ---------------------------------------------------------------------------


def stitch_model(
    host_model: nn.Module,
    guest_model: nn.Module,
    W: Tensor,
    alpha: float = 0.5,
    host_block_idx: int = 0,
    guest_block_idx: int = 0,
) -> nn.Module:
    """Replace one transformer block in the host model with a merged block.

    Produces M_stitched by substituting block *host_block_idx* of
    *host_model* with the transplanted block constructed from
    *guest_model*'s block *guest_block_idx*.

    Args:
        host_model: Pretrained host language model (deep-copied internally).
        guest_model: Pretrained guest language model (read-only).
        W: Orthogonal alignment matrix returned by
           :func:`~stiefel_stitch.align.iterative_procrustes`.
        alpha: Host/guest mixing weight (see :func:`transplant_block`).
        host_block_idx: Which block in *host_model* to replace.
        guest_block_idx: Which block in *guest_model* to use as donor.

    Returns:
        stitched_model: Deep copy of *host_model* with one block replaced.
    """
    stitched = copy.deepcopy(host_model)
    host_block = _get_block(host_model, host_block_idx)
    guest_block = _get_block(guest_model, guest_block_idx)

    merged_block = transplant_block(host_block, guest_block, W, alpha)

    # Inject the merged block back into the stitched model
    block_list = _get_block_list(stitched)
    block_list[host_block_idx] = merged_block

    logger.info(
        "Stitched host block %d with guest block %d (alpha=%.2f).",
        host_block_idx, guest_block_idx, alpha,
    )
    return stitched


# ---------------------------------------------------------------------------
# Internal helpers (mirrored from align.py to keep modules independent)
# ---------------------------------------------------------------------------


def _get_block_list(model: nn.Module):
    """Return the list/ModuleList of transformer blocks."""
    for attr in ("layers", "blocks", "model.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__getitem__"):
            return obj
    raise AttributeError(
        f"Cannot locate transformer block list in {type(model).__name__}."
    )


def _get_block(model: nn.Module, idx: int) -> nn.Module:
    return _get_block_list(model)[idx]
