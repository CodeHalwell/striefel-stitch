"""
align.py – Orthogonal Procrustes alignment on the Stiefel manifold.

Implements:
  - Single-shot Procrustes solution (Schönemann, 1966)
  - Iterative refinement à la Iterative Closest Point (Besl & McKay, 1992)
  - Activation capture helpers for host and guest models
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Procrustes solver
# ---------------------------------------------------------------------------


def procrustes(Y_tilde: Tensor, X: Tensor) -> Tensor:
    """Solve the orthogonal Procrustes problem.

    Find W* ∈ St(d, d) minimising ||Y_tilde @ W - X||_F^2.

    The closed-form solution is W* = U @ V^T, where
    M = Y_tilde^T @ X = U Σ V^T  (Schönemann, 1966).

    Args:
        Y_tilde: Zero-padded guest activations of shape (N, d).
        X: Host activations of shape (N, d).

    Returns:
        W: Orthogonal alignment matrix of shape (d, d).
    """
    if Y_tilde.shape != X.shape:
        raise ValueError(
            f"Y_tilde and X must have the same shape, "
            f"got {Y_tilde.shape} vs {X.shape}"
        )

    M = Y_tilde.T @ X  # (d, d)
    U, _S, Vh = torch.linalg.svd(M, full_matrices=True)
    W = U @ Vh  # W^* = U V^T
    return W


def iterative_procrustes(
    Y_tilde: Tensor,
    X: Tensor,
    alpha: float = 0.5,
    n_iter: int = 8,
) -> Tensor:
    """Iterative Procrustes refinement (ICP-style).

    Iterates:
        W_{k+1} = Proc(Y_tilde, alpha * X + (1 - alpha) * Y_tilde @ W_k)

    starting from W_0 = Proc(Y_tilde, X).

    Args:
        Y_tilde: Zero-padded guest activations of shape (N, d).
        X: Host activations of shape (N, d).
        alpha: Mixing coefficient ∈ [0, 1].  alpha=1 recovers single-shot
               Procrustes; alpha=0 converges to identity.
        n_iter: Number of refinement iterations (K in the paper).

    Returns:
        W: Refined orthogonal alignment matrix of shape (d, d).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    W = procrustes(Y_tilde, X)
    for k in range(1, n_iter):
        target = alpha * X + (1.0 - alpha) * (Y_tilde @ W)
        W_new = procrustes(Y_tilde, target)
        delta = (W_new - W).norm().item()
        W = W_new
        logger.debug("Procrustes iter %d/%d  ΔW=%.6f", k, n_iter - 1, delta)

    return W


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


def _zero_pad_cols(tensor: Tensor, target_cols: int) -> Tensor:
    """Right-pad a 2-D tensor with zeros along dimension 1."""
    if tensor.shape[1] == target_cols:
        return tensor
    pad = torch.zeros(
        tensor.shape[0], target_cols - tensor.shape[1],
        dtype=tensor.dtype, device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


@torch.no_grad()
def capture_activations(
    host_model: nn.Module,
    guest_model: nn.Module,
    input_ids: List[Tensor],
    host_block_idx: int = 0,
    guest_block_idx: int = 0,
    device: Optional[str] = None,
) -> tuple[Tensor, Tensor]:
    """Capture block-output activations from the host and guest models.

    Registers forward hooks on the specified transformer blocks and runs
    the models on each sequence in *input_ids*.  The activations are
    concatenated along the sequence dimension and the guest activations
    are zero-padded to match the host hidden dimension.

    Args:
        host_model: Pretrained host language model.
        guest_model: Pretrained guest language model.
        input_ids: List of integer token-ID tensors, each of shape (1, T).
        host_block_idx: Index of the transformer block to hook in the host.
        guest_block_idx: Index of the transformer block to hook in the guest.
        device: Target device string (e.g. ``"cuda"``).  If *None* the host
                model's device is used.

    Returns:
        X: Host block outputs concatenated over the corpus, shape (NT, d_h).
        Y_tilde: Zero-padded guest block outputs,            shape (NT, d_h).
    """
    if device is None:
        device = next(host_model.parameters()).device

    host_acts: List[Tensor] = []
    guest_acts: List[Tensor] = []

    def _make_hook(store: List[Tensor]):
        def _hook(_module, _input, output):
            # output may be a tuple (hidden_states, ...) for some architectures
            hidden = output[0] if isinstance(output, tuple) else output
            store.append(hidden.detach().cpu().reshape(-1, hidden.shape[-1]))
        return _hook

    # Retrieve the transformer block layers
    host_block = _get_block(host_model, host_block_idx)
    guest_block = _get_block(guest_model, guest_block_idx)

    h_hook = host_block.register_forward_hook(_make_hook(host_acts))
    g_hook = guest_block.register_forward_hook(_make_hook(guest_acts))

    try:
        for ids in input_ids:
            ids = ids.to(device)
            host_model(ids)

        host_device_guest = next(guest_model.parameters()).device
        for ids in input_ids:
            ids = ids.to(host_device_guest)
            guest_model(ids)
    finally:
        h_hook.remove()
        g_hook.remove()

    X = torch.cat(host_acts, dim=0)          # (NT, d_h)
    Y = torch.cat(guest_acts, dim=0)         # (NT, d_g)

    d_h = X.shape[1]
    Y_tilde = _zero_pad_cols(Y, d_h)         # (NT, d_h)

    logger.info(
        "Captured activations: X %s  Y_tilde %s", X.shape, Y_tilde.shape
    )
    return X, Y_tilde


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_block(model: nn.Module, idx: int) -> nn.Module:
    """Return the *idx*-th transformer block from a HuggingFace-style model."""
    # Common attribute names used by various HF architectures
    for attr in ("layers", "blocks", "transformer.h", "model.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__getitem__"):
            return obj[idx]
    raise AttributeError(
        f"Cannot locate transformer block list in {type(model).__name__}. "
        "Pass the block directly or subclass and override _get_block."
    )
