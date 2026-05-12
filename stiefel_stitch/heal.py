"""
heal.py – Low-rank corrective fine-tuning via LoRA (§2.6).

Implements:
  - attach_lora:  wrap linear projections with PEFT LoraConfig
  - train_lora:   causal-LM fine-tuning loop on a streaming dataset
  - merge_lora:   merge adapter weights into the base model and remove adapters
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA attachment
# ---------------------------------------------------------------------------


def attach_lora(
    model: nn.Module,
    r: int = 32,
    lora_alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """Attach LoRA adapters to *model* using PEFT.

    Args:
        model: The model to adapt (modified in place via PEFT).
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        target_modules: List of module name patterns to target.  When *None*
                        a sensible default is used (``q_proj``, ``k_proj``,
                        ``v_proj``, ``o_proj``, ``gate_proj``, ``up_proj``,
                        ``down_proj``).
        lora_dropout: Dropout probability on the LoRA path.

    Returns:
        lora_model: PEFT-wrapped model with trainable LoRA adapters.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install it with: pip install peft"
        ) from exc

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return lora_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class _TokenDataset(IterableDataset):
    """Wrap an iterable of token-ID tensors into a PyTorch IterableDataset."""

    def __init__(self, token_iter: Iterator[torch.Tensor]):
        self._iter = token_iter

    def __iter__(self):
        yield from self._iter


def train_lora(
    model: nn.Module,
    token_iterator: Iterator[torch.Tensor],
    max_tokens: int = 200_000_000,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    seq_len: int = 2048,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
) -> nn.Module:
    """Fine-tune LoRA adapters on a causal-language-modelling objective.

    *token_iterator* should yield 1-D integer tensors of tokenised text that
    will be chunked into sequences of length *seq_len* and then batched.

    Args:
        model: PEFT-wrapped model returned by :func:`attach_lora`.
        token_iterator: Iterable of flat 1-D token-ID tensors.
        max_tokens: Training budget in tokens.
        learning_rate: AdamW learning rate.
        batch_size: Micro-batch size (sequences per step before accumulation).
        gradient_accumulation_steps: Steps between optimiser updates.
        seq_len: Context length in tokens.
        device: Training device.  Defaults to the model's current device.
        save_path: If given, save the adapter checkpoint here after training.

    Returns:
        model: The same PEFT-wrapped model after training.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    tokens_seen = 0
    step = 0
    buffer: List[int] = []
    optimizer.zero_grad()

    for token_tensor in token_iterator:
        buffer.extend(token_tensor.tolist())

        while len(buffer) >= seq_len * batch_size:
            # Carve out one micro-batch
            chunk = buffer[: seq_len * batch_size]
            buffer = buffer[seq_len * batch_size :]

            input_ids = torch.tensor(chunk, dtype=torch.long).reshape(
                batch_size, seq_len
            ).to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            tokens_seen += input_ids.numel()
            step += 1

            if step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                eff_step = step // gradient_accumulation_steps
                logger.info(
                    "Step %d | loss %.4f | tokens %d / %d",
                    eff_step,
                    loss.item() * gradient_accumulation_steps,
                    tokens_seen,
                    max_tokens,
                )

            if tokens_seen >= max_tokens:
                break

        if tokens_seen >= max_tokens:
            break

    # Final optimiser update if gradients are pending
    if step % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if save_path is not None:
        model.save_pretrained(save_path)
        logger.info("Adapter checkpoint saved to %s", save_path)

    return model


# ---------------------------------------------------------------------------
# Adapter merging
# ---------------------------------------------------------------------------


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA adapter weights into the base model and remove adapters.

    After merging, the returned model is an ordinary (non-PEFT) nn.Module
    whose weights incorporate the low-rank corrections.

    Args:
        model: A PEFT-wrapped model (output of :func:`attach_lora`).

    Returns:
        base_model: The unwrapped base model with merged weights.
    """
    try:
        from peft import PeftModel
        is_peft_model = isinstance(model, PeftModel)
    except ImportError:
        is_peft_model = False

    if not is_peft_model:
        logger.warning(
            "merge_lora received a non-PeftModel (%s); returning as-is.",
            type(model).__name__,
        )
        return model

    merged = model.merge_and_unload()
    logger.info("LoRA adapters merged and removed.")
    return merged
