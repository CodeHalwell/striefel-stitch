#!/usr/bin/env python3
"""
run_heal.py – LoRA corrective fine-tuning of the stitched (or baseline) model.

Usage:
    # Condition C3 – heal the stitched model
    python scripts/run_heal.py --model results/stitched_model \\
        --output results/healed_model --config configs/qwen35_ministral3.yaml

    # Condition C4 – Xavier-initialised block, same healing
    python scripts/run_heal.py --model results/stitched_model \\
        --xavier-init --output results/healed_rand \\
        --config configs/ablation.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stiefel_stitch.heal import attach_lora, merge_lora, train_lora

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def xavier_init_block(model: torch.nn.Module, block_idx: int = 0) -> None:
    """Re-initialise all parameters of block *block_idx* with Xavier uniform."""
    from stiefel_stitch.stitch import _get_block

    block = _get_block(model, block_idx)
    for name, param in block.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param.data)
        else:
            torch.nn.init.zeros_(param.data)
    logger.info("Xavier-initialised block %d.", block_idx)


def fineweb_token_iter(tokenizer, max_tokens: int, seq_len: int):
    """Stream tokenised sequences from FineWeb-Edu."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    tokens_yielded = 0
    buffer: list[int] = []
    for row in ds:
        enc = tokenizer(row["text"], add_special_tokens=False)
        buffer.extend(enc["input_ids"])
        while len(buffer) >= seq_len:
            yield torch.tensor(buffer[:seq_len], dtype=torch.long)
            buffer = buffer[seq_len:]
            tokens_yielded += seq_len
            if tokens_yielded >= max_tokens:
                return


def main():
    parser = argparse.ArgumentParser(description="LoRA healing step")
    parser.add_argument("--model", required=True,
                        help="Model checkpoint path")
    parser.add_argument("--config", default="configs/qwen35_ministral3.yaml")
    parser.add_argument("--output", default="results/healed_model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--xavier-init", action="store_true",
                        help="Re-initialise block 0 with Xavier (ablation C4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(device)

    if args.xavier_init:
        logger.info("Applying Xavier initialisation to block 0 (ablation C4).")
        xavier_init_block(model, block_idx=cfg.get("host_block_idx", 0))

    logger.info("Attaching LoRA adapters (r=%d, α=%d)…",
                cfg.get("lora_rank", 32), cfg.get("lora_alpha", 64))
    model = attach_lora(
        model,
        r=cfg.get("lora_rank", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        target_modules=cfg.get("lora_target_modules", None),
    )

    max_tokens = cfg.get("max_train_tokens", 200_000_000)
    seq_len = cfg.get("seq_len", 2048)
    logger.info("Starting LoRA training for %d tokens (seq_len=%d)…",
                max_tokens, seq_len)

    token_it = fineweb_token_iter(tokenizer, max_tokens=max_tokens, seq_len=seq_len)

    model = train_lora(
        model,
        token_iterator=token_it,
        max_tokens=max_tokens,
        learning_rate=cfg.get("learning_rate", 1e-4),
        batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        seq_len=seq_len,
        device=device,
        save_path=os.path.join(args.output, "adapter"),
    )

    logger.info("Merging LoRA adapters…")
    model = merge_lora(model)

    logger.info("Saving healed model to %s", args.output)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
