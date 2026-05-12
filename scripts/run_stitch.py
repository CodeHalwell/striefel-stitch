#!/usr/bin/env python3
"""
run_stitch.py – Stitch a guest transformer block into the host model.

Usage:
    python scripts/run_stitch.py --config configs/qwen35_ministral3.yaml \\
        --alignment results/W.pt --output results/stitched_model

Produces:
    results/stitched_model/  – HuggingFace-compatible checkpoint directory
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stiefel_stitch.stitch import stitch_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Block transplantation step")
    parser.add_argument("--config", default="configs/qwen35_ministral3.yaml")
    parser.add_argument("--alignment", default="results/W.pt",
                        help="Path to the W.pt alignment matrix")
    parser.add_argument("--output", default="results/stitched_model",
                        help="Output directory for the stitched model")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading host model: %s", cfg["host_model"])
    host_tokenizer = AutoTokenizer.from_pretrained(cfg["host_model"])
    host_model = AutoModelForCausalLM.from_pretrained(
        cfg["host_model"], torch_dtype=torch.float32
    ).to(device)

    logger.info("Loading guest model: %s", cfg["guest_model"])
    guest_model = AutoModelForCausalLM.from_pretrained(
        cfg["guest_model"], torch_dtype=torch.float32
    ).to(device)

    logger.info("Loading alignment matrix from %s", args.alignment)
    W = torch.load(args.alignment, map_location=device)

    logger.info(
        "Stitching block %d (host) ← block %d (guest), alpha=%.2f",
        cfg.get("host_block_idx", 0),
        cfg.get("guest_block_idx", 0),
        cfg.get("alpha", 0.5),
    )
    stitched = stitch_model(
        host_model,
        guest_model,
        W,
        alpha=cfg.get("alpha", 0.5),
        host_block_idx=cfg.get("host_block_idx", 0),
        guest_block_idx=cfg.get("guest_block_idx", 0),
    )

    logger.info("Saving stitched model to %s", args.output)
    stitched.save_pretrained(args.output)
    host_tokenizer.save_pretrained(args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
