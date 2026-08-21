#!/usr/bin/env python3
"""
run_alignment.py – Capture activations and compute orthogonal Procrustes alignment.

Usage:
    python scripts/run_alignment.py --config configs/qwen35_ministral3.yaml

Outputs (saved to results/):
    W.pt          – alignment matrix W* of shape (d_h, d_h)
    cosine_sim.txt – mean cosine similarity before/after alignment
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import yaml

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stiefel_stitch.align import capture_activations, iterative_procrustes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_input_ids(tokenizer, texts, max_length: int, n_sequences: int):
    ids_list = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        ids_list.append(enc["input_ids"])
        if len(ids_list) >= n_sequences:
            break
    return ids_list


def mean_cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    A_norm = torch.nn.functional.normalize(A, dim=-1)
    B_norm = torch.nn.functional.normalize(B, dim=-1)
    return (A_norm * B_norm).sum(dim=-1).mean().item()


def permutation_baseline(Y_tilde: torch.Tensor, X: torch.Tensor) -> float:
    """Shuffle rows of Y_tilde and compute cosine similarity with X."""
    perm = torch.randperm(Y_tilde.shape[0])
    return mean_cosine_similarity(Y_tilde[perm], X)


def main():
    parser = argparse.ArgumentParser(description="Stiefel alignment step")
    parser.add_argument("--config", default="configs/qwen35_ministral3.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

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

    # ---- Calibration corpus ----
    from datasets import load_dataset

    logger.info("Loading calibration corpus: %s", cfg["calibration_dataset"])
    ds = load_dataset(cfg["calibration_dataset"], split="train")
    texts = [row["text"] for row in ds if row["text"].strip()][
        : cfg.get("n_calibration_sequences", 1024)
    ]

    input_ids = build_input_ids(
        host_tokenizer,
        texts,
        max_length=cfg.get("calibration_seq_len", 256),
        n_sequences=cfg.get("n_calibration_sequences", 1024),
    )

    logger.info("Capturing activations from %d sequences…", len(input_ids))
    X, Y_tilde = capture_activations(
        host_model,
        guest_model,
        input_ids,
        host_block_idx=cfg.get("host_block_idx", 0),
        guest_block_idx=cfg.get("guest_block_idx", 0),
        device=device,
    )

    # ---- Permutation baseline ----
    c_perm = permutation_baseline(Y_tilde, X)
    logger.info("Cosine similarity (permutation baseline): %.4f", c_perm)

    # ---- Procrustes alignment ----
    logger.info("Running iterative Procrustes (K=%d, α=%.2f)…",
                cfg.get("n_procrustes_iter", 8), cfg.get("alpha", 0.5))

    # Work in float64 for numerical stability of SVD
    X64 = X.double()
    Y64 = Y_tilde.double()

    W = iterative_procrustes(
        Y64, X64,
        alpha=cfg.get("alpha", 0.5),
        n_iter=cfg.get("n_procrustes_iter", 8),
    ).float()

    # ---- Post-alignment similarity ----
    Y_aligned = (Y_tilde.double() @ W.double()).float()
    c_aligned = mean_cosine_similarity(Y_aligned, X)
    logger.info("Cosine similarity (post-alignment):       %.4f", c_aligned)
    logger.info("Δ cosine similarity:                      %.4f", c_aligned - c_perm)

    # ---- Save results ----
    w_path = os.path.join(args.output_dir, "W.pt")
    torch.save(W, w_path)
    logger.info("Alignment matrix saved to %s", w_path)

    sim_path = os.path.join(args.output_dir, "cosine_sim.txt")
    with open(sim_path, "w") as f:
        f.write(f"c_perm={c_perm:.6f}\n")
        f.write(f"c_aligned={c_aligned:.6f}\n")
        f.write(f"delta={c_aligned - c_perm:.6f}\n")
    logger.info("Cosine similarities written to %s", sim_path)


if __name__ == "__main__":
    main()
