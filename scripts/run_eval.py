#!/usr/bin/env python3
"""
run_eval.py – Evaluate a model on perplexity (WikiText-2) and HellaSwag.

Usage:
    python scripts/run_eval.py --model results/stitched_model \\
        --output results/eval_stitched.yaml

    # Evaluate the original host model (condition C1):
    python scripts/run_eval.py --model Qwen/Qwen3.5-4B \\
        --output results/eval_host.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stiefel_stitch.eval import compute_perplexity, evaluate_hellaswag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or local checkpoint path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stride", type=int, default=256,
                        help="Sliding-window stride for perplexity")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Context window size for perplexity")
    parser.add_argument("--hellaswag-n", type=int, default=200,
                        help="Number of HellaSwag examples to evaluate")
    parser.add_argument("--output", default=None,
                        help="YAML file to write results to")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(device)

    # ---- WikiText-2 perplexity ----
    from datasets import load_dataset

    logger.info("Loading WikiText-2 test split…")
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in wt2 if row["text"].strip())
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    ppl = compute_perplexity(
        model, token_ids,
        stride=args.stride,
        max_length=args.max_length,
        device=device,
    )
    logger.info("WikiText-2 perplexity: %.2f", ppl)

    # ---- HellaSwag accuracy ----
    logger.info("Loading HellaSwag validation set…")
    hs_ds = load_dataset("hellaswag", split="validation")
    examples = [
        {"ctx": row["ctx"], "endings": row["endings"], "label": row["label"]}
        for row in hs_ds
    ][: args.hellaswag_n]

    accuracy = evaluate_hellaswag(model, tokenizer, examples, device=device)
    logger.info("HellaSwag accuracy (%d examples): %.4f", len(examples), accuracy)

    results = {
        "model": args.model,
        "wikitext2_perplexity": round(ppl, 4),
        "hellaswag_accuracy": round(accuracy, 4),
        "hellaswag_n": len(examples),
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            yaml.dump(results, f)
        logger.info("Results written to %s", args.output)
    else:
        print(yaml.dump(results))


if __name__ == "__main__":
    main()
