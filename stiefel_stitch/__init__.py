"""
stiefel_stitch – orthogonal alignment of cross-architecture transformer blocks
via Procrustes optimisation on the Stiefel manifold, with low-rank corrective
fine-tuning.
"""

from .align import procrustes, iterative_procrustes, capture_activations
from .stitch import transplant_block, stitch_model
from .heal import attach_lora, train_lora, merge_lora
from .eval import compute_perplexity, evaluate_hellaswag

__all__ = [
    "procrustes",
    "iterative_procrustes",
    "capture_activations",
    "transplant_block",
    "stitch_model",
    "attach_lora",
    "train_lora",
    "merge_lora",
    "compute_perplexity",
    "evaluate_hellaswag",
]
