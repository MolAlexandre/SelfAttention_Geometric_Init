"""
Symmetry score for W_QK = W_Q @ W_K^T.

Definition (Section 2 of the paper):
    s(M) = (||M_s||_F^2 - ||M_a||_F^2) / ||M||_F^2

where M_s = (M + M^T) / 2 is the symmetric part and
      M_a = (M - M^T) / 2 is the skew-symmetric part.

s ∈ [-1, 1]:  +1 → fully symmetric,  -1 → fully skew-symmetric.
"""

import numpy as np
import torch


def symmetry_score(W_qk: torch.Tensor) -> float:
    """
    Args:
        W_qk: square matrix  [d_model, d_model]
    Returns:
        float in [-1, 1]
    """
    M_s = 0.5 * (W_qk + W_qk.T)
    M_a = 0.5 * (W_qk - W_qk.T)
    norm_s = torch.norm(M_s, p='fro').pow(2)
    norm_a = torch.norm(M_a, p='fro').pow(2)
    return ((norm_s - norm_a) / (norm_s + norm_a)).item()


def compute_model_symmetry(model) -> dict:
    """
    Compute per-layer symmetry scores for a BERTForMLM or VITForClassification model.

    Returns:
        dict with keys ``layer_0``, ``layer_1``, ... and ``average``.
    """
    if hasattr(model, 'encoder'):
        layers = model.encoder.layers           # BERT
    elif hasattr(model, 'encoder_layers'):
        layers = model.encoder_layers           # ViT
    else:
        raise AttributeError(
            f"Unsupported model type '{type(model).__name__}'. "
            "Expected 'encoder.layers' (BERT) or 'encoder_layers' (ViT)."
        )

    scores = {}
    with torch.no_grad():
        for i, layer in enumerate(layers):
            Wq = layer.attention.query.weight   # [d_model, d_model]
            Wk = layer.attention.key.weight
            scores[f'layer_{i}'] = symmetry_score(Wq @ Wk.T)

    scores['average'] = float(np.mean([v for k, v in scores.items() if k != 'average']))
    return scores


def log_symmetry_scores(scores: dict, epoch: int, prefix: str = "") -> None:
    """Pretty-print symmetry scores."""
    print(f"\n{prefix}[Epoch {epoch + 1}] Symmetry Scores:")
    print("-" * 50)
    for k, v in scores.items():
        if k != 'average':
            print(f"  {k}: {v:+.4f}")
    print("-" * 50)
    print(f"  Average: {scores['average']:+.4f}")
    print("-" * 50)
