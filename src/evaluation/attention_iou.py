"""Attention map quantitative validation against CATCH polygon GT masks.

Phase B of µPDCA #8: validate that the D-2 v2 cosine attention from
:class:`src.models.gigapath_slide._AttentionExtractingAdapter` actually
concentrates on the polygon-annotated tumor regions.

Three metrics per WSI:
    - **IoU** between binarized attention map and tumor mask
    - **Pearson correlation** between attention values and tumor mask
    - **AUC-PR** of attention as a "tumor or not" classifier

Binarization strategy: **top-K matching** — convert attention to a binary
map by selecting the top-K highest-attention tiles where K is the number
of GT tumor tiles in the same WSI. This is a fair per-WSI threshold
(scale-invariant) and matches the AUC-PR interpretation.

Pre-analysis (``_PRE-µPDCA-8-annotation-survey.md``):
    - 75 WSIs have 0 tumor-mapped tiles → IoU undefined; return ``None``
    - 89 WSIs have < 5% mapping coverage overall → low-power evidence
    - Phase B should report stats on the 275 WSI subset with > 0 tumor tiles

References:
    Plan: docs/01-plan/features/annotation-activation-scenario2.plan.md §3 I4
    Drift: D-23 (HeteroGAT missing) — independent of this evaluator
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.io_utils import (
    CATCH_TUMOR_RANGE,
    TILE_LABEL_UNKNOWN,
)


def tumor_mask_from_labels(labels: np.ndarray) -> np.ndarray:
    """Convert 13-way CATCH labels to a binary tumor mask.

    Returns a ``[N]`` ``bool`` array; True where the tile center lies inside
    a tumor-category polygon (CATCH categories 7-13). Unmapped tiles (-1)
    are False.
    """
    lo, hi = CATCH_TUMOR_RANGE
    return (labels >= lo) & (labels <= hi)


def top_k_binarize(
    attention: np.ndarray,
    k: int,
) -> np.ndarray:
    """Binarize ``attention[N]`` by selecting the top-``k`` highest values.

    Returns ``[N]`` ``bool``. Ties at the threshold are broken arbitrarily
    (``argpartition`` order).
    """
    n = attention.shape[0]
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    threshold_idx = np.argpartition(attention, n - k)[n - k:]
    mask = np.zeros(n, dtype=bool)
    mask[threshold_idx] = True
    return mask


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two ``[N]`` bool masks.

    Returns ``0.0`` when both masks are empty (degenerate but well-defined).
    """
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def compute_pearson(attention: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    """Pearson correlation between continuous attention and binary GT mask.

    Returns ``None`` if either side has zero variance (constant attention
    or all-GT-zero / all-GT-one), which makes Pearson undefined.
    """
    if attention.std() == 0 or gt_mask.std() == 0:
        return None
    r = float(np.corrcoef(attention, gt_mask.astype(np.float64))[0, 1])
    # Guard against NaN escape from edge cases (e.g., 1 unique value)
    if not np.isfinite(r):
        return None
    return r


def compute_auc_pr(attention: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    """Area under the precision-recall curve for attention-as-tumor-classifier.

    Returns ``None`` if there are no positive GT samples (PR is undefined).
    Uses ``sklearn.metrics.average_precision_score`` which is the standard
    convention.
    """
    if gt_mask.sum() == 0:
        return None
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(gt_mask.astype(int), attention))
    except Exception:
        return None


def compute_attention_iou(
    cls_attention,
    tile_labels: np.ndarray,
) -> dict:
    """Compute IoU / Pearson / AUC-PR for one WSI's attention vs polygon GT.

    Args:
        cls_attention: ``[N]`` attention values (Tensor or ndarray). Must
            be on CPU and float. Sum is irrelevant — only relative ordering
            matters for IoU binarization, and absolute values matter for
            Pearson/AUC-PR.
        tile_labels: ``[N]`` int32 array, 13-way CATCH labels (or -1 unmapped).

    Returns dict::

        {
            "iou":        float | None,    # None when no GT tumor tiles
            "pearson":    float | None,    # None when undefined variance
            "auc_pr":     float | None,    # None when no positive GT
            "n_tiles":    int,
            "n_tumor_gt": int,             # GT tumor tile count
            "n_attn_top": int,             # how many top-K we picked
        }
    """
    # Tensor → ndarray
    try:
        attn = cls_attention.detach().cpu().numpy().astype(np.float64)
    except AttributeError:
        attn = np.asarray(cls_attention, dtype=np.float64)
    if attn.ndim != 1:
        raise ValueError(f"cls_attention must be [N]; got shape {attn.shape}")
    n = attn.shape[0]
    if tile_labels.shape != (n,):
        raise ValueError(
            f"tile_labels shape {tile_labels.shape} != cls_attention shape ({n},)"
        )

    gt_mask = tumor_mask_from_labels(tile_labels)
    n_tumor_gt = int(gt_mask.sum())

    result = {
        "iou": None,
        "pearson": None,
        "auc_pr": None,
        "n_tiles": int(n),
        "n_tumor_gt": n_tumor_gt,
        "n_attn_top": 0,
    }

    if n_tumor_gt == 0:
        # WSI has zero GT tumor tiles — IoU + AUC-PR both undefined.
        # Pearson is technically still defined if attention has variance,
        # but the correlation would be against an all-zero vector → 0
        # variance for gt_mask, so we return None too.
        return result

    pred_mask = top_k_binarize(attn, k=n_tumor_gt)
    result["n_attn_top"] = int(pred_mask.sum())
    result["iou"] = compute_iou(pred_mask, gt_mask)
    result["pearson"] = compute_pearson(attn, gt_mask)
    result["auc_pr"] = compute_auc_pr(attn, gt_mask)
    return result
