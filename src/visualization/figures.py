"""Figure generators (Phase 9.1–9.4) — paper-grade plots.

Each generator follows a uniform contract:

* **Inputs** — explicit, structured (no globals, no hidden coupling
  to ``results/``). The script wrapper (`scripts/06_visualize.py`)
  loads the relevant CSVs/NPZ and passes data in.
* **Output** — single file at ``out_path``; parent dir auto-created.
* **Determinism** — ``matplotlib.savefig(..., metadata={})`` strips
  timestamps so re-runs are bit-identical (per design §5.3).
* **Structural shape** — each generator's output passes the
  per-figure structural check defined in
  ``docs/02-design/09-deliverables.md`` §5.1.

Figures Phase 9 implements:

* :func:`generate_fig03_baseline_bars` — Fig 3 (BACC/F1/AUROC × 7 models)
* :func:`generate_fig04_graph_radar` — Fig 4 (radar over 5 graph variants)
* :func:`generate_fig05_confusion_matrix` — Fig 5 (7×7 confusion matrix)
* :func:`generate_fig06_roc` — Fig 6 (one-vs-rest ROC × 7 classes + macro)
* :func:`generate_fig08_tsne` — Fig 8 (t-SNE / UMAP scatter, pre vs post)
* :func:`generate_fig09_transfer` — Fig 9 (transfer settings T1–T5 bars)
* :func:`generate_figS1_training_curves` — Fig S1 (loss/accuracy per epoch)

Figures deferred to integration time (need real WSI / attention dumps):

* :func:`generate_fig01_pipeline` — hand-drawn diagram template
* :func:`generate_fig02_graph_variants` — overlay on a sample WSI
* :func:`generate_fig07_attention_overlays` — heatmap × WSI thumbnail

The deferred functions raise :class:`NotImplementedError` with a
Phase-12 pointer so callers fail loudly instead of producing empty
plots.

References:
    Design: docs/02-design/09-deliverables.md §1, §5.1
    Tests:  docs/02-design/09-deliverables.md §5.1 structural checks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

# Always use the non-interactive Agg backend — tests + CI must not
# require a DISPLAY, and matplotlib's default can pick GUI backends
# on dev machines.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.metrics import CLASS_ORDER, NUM_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

#: Strip metadata from ``savefig`` so re-runs are bit-identical (design §5.3).
#: Only ``Date`` is universally accepted by both PNG and SVG writers; other
#: keys (Producer, Creator) are SVG-rejected. Passing ``Date=None`` is enough
#: to drop the auto-generated timestamp from both formats.
_NO_METADATA: dict = {"Date": None}

#: Locked colour palette — categorical, colourblind-safe.
PALETTE = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, out_path: Path, *, dpi: int = 300) -> Path:
    """Save fig with deterministic metadata + close."""
    _ensure_parent(out_path)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        metadata=_NO_METADATA,
    )
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Fig 3 — Baseline bar chart
# --------------------------------------------------------------------------- #


def generate_fig03_baseline_bars(
    *,
    out_path: Union[str, Path],
    model_names: Sequence[str],
    metric_means: Mapping[str, Sequence[float]],
    metric_stds: Optional[Mapping[str, Sequence[float]]] = None,
) -> Path:
    """Fig 3 — grouped bar chart of {BACC, F1, AUROC} × N models.

    Args:
        out_path: SVG output path.
        model_names: Length-N model labels (e.g. baseline registry order).
        metric_means: Dict ``{metric_name → length-N array}``. Must
            contain at least ``balanced_accuracy``, ``weighted_f1``,
            ``macro_auroc`` (rest are ignored).
        metric_stds: Optional matching std arrays for error bars.

    Structural check: SVG XML has ``len(model_names)`` ``<g class="bar-group">``
    rectangles per metric, and the file validates as XML.
    """
    out = Path(out_path)
    metrics = ["balanced_accuracy", "weighted_f1", "macro_auroc"]
    metric_labels = ["BACC", "F1", "AUROC"]

    means = np.asarray([metric_means[m] for m in metrics], dtype=np.float64)
    stds = (
        np.asarray([metric_stds[m] for m in metrics], dtype=np.float64)
        if metric_stds is not None
        else np.zeros_like(means)
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_models = len(model_names)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        ax.bar(
            x + (i - (n_models - 1) / 2) * width,
            means[:, i],
            width=width,
            yerr=stds[:, i] if metric_stds is not None else None,
            label=name,
            color=PALETTE[i % len(PALETTE)],
            capsize=3,
            gid=f"bar-group-{name}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title("Baseline comparison (mean ± std, 5-fold)")
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Fig 4 — Graph radar
# --------------------------------------------------------------------------- #


def generate_fig04_graph_radar(
    *,
    out_path: Union[str, Path],
    graph_variants: Sequence[str],
    axes: Sequence[str],
    values: Sequence[Sequence[float]],
) -> Path:
    """Fig 4 — radar chart of metrics across the 5 graph variants.

    Args:
        out_path: SVG output path.
        graph_variants: 5 variant names (locked: spatial_knn, feature_sim,
            dual_edge, hierarchical, heterogeneous).
        axes: K metric axis labels (e.g. ``[BACC, F1, AUROC, kappa]``).
        values: ``[len(graph_variants) × K]`` matrix of metric values.

    Structural check: SVG has ``len(graph_variants)`` filled polygons.
    """
    out = Path(out_path)
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape != (len(graph_variants), len(axes)):
        raise ValueError(
            f"values shape {vals.shape} != ({len(graph_variants)}, {len(axes)})"
        )

    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes)
    ax.set_ylim(0, 1)

    for i, name in enumerate(graph_variants):
        row = vals[i].tolist()
        row += row[:1]
        line = ax.plot(angles, row, label=name, color=PALETTE[i % len(PALETTE)])
        line[0].set_gid(f"polygon-{name}")
        ax.fill(angles, row, alpha=0.1, color=PALETTE[i % len(PALETTE)], gid=f"polygon-{name}-fill")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Graph-variant performance comparison")
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Fig 5 — Confusion matrix
# --------------------------------------------------------------------------- #


def generate_fig05_confusion_matrix(
    *,
    out_path: Union[str, Path],
    confusion: np.ndarray,
    class_names: Sequence[str] = CLASS_ORDER,
    normalize: bool = True,
) -> Path:
    """Fig 5 — 7×7 confusion matrix heatmap.

    Args:
        out_path: PNG output path.
        confusion: ``[7, 7]`` integer count matrix.
        class_names: Axis labels in locked order
            ``[MEL, MCT, SCC, PNST, PLC, TRB, HIS]``.
        normalize: If True, plot row-normalised (recall) values.

    Structural check (PIL): 7×7 grid; class labels match locked order.
    """
    out = Path(out_path)
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"confusion must be [{NUM_CLASSES}, {NUM_CLASSES}]; got {cm.shape}"
        )
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1.0 if normalize else cm.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Fig 6 — ROC curves
# --------------------------------------------------------------------------- #


def generate_fig06_roc(
    *,
    out_path: Union[str, Path],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str] = CLASS_ORDER,
) -> Path:
    """Fig 6 — one-vs-rest ROC for each of the 7 classes + macro-average.

    Structural check: SVG has 7 class curves + 1 macro + 1 diagonal = 9 lines.
    """
    from sklearn.metrics import auc, roc_curve

    out = Path(out_path)
    yp = np.asarray(y_prob, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.int64)
    if yp.ndim != 2 or yp.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"y_prob must be [N, {NUM_CLASSES}]; got {yp.shape}"
        )

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fprs = []
    tprs_interp = []
    common_grid = np.linspace(0, 1, 200)

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((yt == i).astype(int), yp[:, i])
        a = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=PALETTE[i % len(PALETTE)],
            label=f"{name} (AUC={a:.3f})",
            gid=f"roc-{name}",
        )
        fprs.append(fpr)
        tprs_interp.append(np.interp(common_grid, fpr, tpr))

    macro_tpr = np.mean(tprs_interp, axis=0)
    ax.plot(
        common_grid,
        macro_tpr,
        linestyle="--",
        color="black",
        linewidth=2,
        label=f"Macro-avg (AUC={auc(common_grid, macro_tpr):.3f})",
        gid="roc-macro",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", linewidth=1, gid="roc-diagonal")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Fig 8 — t-SNE / UMAP scatter
# --------------------------------------------------------------------------- #


def generate_fig08_tsne(
    *,
    out_path: Union[str, Path],
    embeddings_pre: np.ndarray,
    embeddings_post: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str] = CLASS_ORDER,
    seed: int = 42,
    perplexity: float = 15.0,
) -> Path:
    """Fig 8 — t-SNE projections of pre-GNN vs post-GNN embeddings.

    Two side-by-side scatter plots, colored by class.
    Structural check: 7 distinct cluster colours present.
    """
    from sklearn.manifold import TSNE

    out = Path(out_path)
    if embeddings_pre.shape[0] != labels.shape[0]:
        raise ValueError("embeddings_pre rows must equal labels length")
    if embeddings_post.shape[0] != labels.shape[0]:
        raise ValueError("embeddings_post rows must equal labels length")

    n_samples = embeddings_pre.shape[0]
    eff_perp = float(min(perplexity, max(2.0, (n_samples - 1) / 3)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, emb, title in (
        (axes[0], embeddings_pre, "Pre-GNN tile embeddings"),
        (axes[1], embeddings_post, "Post-GNN tile embeddings"),
    ):
        proj = TSNE(
            n_components=2, perplexity=eff_perp, random_state=seed, init="pca"
        ).fit_transform(emb)
        for c in range(NUM_CLASSES):
            mask = labels == c
            if not mask.any():
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                s=12,
                color=PALETTE[c % len(PALETTE)],
                label=class_names[c],
                alpha=0.7,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.32, 1.0), fontsize=8)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Fig 9 — Transfer comparison
# --------------------------------------------------------------------------- #


def generate_fig09_transfer(
    *,
    out_path: Union[str, Path],
    settings: Sequence[str],
    bacc_means: Sequence[float],
    bacc_stds: Optional[Sequence[float]] = None,
) -> Path:
    """Fig 9 — bar chart of T1–T5 transfer settings.

    Structural check: SVG has ``len(settings)`` bars on the BACC axis.
    """
    out = Path(out_path)
    means = np.asarray(bacc_means, dtype=np.float64)
    stds = (
        np.asarray(bacc_stds, dtype=np.float64)
        if bacc_stds is not None
        else np.zeros_like(means)
    )
    if means.shape[0] != len(settings):
        raise ValueError("settings and bacc_means must have equal length")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        list(settings),
        means,
        yerr=stds,
        color=PALETTE[: len(settings)],
        capsize=4,
    )
    for i, b in enumerate(bars):
        b.set_gid(f"transfer-bar-{settings[i]}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Cross-species transfer (T1–T5)")
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Fig S1 — Training curves
# --------------------------------------------------------------------------- #


def generate_figS1_training_curves(
    *,
    out_path: Union[str, Path],
    history_per_run: Mapping[str, Mapping[str, Sequence[float]]],
    primary_metric: str = "val_balanced_accuracy",
) -> Path:
    """Fig S1 — supplementary training curves (loss + primary metric per run).

    Args:
        history_per_run: ``{run_name: {metric_name: [per-epoch values]}}``.
            Each run must have at least ``train_loss`` and the
            ``primary_metric``.

    Structural check: epoch axis ≤ 100; one subplot row per metric (2 rows).
    """
    out = Path(out_path)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    for i, (run_name, metrics) in enumerate(history_per_run.items()):
        loss = list(metrics.get("train_loss", []))
        bacc = list(metrics.get(primary_metric, []))
        epochs_loss = np.arange(len(loss))
        epochs_bacc = np.arange(len(bacc))
        color = PALETTE[i % len(PALETTE)]
        axes[0].plot(epochs_loss, loss, label=run_name, color=color, gid=f"loss-{run_name}")
        axes[1].plot(epochs_bacc, bacc, label=run_name, color=color, gid=f"bacc-{run_name}")

    axes[0].set_ylabel("Train loss")
    axes[1].set_ylabel(primary_metric)
    axes[1].set_xlabel("Epoch")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Training curves (Fig S1)")
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Deferred figures — heavy data dependencies
# --------------------------------------------------------------------------- #


def generate_fig01_pipeline(*, out_path: Union[str, Path]) -> Path:  # pragma: no cover
    """Fig 1 — pipeline architecture diagram (deferred to Phase 12).

    The published version is a hand-drawn SVG template populated with
    auto-generated stage labels. The template lives in
    ``docs/02-design/figures/`` once Phase 12 lands.
    """
    raise NotImplementedError(
        "Fig 1 (pipeline diagram) is hand-drawn + auto-fill; deferred to Phase 12. "
        "See docs/02-design/09-deliverables.md §1."
    )


def generate_fig02_graph_variants(*, out_path: Union[str, Path], **kwargs) -> Path:  # pragma: no cover
    """Fig 2 — overlay 5 graph variants on a sample WSI (deferred to Phase 12).

    Requires a real WSI thumbnail and the 5 ``data/graphs/<variant>/MEL_001.pt``
    files; both arrive only after Phases 2/4 run end-to-end on real data.
    """
    raise NotImplementedError(
        "Fig 2 (graph variants on a WSI) requires real preprocessing+graph output; "
        "deferred to Phase 12. See docs/02-design/09-deliverables.md §1."
    )


def generate_fig07_attention_overlays(
    *, out_path: Union[str, Path], **kwargs
) -> Path:  # pragma: no cover
    """Fig 7 — attention heatmaps × WSI thumbnails (deferred to Phase 12).

    Requires a trained checkpoint + per-tile attention dump from
    :class:`src.models.VetGigaGraph`; both arrive only after Phase 12
    runs the smoke training on a 5-WSI fixture.
    """
    raise NotImplementedError(
        "Fig 7 (attention overlays) needs a trained checkpoint + per-tile "
        "attention dump; deferred to Phase 12. See docs/02-design/09-deliverables.md §1."
    )
