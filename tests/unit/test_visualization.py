"""Unit tests for ``src/visualization/`` (Phase 9, deliverables).

Coverage map (design ``docs/02-design/09-deliverables.md`` §5):

| Section | What we cover                                             |
|---------|-----------------------------------------------------------|
| §5.1    | Per-figure existence + structural checks (figs 3-6, 8-9, S1) |
| §5.1    | Deferred figures raise NotImplementedError (figs 1, 2, 7)    |
| §5.2    | Per-table .csv + .tex existence + schema check (8 tables)    |
| §5.3    | Reproducibility — re-run produces identical CSV/PNG bytes    |

Figures 1, 2, 7 require real WSI / attention data (`Phase 12`) — we
only verify the deferred-stub raises with a Phase-12 pointer, not the
output shape.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation.metrics import CLASS_ORDER, NUM_CLASSES
from src.visualization import (
    DEFERRED_FIGURES,
    FIGURE_REGISTRY,
    TABLE_REGISTRY,
    generate_fig01_pipeline,
    generate_fig02_graph_variants,
    generate_fig03_baseline_bars,
    generate_fig04_graph_radar,
    generate_fig05_confusion_matrix,
    generate_fig06_roc,
    generate_fig07_attention_overlays,
    generate_fig08_tsne,
    generate_fig09_transfer,
    generate_figS1_training_curves,
    generate_table01_dataset,
    generate_table02_baseline,
    generate_table03_graphs,
    generate_table04_backbones,
    generate_table05_fusion,
    generate_table06_transfer,
    generate_table07_per_class,
    generate_table08_cost,
)

NUM_BASELINES = 7  # ABMIL, DSMIL, TransMIL, CLAM_SB, CLAM_MB, VetGigaGraph, random

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def baseline_metrics():
    rng = np.random.default_rng(0)
    return {
        "model_names": ["abmil", "dsmil", "transmil", "clam_sb", "clam_mb", "vetgigagraph", "random"],
        "means": {
            "balanced_accuracy": rng.uniform(0.4, 0.85, NUM_BASELINES).tolist(),
            "weighted_f1": rng.uniform(0.4, 0.85, NUM_BASELINES).tolist(),
            "macro_auroc": rng.uniform(0.6, 0.95, NUM_BASELINES).tolist(),
        },
        "stds": {
            "balanced_accuracy": rng.uniform(0.01, 0.05, NUM_BASELINES).tolist(),
            "weighted_f1": rng.uniform(0.01, 0.05, NUM_BASELINES).tolist(),
            "macro_auroc": rng.uniform(0.01, 0.04, NUM_BASELINES).tolist(),
        },
    }


@pytest.fixture
def synthetic_predictions():
    rng = np.random.default_rng(1)
    n = 100
    y_true = rng.integers(0, NUM_CLASSES, size=n)
    y_prob = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for c in range(NUM_CLASSES):
        for d in range(NUM_CLASSES):
            cm[c, d] = rng.integers(0 if c != d else 8, 12 if c == d else 3)
    return {"y_true": y_true, "y_prob": y_prob, "confusion": cm}


# --------------------------------------------------------------------------- #
# Figure 3 — baseline bars
# --------------------------------------------------------------------------- #


def test_fig03_baseline_bars_creates_svg(tmp_path: Path, baseline_metrics) -> None:
    out = tmp_path / "fig03_baseline_bars.svg"
    p = generate_fig03_baseline_bars(
        out_path=out,
        model_names=baseline_metrics["model_names"],
        metric_means=baseline_metrics["means"],
        metric_stds=baseline_metrics["stds"],
    )
    assert p == out
    assert out.exists() and out.stat().st_size > 0
    # SVG must validate as XML.
    tree = ET.parse(out)
    assert tree.getroot().tag.endswith("svg")
    text = out.read_text()
    # Each baseline name appears at least once in the SVG (gid + legend).
    for name in baseline_metrics["model_names"]:
        assert name in text, f"baseline {name!r} not present in SVG"


# --------------------------------------------------------------------------- #
# Figure 4 — graph radar
# --------------------------------------------------------------------------- #


def test_fig04_graph_radar_creates_svg_with_5_polygons(tmp_path: Path) -> None:
    out = tmp_path / "fig04_graph_radar.svg"
    variants = ["spatial_knn", "feature_sim", "dual_edge", "hierarchical", "heterogeneous"]
    axes = ["BACC", "F1", "AUROC", "kappa"]
    rng = np.random.default_rng(0)
    values = rng.uniform(0.5, 0.95, size=(5, 4))
    p = generate_fig04_graph_radar(
        out_path=out, graph_variants=variants, axes=axes, values=values
    )
    assert p == out
    text = out.read_text()
    # Each variant name appears at least once; matches design §5.1 row 4
    # (5 polygons via gid="polygon-<name>").
    for v in variants:
        assert f"polygon-{v}" in text


# --------------------------------------------------------------------------- #
# Figure 5 — confusion matrix
# --------------------------------------------------------------------------- #


def test_fig05_confusion_matrix_creates_png(tmp_path: Path, synthetic_predictions) -> None:
    out = tmp_path / "fig05_confusion.png"
    generate_fig05_confusion_matrix(
        out_path=out, confusion=synthetic_predictions["confusion"], normalize=True
    )
    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"
        # Sanity: image is at least 300×300 (heatmap + ticks + colorbar).
        assert img.size[0] >= 300 and img.size[1] >= 300


def test_fig05_confusion_matrix_rejects_wrong_shape(tmp_path: Path) -> None:
    out = tmp_path / "fig05.png"
    with pytest.raises(ValueError, match=f"\\[{NUM_CLASSES}, {NUM_CLASSES}\\]"):
        generate_fig05_confusion_matrix(out_path=out, confusion=np.zeros((5, 5), dtype=int))


# --------------------------------------------------------------------------- #
# Figure 6 — ROC curves
# --------------------------------------------------------------------------- #


def test_fig06_roc_has_seven_class_curves_plus_macro(
    tmp_path: Path, synthetic_predictions
) -> None:
    out = tmp_path / "fig06_roc.svg"
    generate_fig06_roc(
        out_path=out,
        y_true=synthetic_predictions["y_true"],
        y_prob=synthetic_predictions["y_prob"],
    )
    text = out.read_text()
    # Each of the 7 classes has a gid'd ROC line.
    for c in CLASS_ORDER:
        assert f"roc-{c}" in text, f"ROC curve for {c} missing"
    assert "roc-macro" in text
    assert "roc-diagonal" in text


# --------------------------------------------------------------------------- #
# Figure 8 — t-SNE
# --------------------------------------------------------------------------- #


def test_fig08_tsne_creates_png_with_seven_classes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 70
    labels = np.tile(np.arange(NUM_CLASSES), n // NUM_CLASSES + 1)[:n]
    # Cluster embeddings around per-class means so t-SNE separates them.
    centres = rng.normal(scale=5.0, size=(NUM_CLASSES, 16))
    pre = centres[labels] + rng.normal(scale=0.2, size=(n, 16))
    post = centres[labels] + rng.normal(scale=0.05, size=(n, 16))
    out = tmp_path / "fig08_tsne.png"
    generate_fig08_tsne(
        out_path=out,
        embeddings_pre=pre,
        embeddings_post=post,
        labels=labels,
        seed=0,
        perplexity=10.0,
    )
    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"
        # Expect 7 distinct class colors → many unique RGB tuples in the image.
        rgb = img.convert("RGB").getcolors(maxcolors=2_000_000)
        assert rgb is not None and len(rgb) >= 7


# --------------------------------------------------------------------------- #
# Figure 9 — transfer comparison
# --------------------------------------------------------------------------- #


def test_fig09_transfer_creates_svg_with_five_bars(tmp_path: Path) -> None:
    out = tmp_path / "fig09_transfer.svg"
    settings = ["T1", "T2", "T3", "T4", "T5"]
    generate_fig09_transfer(
        out_path=out,
        settings=settings,
        bacc_means=[0.55, 0.60, 0.66, 0.69, 0.71],
        bacc_stds=[0.02, 0.02, 0.02, 0.02, 0.02],
    )
    text = out.read_text()
    for s in settings:
        assert f"transfer-bar-{s}" in text


# --------------------------------------------------------------------------- #
# Figure S1 — training curves
# --------------------------------------------------------------------------- #


def test_figS1_training_curves_creates_svg(tmp_path: Path) -> None:
    out = tmp_path / "figS1_curves.svg"
    history = {
        "abmil": {
            "train_loss": [1.9, 1.5, 1.2, 0.9, 0.7],
            "val_balanced_accuracy": [0.30, 0.42, 0.55, 0.62, 0.68],
        },
        "vetgigagraph": {
            "train_loss": [1.95, 1.4, 1.0, 0.7, 0.5],
            "val_balanced_accuracy": [0.32, 0.48, 0.62, 0.74, 0.80],
        },
    }
    generate_figS1_training_curves(out_path=out, history_per_run=history)
    text = out.read_text()
    for run_name in history:
        assert f"loss-{run_name}" in text
        assert f"bacc-{run_name}" in text


# --------------------------------------------------------------------------- #
# Deferred figures raise (Phase 12 fixtures not available)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", DEFERRED_FIGURES)
def test_deferred_figures_raise(name: str, tmp_path: Path) -> None:
    fn = FIGURE_REGISTRY[name]
    with pytest.raises(NotImplementedError, match="Phase 12"):
        # Some deferred fns take **kwargs; some don't. Use a permissive call.
        try:
            fn(out_path=tmp_path / f"{name}.svg")
        except TypeError:
            fn(out_path=tmp_path / f"{name}.svg", placeholder=None)


# --------------------------------------------------------------------------- #
# Tables — schema enforcement + CSV/TEX pair
# --------------------------------------------------------------------------- #


def test_table01_dataset(tmp_path: Path) -> None:
    rows = [
        {"class": "Melanoma", "abbrev": c, "n_wsi": 50, "n_patient": 35}
        for c in CLASS_ORDER
    ]
    rows.append({"class": "Total", "abbrev": "ALL", "n_wsi": 350, "n_patient": 282})
    csv, tex = generate_table01_dataset(out_path_no_ext=tmp_path / "table01_dataset", rows=rows)
    assert csv.exists() and tex.exists()
    df = pd.read_csv(csv)
    assert set(df.columns) >= {"class", "abbrev", "n_wsi", "n_patient"}
    assert len(df) == NUM_CLASSES + 1
    assert "\\caption" in tex.read_text()


def test_table02_baseline(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for m in ["abmil", "dsmil", "transmil", "clam_sb", "clam_mb", "vetgigagraph", "random"]:
        scores = rng.uniform(0.4, 0.85, 5)
        rows.append({
            "model": m,
            "fold_0_BACC": scores[0],
            "fold_1_BACC": scores[1],
            "fold_2_BACC": scores[2],
            "fold_3_BACC": scores[3],
            "fold_4_BACC": scores[4],
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "mean_F1": float(rng.uniform(0.4, 0.85)),
            "std_F1": float(rng.uniform(0.01, 0.05)),
            "mean_AUROC": float(rng.uniform(0.6, 0.95)),
            "std_AUROC": float(rng.uniform(0.01, 0.05)),
            "p_value_vs_ours": float(rng.uniform(0, 0.5)),
        })
    csv, tex = generate_table02_baseline(out_path_no_ext=tmp_path / "table02_baseline", rows=rows)
    df = pd.read_csv(csv)
    assert len(df) == 7
    for c in ["model", "mean", "std", "p_value_vs_ours"]:
        assert c in df.columns


def test_table03_graphs(tmp_path: Path) -> None:
    rows = [
        {"graph_variant": v, "mean_BACC": 0.7, "std_BACC": 0.02, "p_value_friedman": 0.001}
        for v in ["spatial_knn", "feature_sim", "dual_edge", "hierarchical", "heterogeneous"]
    ]
    csv, _ = generate_table03_graphs(out_path_no_ext=tmp_path / "table03_graphs", rows=rows)
    df = pd.read_csv(csv)
    assert len(df) == 5


def test_table04_backbones(tmp_path: Path) -> None:
    rows = [
        {"backbone": b, "num_params": 3_000_000, "mean_BACC": 0.7, "std_BACC": 0.02}
        for b in ["gat", "gcn", "graphsage", "gin", "vetgigagraph"]
    ]
    csv, _ = generate_table04_backbones(out_path_no_ext=tmp_path / "table04_backbones", rows=rows)
    df = pd.read_csv(csv)
    assert len(df) == 5


def test_table05_fusion(tmp_path: Path) -> None:
    rows = [
        {"strategy": s, "mean_BACC": 0.7, "std_BACC": 0.02, "learnable_alpha_mean": float("nan")}
        for s in ["gnn_only", "slide_only", "concat", "learnable_weighted", "cross_attention", "gated"]
    ]
    rows[3]["learnable_alpha_mean"] = 0.62  # only learnable_weighted populates this
    csv, _ = generate_table05_fusion(out_path_no_ext=tmp_path / "table05_fusion", rows=rows)
    df = pd.read_csv(csv)
    assert len(df) == 6


def test_table06_transfer(tmp_path: Path) -> None:
    rows = [
        {
            "setting": f"T{i}",
            "pretrain_corpus": "TCGA",
            "finetune_mode": "linear",
            "mean_BACC": 0.6 + 0.04 * i,
            "std_BACC": 0.02,
            "p_value_vs_T1": 0.05 / max(1, i),
        }
        for i in range(1, 6)
    ]
    csv, _ = generate_table06_transfer(out_path_no_ext=tmp_path / "table06_transfer", rows=rows)
    df = pd.read_csv(csv)
    assert len(df) == 5


def test_table07_per_class(tmp_path: Path) -> None:
    rows = [
        {
            "class": c,
            "precision_mean": 0.7,
            "precision_std": 0.02,
            "recall_mean": 0.7,
            "recall_std": 0.02,
            "f1_mean": 0.7,
            "f1_std": 0.02,
        }
        for c in CLASS_ORDER
    ]
    csv, _ = generate_table07_per_class(out_path_no_ext=tmp_path / "table07_per_class", rows=rows)
    df = pd.read_csv(csv)
    assert len(df) == NUM_CLASSES


def test_table08_cost(tmp_path: Path) -> None:
    rows = [
        {
            "model": m,
            "num_params": 3_000_000 if m != "transmil" else 5_000_000,
            "flops": 1_000_000_000,
            "inference_time_per_slide_ms": 200.0,
        }
        for m in ["abmil", "dsmil", "transmil", "clam_sb", "clam_mb", "vetgigagraph", "random"]
    ]
    csv, _ = generate_table08_cost(out_path_no_ext=tmp_path / "table08_cost", rows=rows)
    df = pd.read_csv(csv)
    assert len(df) >= 7


# --------------------------------------------------------------------------- #
# §5.3 reproducibility — re-run is bit-identical
# --------------------------------------------------------------------------- #


def test_table_regen_is_identical(tmp_path: Path) -> None:
    rows = [
        {"graph_variant": v, "mean_BACC": 0.7, "std_BACC": 0.02, "p_value_friedman": 0.001}
        for v in ["spatial_knn", "feature_sim", "dual_edge", "hierarchical", "heterogeneous"]
    ]
    csv1, tex1 = generate_table03_graphs(out_path_no_ext=tmp_path / "t1", rows=rows)
    csv2, tex2 = generate_table03_graphs(out_path_no_ext=tmp_path / "t2", rows=rows)
    assert csv1.read_bytes() == csv2.read_bytes()
    assert tex1.read_bytes() == tex2.read_bytes()


# --------------------------------------------------------------------------- #
# Registry coverage — every figure/table key is callable + locked-name guard
# --------------------------------------------------------------------------- #


def test_figure_registry_locked_names() -> None:
    assert sorted(FIGURE_REGISTRY) == sorted(
        [
            "fig01_pipeline",
            "fig02_graph_variants",
            "fig03_baseline_bars",
            "fig04_graph_radar",
            "fig05_confusion",
            "fig06_roc",
            "fig07_attention",
            "fig08_tsne",
            "fig09_transfer",
            "figS1_curves",
        ]
    )


def test_table_registry_locked_names() -> None:
    assert sorted(TABLE_REGISTRY) == sorted(
        [f"table0{i}_" + s for i, s in enumerate(
            ["dataset", "baseline", "graphs", "backbones", "fusion", "transfer", "per_class", "cost"], start=1
        )]
    )
