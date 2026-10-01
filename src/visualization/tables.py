"""LaTeX/CSV table generators (Phase 9.5).

Each generator writes both a ``.csv`` (source of truth) and a ``.tex``
(auto-generated from the CSV). This split keeps the locked schemas in
``docs/02-design/09-deliverables.md`` §5.2 simple to verify — the
script just reads the CSV.

LaTeX tables follow the booktabs style (no vertical rules) with the
caption referencing the experiment. Pandas' ``DataFrame.to_latex`` is
the rendering engine.

References:
    Design: docs/02-design/09-deliverables.md §2, §5.2
    Tests:  docs/02-design/09-deliverables.md §5.2 schema checks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _save_pair(df, base_path: Path, *, caption: str, label: str) -> tuple[Path, Path]:
    """Write ``base_path.csv`` + ``base_path.tex`` (booktabs)."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    tex_path = base_path.with_suffix(".tex")
    df.to_csv(csv_path, index=False)
    tex = df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        escape=False,
        caption=caption,
        label=label,
    )
    tex_path.write_text(tex, encoding="utf-8")
    return csv_path, tex_path


def _ensure_columns(df, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: input is missing required columns: {missing}")


# --------------------------------------------------------------------------- #
# Table 1 — Dataset summary
# --------------------------------------------------------------------------- #


def generate_table01_dataset(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 1 — CATCH dataset summary (1 row per class + 1 total).

    Required keys per row: ``class, abbrev, n_wsi, n_patient``.
    """
    import pandas as pd

    df = pd.DataFrame(list(rows))
    _ensure_columns(df, ["class", "abbrev", "n_wsi", "n_patient"], "table01_dataset")
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="CATCH dataset summary statistics.",
        label="tab:dataset",
    )


# --------------------------------------------------------------------------- #
# Table 2 — Baseline comparison
# --------------------------------------------------------------------------- #


def generate_table02_baseline(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 2 — baseline 5-fold means/stds + p-value vs ours."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    required = [
        "model",
        "fold_0_BACC", "fold_1_BACC", "fold_2_BACC", "fold_3_BACC", "fold_4_BACC",
        "mean", "std",
        "mean_F1", "std_F1",
        "mean_AUROC", "std_AUROC",
        "p_value_vs_ours",
    ]
    _ensure_columns(df, required, "table02_baseline")
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="Baseline comparison on Experiment 1 (5-fold mean ± std).",
        label="tab:baseline",
    )


# --------------------------------------------------------------------------- #
# Table 3 — Graph structure comparison
# --------------------------------------------------------------------------- #


def generate_table03_graphs(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 3 — five graph variants × BACC mean/std + Friedman p."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    _ensure_columns(
        df,
        ["graph_variant", "mean_BACC", "std_BACC", "p_value_friedman"],
        "table03_graphs",
    )
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="Graph-construction variant comparison (Experiment 2).",
        label="tab:graphs",
    )


# --------------------------------------------------------------------------- #
# Table 4 — GNN backbone comparison
# --------------------------------------------------------------------------- #


def generate_table04_backbones(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 4 — GAT/GCN/SAGE/GIN backbones × params + BACC."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    _ensure_columns(
        df,
        ["backbone", "num_params", "mean_BACC", "std_BACC"],
        "table04_backbones",
    )
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="GNN backbone comparison (Experiment 3).",
        label="tab:backbones",
    )


# --------------------------------------------------------------------------- #
# Table 5 — Fusion strategy comparison
# --------------------------------------------------------------------------- #


def generate_table05_fusion(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 5 — 6 fusion strategies × BACC + learnable α (NaN where N/A)."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    _ensure_columns(
        df,
        ["strategy", "mean_BACC", "std_BACC", "learnable_alpha_mean"],
        "table05_fusion",
    )
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="Fusion strategy comparison (Experiment 4).",
        label="tab:fusion",
    )


# --------------------------------------------------------------------------- #
# Table 6 — Cross-species transfer
# --------------------------------------------------------------------------- #


def generate_table06_transfer(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 6 — T1–T5 settings × BACC + p vs T1."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    _ensure_columns(
        df,
        [
            "setting",
            "pretrain_corpus",
            "finetune_mode",
            "mean_BACC",
            "std_BACC",
            "p_value_vs_T1",
        ],
        "table06_transfer",
    )
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="Cross-species transfer (Experiment 5).",
        label="tab:transfer",
    )


# --------------------------------------------------------------------------- #
# Table 7 — Per-class precision / recall / F1
# --------------------------------------------------------------------------- #


def generate_table07_per_class(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 7 — per-class precision/recall/F1 across 7 classes."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    _ensure_columns(
        df,
        [
            "class",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
        ],
        "table07_per_class",
    )
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="Per-class precision / recall / F1 (best model).",
        label="tab:per_class",
    )


# --------------------------------------------------------------------------- #
# Table 8 — Computational cost
# --------------------------------------------------------------------------- #


def generate_table08_cost(
    *,
    out_path_no_ext: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Table 8 — per-model parameter / FLOP / inference-time cost."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    _ensure_columns(
        df,
        ["model", "num_params", "flops", "inference_time_per_slide_ms"],
        "table08_cost",
    )
    return _save_pair(
        df,
        Path(out_path_no_ext),
        caption="Computational cost comparison.",
        label="tab:cost",
    )
