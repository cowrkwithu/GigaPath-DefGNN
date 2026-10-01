"""Cross-validation split generation + integrity verifier (Phase 8.2).

Generates the locked patient-level + class-stratified 5-fold split
defined in ``docs/02-design/08-statistics-reproducibility.md`` §4.3
using ``StratifiedGroupKFold`` (groups = patient_id, y = tumor_class)
with the locked seed list :data:`src.utils.seed.LOCKED_FOLD_SEEDS`.

After the outer 5-fold test partition, an inner 7-fold split carves
out ~14% of the train+val patients as val (also stratified by class
and grouped by patient).

The locked CSV schema (`08-statistics-reproducibility.md` §4.5):

    | slide_id | patient_id | tumor_class | fold | split |

``fold`` ∈ {0..4}, ``split`` ∈ {train, val, test}.

:func:`verify_splits` enforces the 5 integrity assertions in
`08-statistics-reproducibility.md` §4.4 and is the gate the smoke
script and the U4 validator both call.

References:
    Design: docs/02-design/08-statistics-reproducibility.md §4
    Tests:  docs/02-design/03-architecture.md §6.E (kfold_no_patient_leak,
            kfold_class_coverage)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from src.utils.errors import DataIntegrityError
from src.utils.io_utils import LABEL_TO_INT
from src.utils.seed import LOCKED_FOLD_SEEDS

logger = logging.getLogger(__name__)

#: Locked CSV column order.
SPLIT_CSV_COLUMNS: tuple[str, ...] = (
    "slide_id",
    "patient_id",
    "tumor_class",
    "fold",
    "split",
)

#: Locked split values.
SPLITS = ("train", "val", "test")

#: Inner split size (carve val out of train+val).
INNER_K = 7  # 1/7 ≈ 14.3% — close to the 15% target in design §4.3


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def make_5fold_splits(
    slide_df,
    *,
    seed: int = LOCKED_FOLD_SEEDS[0],
    seeds: Sequence[int] = LOCKED_FOLD_SEEDS,  # noqa: ARG001 — kept for API compat
    inner_k: int = INNER_K,
):
    """Generate the locked canonical 5-fold split as a long-format DataFrame.

    The canonical CV uses **one** outer ``StratifiedGroupKFold`` (single
    seed) so that the 5 test partitions are mutually disjoint and their
    union covers every slide exactly once — the invariant
    ``08-statistics-reproducibility.md`` §4.4 requires
    (``⋃ test_f == all_slides``).

    The other locked seeds (``LOCKED_FOLD_SEEDS[1:]``) are reserved for
    *seed-repetition* analyses (re-running the full CV under a different
    partition to estimate split-induced variance) — those produce
    independent CSVs, not folds within one CSV.

    Args:
        slide_df: pandas DataFrame with at least ``slide_id``,
            ``patient_id``, ``tumor_class`` columns.
        seed: Outer-CV random seed. Default is the canonical
            ``LOCKED_FOLD_SEEDS[0] = 42``.
        seeds: Kept for backward-compat — not consumed by the canonical
            split. Pass ``seeds=[s]`` to override the canonical seed via
            the legacy keyword.
        inner_k: Inner stratified split size (default = ``INNER_K`` = 7,
            yielding ~14% of trainval as val). Smoke fixtures with
            small per-class counts may need ``inner_k=5`` so each val
            partition has enough samples per class.

    Returns:
        DataFrame in :data:`SPLIT_CSV_COLUMNS` order with one row per
        ``(slide_id, fold)`` pair. ``len(out) == 5 * len(slide_df)``.

    Raises:
        DataIntegrityError: if required columns are missing.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    required = {"slide_id", "patient_id", "tumor_class"}
    missing = required - set(slide_df.columns)
    if missing:
        raise DataIntegrityError(
            f"slide_df missing required columns: {sorted(missing)}"
        )

    df = slide_df.reset_index(drop=True).copy()
    y = df["tumor_class"].map(LABEL_TO_INT).to_numpy()
    if np.isnan(y).any() or (y < 0).any():
        bad = df.loc[df["tumor_class"].map(LABEL_TO_INT).isna(), "tumor_class"].unique()
        raise DataIntegrityError(
            f"unknown tumor_class values: {sorted(bad.tolist())}; "
            f"expected one of {sorted(LABEL_TO_INT.keys())}"
        )
    groups = df["patient_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
    outer_splits = list(sgkf.split(df, y=y, groups=groups))

    rows: list[dict] = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(outer_splits):
        # Inner stratified group split: hold out 1/inner_k of trainval as val.
        # Use the outer seed offset by fold_idx so each fold's val is deterministic.
        inner = StratifiedGroupKFold(
            n_splits=int(inner_k), shuffle=True, random_state=int(seed) + fold_idx
        )
        first = next(
            inner.split(
                df.iloc[trainval_idx],
                y=y[trainval_idx],
                groups=groups[trainval_idx],
            )
        )
        val_local_idx = first[1]
        val_idx = trainval_idx[val_local_idx]
        train_idx = np.setdiff1d(trainval_idx, val_idx)

        for split_name, split_idx in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            for i in split_idx:
                rec = df.iloc[int(i)]
                rows.append(
                    {
                        "slide_id": str(rec["slide_id"]),
                        "patient_id": str(rec["patient_id"]),
                        "tumor_class": str(rec["tumor_class"]),
                        "fold": int(fold_idx),
                        "split": split_name,
                    }
                )
    return pd.DataFrame(rows, columns=list(SPLIT_CSV_COLUMNS))


def write_split_csv(splits_df, out_path: Union[str, Path]) -> Path:
    """Write the locked-schema CSV (atomic via temp + rename)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    splits_df[list(SPLIT_CSV_COLUMNS)].to_csv(tmp, index=False)
    tmp.replace(out)
    return out


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def verify_splits(
    splits_csv: Union[str, Path],
    *,
    expected_folds: Sequence[int] = (0, 1, 2, 3, 4),
    expected_classes: Sequence[str] = tuple(LABEL_TO_INT.keys()),
    raise_on_error: bool = True,
) -> Optional[list[str]]:
    """Run all 5 integrity assertions from design §4.4.

    Args:
        splits_csv: Path to the split CSV.
        expected_folds: Default ``[0..4]``.
        expected_classes: Default = locked 7 class labels.
        raise_on_error: When ``True`` (default), raise
            :class:`DataIntegrityError` on the first failure. When
            ``False``, return the list of error strings.

    Returns:
        ``None`` (or empty list if ``raise_on_error=False``) on success.
        On failure with ``raise_on_error=False``, a non-empty list of
        error messages.
    """
    import pandas as pd

    df = pd.read_csv(splits_csv)
    errors: list[str] = []

    missing_cols = set(SPLIT_CSV_COLUMNS) - set(df.columns)
    if missing_cols:
        msg = f"split CSV missing columns: {sorted(missing_cols)}"
        if raise_on_error:
            raise DataIntegrityError(msg)
        errors.append(msg)

    folds_present = sorted(df["fold"].unique().tolist())
    if folds_present != list(expected_folds):
        errors.append(
            f"folds mismatch: expected {list(expected_folds)}, got {folds_present}"
        )

    for fold in folds_present:
        train_p = set(df[(df["fold"] == fold) & (df["split"] == "train")]["patient_id"])
        val_p = set(df[(df["fold"] == fold) & (df["split"] == "val")]["patient_id"])
        test_p = set(df[(df["fold"] == fold) & (df["split"] == "test")]["patient_id"])

        if train_p & val_p:
            errors.append(f"fold {fold}: train ∩ val patient leak ({len(train_p & val_p)} patients)")
        if train_p & test_p:
            errors.append(f"fold {fold}: train ∩ test patient leak ({len(train_p & test_p)} patients)")
        if val_p & test_p:
            errors.append(f"fold {fold}: val ∩ test patient leak ({len(val_p & test_p)} patients)")

        for c in expected_classes:
            for split_name in ("val", "test"):
                count = (
                    (df["fold"] == fold)
                    & (df["split"] == split_name)
                    & (df["tumor_class"] == c)
                ).sum()
                if count < 1:
                    errors.append(
                        f"fold {fold} {split_name} missing class {c} (count={count})"
                    )

    # Aggregate test coverage = every slide appears in some fold's test.
    test_slides = set(df[df["split"] == "test"]["slide_id"])
    all_slides = set(df["slide_id"])
    if test_slides != all_slides:
        errors.append(
            f"5-fold test sets cover {len(test_slides)}/{len(all_slides)} slides"
        )

    if errors:
        msg = "verify_splits failed:\n  - " + "\n  - ".join(errors)
        if raise_on_error:
            raise DataIntegrityError(msg)
        return errors

    logger.info(
        "verify_splits OK: %d folds, no patient leakage, all %d classes present in val and test.",
        len(folds_present),
        len(expected_classes),
    )
    return [] if not raise_on_error else None
