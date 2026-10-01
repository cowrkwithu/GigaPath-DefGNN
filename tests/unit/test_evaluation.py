"""Unit tests for ``src/evaluation/`` (Phase 8, Module E evaluation side).

Coverage map (design ``docs/02-design/03-architecture.md`` §6.E
evaluation rows + ``docs/02-design/08-statistics-reproducibility.md``
§6 statistical sanity gates):

| # | Design test                              | Status |
|---|------------------------------------------|--------|
| 1 | test_balanced_accuracy_vs_sklearn        | ✅ |
| 2 | test_weighted_f1_vs_sklearn              | ✅ |
| 3 | test_macro_auroc_vs_sklearn              | ✅ |
| 4 | test_cohen_kappa_vs_sklearn              | ✅ |
| 5 | test_per_class_accuracy_sums_to_avg      | ✅ |
| 6 | test_metric_perfect_classification       | ✅ |
| 7 | test_metric_random_classification        | ✅ |
| 8 | test_confusion_matrix_class_order        | ✅ |
| 9 | test_paired_ttest_identical_arrays       | ✅ |
| 10 | test_wilcoxon_identical_arrays          | ✅ |
| 11 | test_mcnemar_identical_classifiers      | ✅ |
| 12 | test_friedman_three_models              | ✅ |
| 13 | test_bonferroni_correction              | ✅ |
| 14 | test_kfold_no_patient_leak              | ✅ |
| 15 | test_kfold_class_coverage               | ✅ |
| +  | test_paired_ttest_recovers_known_effect | ✅ |
| +  | test_mcnemar_inverted_classifier         | ✅ |
| +  | test_insufficient_folds_raises          | ✅ |
| +  | run_validators U/S gates                 | ✅ |
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from src.evaluation import (
    CLASS_ORDER,
    NUM_CLASSES,
    RunMetadata,
    aggregate_severity,
    balanced_accuracy,
    bonferroni_correction,
    cohen_kappa,
    compute_confusion_matrix,
    compute_metrics,
    friedman_nemenyi,
    macro_auroc,
    make_5fold_splits,
    mcnemar,
    paired_ttest,
    per_class_accuracy,
    run_validators,
    verify_splits,
    weighted_f1,
    write_split_csv,
)
from src.evaluation.run_validators import SEVERITY_INVALIDATE, SEVERITY_INVESTIGATE
from src.utils.errors import DataIntegrityError, InsufficientFoldsError


# --------------------------------------------------------------------------- #
# Synthetic prediction fixture
# --------------------------------------------------------------------------- #


def _make_predictions(seed: int = 0, n: int = 70) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred, y_prob)`` with all 7 classes covered."""
    rng = np.random.default_rng(seed)
    # Force at least one of each class.
    base = np.tile(np.arange(NUM_CLASSES), n // NUM_CLASSES + 1)[:n]
    y_true = rng.permutation(base)
    # Pred = true with 30% noise → realistic mid-quality classifier.
    noise_mask = rng.random(n) < 0.3
    y_pred = np.where(noise_mask, rng.integers(0, NUM_CLASSES, size=n), y_true)
    # Probabilities: softmax of one-hot + small noise on the true class.
    y_prob = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    y_prob[np.arange(n), y_pred] = 0.7
    fill = (1.0 - 0.7) / (NUM_CLASSES - 1)
    for i in range(n):
        for c in range(NUM_CLASSES):
            if c != y_pred[i]:
                y_prob[i, c] = fill
    # Add a small jitter so AUROC has variance across classes.
    y_prob += rng.normal(scale=0.01, size=y_prob.shape)
    y_prob = np.clip(y_prob, 1e-9, None)
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    return y_true.astype(np.int64), y_pred.astype(np.int64), y_prob


# --------------------------------------------------------------------------- #
# Metric vs sklearn (8 tests)
# --------------------------------------------------------------------------- #


def test_balanced_accuracy_vs_sklearn() -> None:
    y_true, y_pred, _ = _make_predictions(seed=1)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred), rel=0, abs=0
    )


def test_weighted_f1_vs_sklearn() -> None:
    y_true, y_pred, _ = _make_predictions(seed=2)
    expected = f1_score(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), average="weighted", zero_division=0
    )
    assert weighted_f1(y_true, y_pred) == pytest.approx(expected, rel=0, abs=0)


def test_macro_auroc_vs_sklearn() -> None:
    y_true, _, y_prob = _make_predictions(seed=3)
    expected = roc_auc_score(
        y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(NUM_CLASSES))
    )
    assert macro_auroc(y_true, y_prob) == pytest.approx(expected, rel=0, abs=0)


def test_cohen_kappa_vs_sklearn() -> None:
    y_true, y_pred, _ = _make_predictions(seed=4)
    expected = cohen_kappa_score(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    assert cohen_kappa(y_true, y_pred) == pytest.approx(expected, rel=0, abs=0)


def test_per_class_accuracy_sums_to_avg() -> None:
    """mean(per-class accuracy) over present classes equals balanced accuracy."""
    y_true, y_pred, _ = _make_predictions(seed=5)
    pca = per_class_accuracy(y_true, y_pred)
    pca_present = pca[~np.isnan(pca)]
    assert float(pca_present.mean()) == pytest.approx(
        balanced_accuracy(y_true, y_pred), rel=1e-9
    )


def test_metric_perfect_classification() -> None:
    """y_true == y_pred → BACC=1, F1=1, kappa=1, AUROC=1."""
    y_true = np.tile(np.arange(NUM_CLASSES), 5)
    y_pred = y_true.copy()
    y_prob = np.eye(NUM_CLASSES)[y_pred]
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["balanced_accuracy"] == 1.0
    assert m["weighted_f1"] == 1.0
    assert m["cohen_kappa"] == 1.0
    assert m["macro_auroc"] == 1.0


def test_metric_random_classification() -> None:
    """Uniform random predictions over 7 classes → BACC ≈ 1/7 ± 5%."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, NUM_CLASSES, size=2000)
    y_pred = rng.integers(0, NUM_CLASSES, size=2000)
    bacc = balanced_accuracy(y_true, y_pred)
    assert abs(bacc - 1.0 / NUM_CLASSES) < 0.05, (
        f"random BACC {bacc:.4f} too far from {1.0/NUM_CLASSES:.4f}"
    )


def test_confusion_matrix_class_order() -> None:
    """Confusion matrix axes follow the locked order [MEL, MCT, SCC, PNST, PLC, TRB, HIS]."""
    assert CLASS_ORDER == ("MEL", "MCT", "SCC", "PNST", "PLC", "TRB", "HIS")
    # Diagonal entries equal per-class true counts when y_true == y_pred.
    y_true = np.tile(np.arange(NUM_CLASSES), 3)
    y_pred = y_true.copy()
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    np.testing.assert_array_equal(np.diag(cm), np.full(NUM_CLASSES, 3))


# --------------------------------------------------------------------------- #
# Statistical tests (8+ tests)
# --------------------------------------------------------------------------- #


def test_paired_ttest_identical_arrays() -> None:
    a = np.array([0.7, 0.72, 0.71, 0.73, 0.69])
    res = paired_ttest(a, a)
    assert res.p > 0.99


def test_paired_ttest_recovers_known_effect() -> None:
    """Clear positive shift → p < 0.001."""
    a = np.array([0.70, 0.72, 0.71, 0.73, 0.69, 0.71, 0.70])
    b = a + 0.10
    res = paired_ttest(a, b)
    assert res.p < 0.001


def test_wilcoxon_identical_arrays() -> None:
    a = np.array([0.7, 0.72, 0.71, 0.73, 0.69])
    res = wilcoxon_alias(a, a)
    assert res.p > 0.99


def wilcoxon_alias(a, b):
    """Wrap to keep the test name aligned with design row 10 verbatim."""
    from src.evaluation import wilcoxon

    return wilcoxon(a, b)


def test_mcnemar_identical_classifiers() -> None:
    """Identical predictions → p > 0.99."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, NUM_CLASSES, size=80)
    y_pred = rng.integers(0, NUM_CLASSES, size=80)
    res = mcnemar(y_true, y_pred, y_pred)
    assert res.p > 0.99


def test_mcnemar_inverted_classifier() -> None:
    """One classifier always right, the other always wrong → p < 0.001 (large N)."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, NUM_CLASSES, size=n)
    y_a = y_true.copy()  # always correct
    y_b = (y_true + 1) % NUM_CLASSES  # always wrong
    res = mcnemar(y_true, y_a, y_b)
    assert res.p < 0.001


def test_friedman_three_models() -> None:
    """Friedman on three clearly different score arrays → p < 0.001."""
    weak = [0.30, 0.31, 0.29, 0.32, 0.28, 0.30, 0.29, 0.31]
    medium = [0.55, 0.56, 0.54, 0.57, 0.53, 0.55, 0.54, 0.56]
    strong = [0.85, 0.86, 0.84, 0.87, 0.83, 0.85, 0.84, 0.86]
    res = friedman_nemenyi([weak, medium, strong])
    assert res.p < 0.001


def test_bonferroni_correction() -> None:
    """For 4 tests at α=0.05, corrected α = 0.0125."""
    assert bonferroni_correction(0.05, 4) == pytest.approx(0.0125, rel=0)
    with pytest.raises(ValueError):
        bonferroni_correction(0.0, 4)
    with pytest.raises(ValueError):
        bonferroni_correction(0.05, 0)


def test_insufficient_folds_raises() -> None:
    """Paired tests on N<5 must raise InsufficientFoldsError."""
    short = np.array([0.7, 0.72, 0.71])  # 3 obs
    with pytest.raises(InsufficientFoldsError):
        paired_ttest(short, short)


# --------------------------------------------------------------------------- #
# K-fold CV runner + verifier
# --------------------------------------------------------------------------- #


def _toy_slide_df(n_per_class: int = 30) -> pd.DataFrame:
    """Synthetic CATCH-shaped manifest: 7 classes × n patients × 1–2 slots.

    ``n_per_class`` must be large enough for StratifiedGroupKFold to
    reliably keep every class represented in every val/test fold across
    the inner 7-fold val carve-out. 30 patients/class (≈ 6 per outer
    fold) is the empirical sweet spot — much smaller and the design's
    own warning in `08-statistics-reproducibility.md` §4.2 (one class
    missing under unlucky seeds) starts firing on these synthetic
    fixtures even when the real CATCH-50-per-class would not.
    """
    rows = []
    for c in CLASS_ORDER:
        for i in range(n_per_class):
            patient = f"{c}_{i:02d}"
            for slot in (1, 2) if i < 3 else (1,):
                rows.append(
                    {
                        "slide_id": f"{c}_{i:02d}_{slot}",
                        "patient_id": patient,
                        "tumor_class": c,
                    }
                )
    return pd.DataFrame(rows)


def test_make_5fold_splits_schema(tmp_path: Path) -> None:
    df = _toy_slide_df(n_per_class=30)
    splits = make_5fold_splits(df)
    assert set(splits.columns) == {"slide_id", "patient_id", "tumor_class", "fold", "split"}
    assert sorted(splits["fold"].unique()) == [0, 1, 2, 3, 4]
    assert set(splits["split"].unique()).issubset({"train", "val", "test"})


def test_kfold_no_patient_leak(tmp_path: Path) -> None:
    df = _toy_slide_df(n_per_class=30)
    splits = make_5fold_splits(df)
    csv_path = tmp_path / "cv5fold.csv"
    write_split_csv(splits, csv_path)
    # Should not raise:
    verify_splits(csv_path)


def test_verify_splits_detects_leak(tmp_path: Path) -> None:
    df = _toy_slide_df(n_per_class=30)
    splits = make_5fold_splits(df)
    # Inject a leak: copy fold-0 train rows into fold-0 val.
    leak_row = splits[(splits["fold"] == 0) & (splits["split"] == "train")].iloc[0].copy()
    leak_row["split"] = "val"
    bad = pd.concat([splits, pd.DataFrame([leak_row])], ignore_index=True)
    csv_path = tmp_path / "leaky.csv"
    write_split_csv(bad, csv_path)
    with pytest.raises(DataIntegrityError, match="leak"):
        verify_splits(csv_path)


def test_kfold_class_coverage(tmp_path: Path) -> None:
    """Every class appears in val and test of every fold (StratifiedGroupKFold contract)."""
    df = _toy_slide_df(n_per_class=30)
    splits = make_5fold_splits(df)
    for fold in sorted(splits["fold"].unique()):
        for split in ("val", "test"):
            sub = splits[(splits["fold"] == fold) & (splits["split"] == split)]
            classes_present = set(sub["tumor_class"].unique())
            assert classes_present == set(CLASS_ORDER), (
                f"fold {fold} {split} missing classes: {set(CLASS_ORDER) - classes_present}"
            )


def test_verify_splits_missing_class(tmp_path: Path) -> None:
    """Drop one class from a fold's val → verify_splits flags it."""
    df = _toy_slide_df(n_per_class=30)
    splits = make_5fold_splits(df)
    # Remove all MEL rows from fold-0 val.
    bad = splits[~((splits["fold"] == 0) & (splits["split"] == "val") & (splits["tumor_class"] == "MEL"))]
    csv_path = tmp_path / "missing_class.csv"
    write_split_csv(bad, csv_path)
    with pytest.raises(DataIntegrityError, match="missing class"):
        verify_splits(csv_path)


# --------------------------------------------------------------------------- #
# Run validators
# --------------------------------------------------------------------------- #


def _good_metrics() -> dict:
    return {
        "balanced_accuracy": 0.85,
        "weighted_f1": 0.84,
        "macro_auroc": 0.95,
        "cohen_kappa": 0.82,
        "per_class_accuracy": np.array([0.85, 0.82, 0.88, 0.83, 0.86, 0.84, 0.87]),
    }


def test_run_validators_all_pass(tmp_path: Path) -> None:
    df = _toy_slide_df()
    splits = make_5fold_splits(df)
    csv_path = tmp_path / "cv.csv"
    write_split_csv(splits, csv_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"\x00" * 64)

    meta = RunMetadata(
        config_logged=True,
        seed_was_set=True,
        fold=0,
        splits_csv=csv_path,
        metrics=_good_metrics(),
        checkpoint_path=ckpt,
        training_finished=True,
        train_bacc=0.90,
        predictions=[0, 1, 2, 3, 4, 5, 6],
    )
    results = run_validators(meta)
    failed = [r for r in results if not r.passed]
    assert failed == [], f"unexpected gate failures: {[(r.name, r.detail) for r in failed]}"
    assert aggregate_severity(results) == "pass"


def test_run_validators_invalidates_on_seed_not_set(tmp_path: Path) -> None:
    df = _toy_slide_df()
    splits = make_5fold_splits(df)
    csv_path = tmp_path / "cv.csv"
    write_split_csv(splits, csv_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"\x00" * 64)
    meta = RunMetadata(
        config_logged=True,
        seed_was_set=False,  # U2 fail
        fold=0,
        splits_csv=csv_path,
        metrics=_good_metrics(),
        checkpoint_path=ckpt,
        training_finished=True,
        train_bacc=0.85,
        predictions=[0, 1, 2, 3, 4, 5, 6],
    )
    results = run_validators(meta)
    u2 = next(r for r in results if r.name == "U2")
    assert not u2.passed
    assert u2.severity == SEVERITY_INVALIDATE
    assert aggregate_severity(results) == SEVERITY_INVALIDATE


def test_run_validators_investigates_low_bacc(tmp_path: Path) -> None:
    df = _toy_slide_df()
    splits = make_5fold_splits(df)
    csv_path = tmp_path / "cv.csv"
    write_split_csv(splits, csv_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"\x00" * 64)
    bad_metrics = _good_metrics()
    bad_metrics["balanced_accuracy"] = 0.10  # below S1 floor (0.163)
    meta = RunMetadata(
        config_logged=True,
        seed_was_set=True,
        fold=0,
        splits_csv=csv_path,
        metrics=bad_metrics,
        checkpoint_path=ckpt,
        training_finished=True,
        train_bacc=0.12,
        predictions=[0, 1, 2, 3, 4, 5, 6],
    )
    results = run_validators(meta)
    s1 = next(r for r in results if r.name == "S1")
    assert not s1.passed
    assert s1.severity == SEVERITY_INVESTIGATE
    # No U-gate failure means severity is "investigate", not "invalidate".
    assert aggregate_severity(results) == SEVERITY_INVESTIGATE
