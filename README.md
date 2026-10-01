# Benchmarking Deformable-Attention Graph Networks and Other Aggregators on Prov-GigaPath Features for Canine Cutaneous Tumor Classification

On the public [CATCH dataset](https://doi.org/10.7937/TCIA.2M93-FX66)
(350 canine cutaneous tumor WSIs, seven subtypes), 14 slide-level aggregators
are trained on the same frozen Prov-GigaPath tile embeddings, patient-level
five-fold splits, and training budgets, and evaluated on the held-out test
partitions: MIL poolings (ABMIL, DSMIL, TransMIL, CLAM-SB/MB with and without
the instance-clustering loss, ACMIL), static- and dynamic-graph GNNs (GCN,
GraphSAGE, GAT, GIN, WiKG), and **GigaPath-DefGNN**, a coordinate-aware
deformable graph-attention aggregator adapted to tile point clouds of up to
~94,000 tiles.

## What is in this repo

- [`src/`](src/) — the Python package: preprocessing (40× and 20× tiling),
  Prov-GigaPath feature extraction, dual-edge graph construction, the GNN
  backbones, the MIL baselines (including ACMIL and WiKG under
  [`src/models/baselines/`](src/models/baselines/)), the deformable-attention
  layer under [`src/deformable_attention/`](src/deformable_attention/), and the
  training and evaluation harness.
- [`scripts/`](scripts/) — entry points:
  - preprocessing and features: `01_…` to `03_…`, `25_extract_feature_knn_graphs.py`;
  - training: `04_train.py`, `04b_train_deformable.py`, `run_deformable_cv.sh`;
  - cross-validation evaluation and ablation sweeps: `05_…` to `15_…`;
  - parameter counts and extra statistics: `16_param_counts.py`, `17_extra_stats.py`;
  - held-out evaluation (revision): `18_test_eval.py` (every model under three
    checkpoint rules) and `19_test_stats.py` (patient-clustered bootstrap CIs,
    exact McNemar tests with Holm correction, 20× vs 40× comparison);
  - revision analyses: `20_complexity.py` (parameters, FLOPs, latency),
    `21_offset_stats.py` (learned offsets), `22_magnification_features.py`
    (20× vs 40× feature shift), `23_gin_loss_curves.py`, `24_revision_figures.py`;
  - job orchestration: `rev1_queue.sh`, `rev1_eval_chain.sh`.
- [`configs/`](configs/) — Hydra configs; [`configs/rev1/`](configs/rev1/) holds
  the retrained GCN/GIN runs and the GIN diagnostics (fp32, LayerNorm, mean).
- [`tests/`](tests/) — unit and integration tests (301 tests).
- [`results/`](results/) — frozen experiment outputs, including
  [`results/test_eval/`](results/test_eval/) (per-WSI held-out predictions,
  [`test_stats.json`](results/test_eval/test_stats.json),
  [`test_tables.md`](results/test_eval/test_tables.md), attention–annotation
  alignment, offset statistics), [`results/complexity/`](results/complexity/),
  [`results/magnification/`](results/magnification/), the MIL baselines, and
  the cross-validation ablation sweeps.
- [`paper/figures/`](paper/figures/) — Figures 1–6 of the revised manuscript
  at 600 dpi and, in [`paper/figures/panels/`](paper/figures/panels/), the
  per-WSI panels behind Figure 5.
- [`notebooks/`](notebooks/) — analysis notebooks.

## Quick start

```bash
# 1. Recreate data symlinks (features / graphs / splits) into local data/ paths.
bash scripts/00_link_v1_artifacts.sh

# 2. Train the deformable-attention model (single fold, smoke test).
python scripts/04b_train_deformable.py \
    --config-name experiment_deformable \
    fold=0

# 3. Run all 5 folds (locked seeds {42, 123, 456, 789, 1024}).
bash scripts/run_deformable_cv.sh

# 4. Evaluate every trained model on the held-out test partitions, then
#    compute the statistics and tables of the revised manuscript.
python scripts/18_test_eval.py
python scripts/19_test_stats.py
```

## Dependencies

PyTorch + PyTorch Lightning + PyTorch Geometric + `torch_cluster` + Hydra +
W&B + OpenSlide. See [`requirements.txt`](requirements.txt). No additional
packages are needed for the deformable-attention implementation (it uses
standard PyG primitives).

## Main results

Held-out balanced accuracy, pooled out-of-fold test predictions over all 350
WSIs, best-validation checkpoints, with patient-clustered bootstrap 95% CIs
(from [`results/test_eval/test_tables.md`](results/test_eval/test_tables.md)):

| Model | Balanced accuracy [95% CI] |
|---|---|
| GraphSAGE | 0.960 [0.935, 0.981] |
| CLAM-SB | 0.954 [0.928, 0.977] |
| CLAM-MB (instance loss) | 0.951 [0.926, 0.974] |
| ABMIL | 0.949 [0.921, 0.973] |
| CLAM-MB | 0.949 [0.921, 0.972] |
| GAT | 0.946 [0.918, 0.970] |
| CLAM-SB (instance loss) | 0.943 [0.912, 0.969] |
| **GigaPath-DefGNN** | **0.940 [0.910, 0.967]** |
| ACMIL | 0.937 [0.906, 0.964] |
| GCN | 0.934 [0.905, 0.960] |
| DSMIL | 0.934 [0.904, 0.962] |
| TransMIL | 0.914 [0.879, 0.945] |
| WiKG | 0.914 [0.881, 0.942] |
| GIN | 0.889 [0.851, 0.922] |

- The aggregators saturate: GigaPath-DefGNN does not differ significantly from
  any baseline (exact McNemar tests, smallest Holm-adjusted p = 0.148), and its
  attention maps match the tumor annotations no better than GAT's.
- Evaluation design matters as much as the aggregator: validation estimates
  exceed held-out accuracy by up to 0.066, and the checkpoint-selection rule
  alone changes held-out accuracy by up to 0.037 and reorders the models
  (GigaPath-DefGNN ranks eighth with best-validation checkpoints, first with
  the latest-top-3 rule, and second with the final weights of a re-run).
- The GCN and GIN results are from runs retrained with a corrected training
  safeguard; the original safeguard stopped half-precision runs at the first
  gradient overflow.

## License

Code is released under the [MIT License](LICENSE).
