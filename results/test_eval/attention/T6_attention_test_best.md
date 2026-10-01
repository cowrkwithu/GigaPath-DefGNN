# Table T6 — Attention–Annotation Alignment (n WSIs with > 0 GT tumor tiles)

Per-WSI metrics computed via the v1 evaluator `src.evaluation.attention_iou.compute_attention_iou`:
  * **IoU**: between top-K binarized attention map and CATCH polygon tumor mask, K = #GT tumor tiles.
  * **Pearson**: continuous attention vs binary tumor mask.
  * **AUC-PR**: attention as a tumor-vs-non-tumor classifier.

Subset (n_tumor_gt > 0): v1 n = 275, v2 n = 275, paired n = 275.

| Metric | v1 GAT | v2 Deformable | Δ |
|---|---|---|---|
| IoU (mean ± std)          | 0.391 ± 0.370 | 0.381 ± 0.360 | -0.009 |
| Pearson (median)           | 0.224 | 0.226 | +0.002 |
| AUC-PR (mean)              | 0.477 | 0.470 | -0.007 |

**Significance**:
  * Wilcoxon (v1 Pearson vs 0): p = 0.0000 (median 0.224)
  * Wilcoxon (v2 Pearson vs 0): p = 0.0000 (median 0.226)
  * Paired Wilcoxon (v2 IoU − v1 IoU on common WSIs): p = 0.1075, mean Δ IoU = -0.0095

## Per-class IoU

| Class | n | v1 IoU (mean ± std) | v2 IoU (mean ± std) | Δ |
|---|---|---|---|---|
| HIS | 37 | 0.335 ± 0.292 | 0.318 ± 0.271 | -0.017 |
| MCT | 30 | 0.487 ± 0.437 | 0.486 ± 0.439 | -0.001 |
| MEL | 41 | 0.392 ± 0.344 | 0.384 ± 0.329 | -0.009 |
| PLC | 47 | 0.316 ± 0.326 | 0.306 ± 0.320 | -0.010 |
| PNST | 32 | 0.591 ± 0.443 | 0.592 ± 0.442 | +0.001 |
| SCC | 44 | 0.310 ± 0.289 | 0.296 ± 0.255 | -0.014 |
| TRB | 44 | 0.385 ± 0.391 | 0.373 ± 0.378 | -0.012 |
