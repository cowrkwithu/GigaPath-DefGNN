# 20x vs 40x Prov-GigaPath features (349 slides)

| Metric (per-slide median, then median over slides) | Median [IQR] |
|---|---|
| coverage_20x | 0.990 [0.974, 0.997] |
| cos_matched_median | 0.554 [0.486, 0.613] |
| cos_unmatched_median | 0.309 [0.263, 0.357] |
| cos_40x_same_region_median | 0.743 [0.711, 0.782] |
| cos_40x_vs_region_mean_loo_median | 0.803 [0.771, 0.832] |
| cka_linear | 0.875 [0.844, 0.907] |
| retrieval_top1 | 0.196 [0.132, 0.289] |
| norm_ratio_median | 1.089 [1.067, 1.116] |

| Slide-mean probe (outer CV) | Balanced accuracy | n |
|---|---|---|
| train40x_test40x | 0.937 | 349 |
| train40x_test20x | 0.862 | 349 |
| train20x_test40x | 0.851 | 349 |
| train20x_test20x | 0.960 | 349 |
