# `src/deformable_attention/` — Fu 2025 deformable attention for WSI graphs

This package is the **only new code** in v2. It implements a deformable-attention
GNN that slots into v1's `VetGigaGraph` pipeline via the `src/shared/` symlink.

## Files

| File | Role |
|---|---|
| `kernel.py` | Soft K-NN interpolation at arbitrary 2-D coordinates over the irregular WSI tile cloud. Replaces Fu 2025's bilinear-on-grid sampling. |
| `layer.py` | `DeformableAttentionLayer` — one block: offset MLP → coord-sample → multi-head attention over (static neighbors ∪ K sampled) → residual + LN. |
| `model.py` | `DeformableAttentionStack` (v1 `GnnBackbone`-compatible) + `build_deformable_model` factory that hosts the stack inside v1's `VetGigaGraph`. |
| `__init__.py` | Public re-exports. |

## Adaptation vs the original Fu 2025 paper

Fu et al. assume a *regular 2-D feature map* (typical convolutional backbone).
Our pipeline ships per-tile features from a frozen Prov-GigaPath ViT-G as an
*irregular point cloud* — variable tissue shape, ~3 k–35 k tiles per WSI.
Bilinear sampling on a grid doesn't apply directly. We replace it with a soft
K-nearest-neighbor weighted average in tile-coordinate space:

```
for each node v:
    p_v = (x_v, y_v)
    Δ_1..K = OffsetMLP(h_v, p_v)
    for k in 1..K:
        q_k = p_v + Δ_k
        nbrs = KNN(q_k, all_node_coords, K=knn_k)
        s_k = softmax(-||q_k - nbr.coord||² / T) @ nbr.feat
    attention over (static_neighbors(v) ∪ s_1..s_K)
```

Everything else is held identical to v1 GAT — hidden=128, heads=2, layers=2,
dropout=0.25, dual-edge graph, gnn_only fusion, class-weighted CE, AdamW
(lr=1e-4), CosineAnnealingWarmRestarts, fp16, 5-fold patient-level CV with
locked seeds {42, 123, 456, 789, 1024}.

## How it slots into v1

```
VetGigaGraph (v1, from src/shared/models/vetgigagraph.py)
├── gnn = DeformableAttentionStack(...)   ← v2 (this package)
├── slide_encoder = None                   ← v1 (fusion = gnn_only)
├── fusion = GnnOnlyFusion                 ← v1
└── classifier = MlpClassifier             ← v1
```

The only sharp edge: v1's `VetGigaGraph.forward` doesn't pass `coords` into the
GNN. We monkey-patch the model in `build_deformable_model` to inject
`graph.pos` into the GNN call before each forward. No v1 source files are
modified.

## Risks (mirrored from `docs/02-design-v2.md`)

| Risk | Mitigation |
|---|---|
| Offset MLP collapses to zero ⇒ degenerates to GAT | Small init + offset L2 penalty for first 20 epochs (`offset_penalty_*` config keys). |
| Fp16 instability in offset MLP | `fp32_offset_mlp: true` (default) promotes that one MLP to fp32. |
| O(N²) memory in `coord_knn_sample` | Acceptable up to ~50 k tiles per WSI (~5 GiB fp16). Chunk queries if a slide exceeds this. |
| Inner scatter loop in attention softmax | Documented hotspot — replace with `torch_scatter.scatter_softmax` if it dominates profiles. |

## Running

```bash
# One fold (smoke test)
python scripts/04b_train_deformable.py --fold 0 --config configs/experiment_deformable.yaml

# All 5 folds
bash scripts/run_deformable_cv.sh

# Build v1 vs v2 comparison
python scripts/05b_compare_to_v1.py
```

See `paper/manuscript.md` §3.4 for the formal write-up and `docs/02-design-v2.md`
for the full design rationale and reproducibility checklist.
