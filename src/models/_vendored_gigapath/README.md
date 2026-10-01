# Vendored `gigapath` (slide encoder)

This directory contains a vendored copy of the GigaPath slide encoder
components from `prov-gigapath/prov-gigapath` (GitHub), needed by
`src/models/gigapath_slide.py::production_slide_backbone_loader`.

## Upstream pin

| Field | Value |
|---|---|
| Repo | https://github.com/prov-gigapath/prov-gigapath |
| Commit SHA | `3505f87e197d167522be491bb3f18fb5a08ca584` |
| Vendored on | 2026-05-15 |
| License | Apache 2.0 (see LICENSE-vendored) |

## What's vendored

Verbatim from upstream `gigapath/` tree:

- `slide_encoder.py` — LongNetViT model + `create_model` factory + `@register_model` factories
- `pos_embed.py` — 2D sincos positional embeddings (MAE-style)
- `torchscale/` — full subtree (architecture, component, model) — LongNet attention + encoder/decoder blocks

## Adaptations from upstream

1. `fairscale.nn.{checkpoint_wrapper,wrap}` imports → replaced with no-op identity stubs (`_fairscale_stubs.py`). Used by upstream only when `args.checkpoint_activations` or `args.fsdp` is True — both default False for the slide-encoder use case, so the stubs are never actually invoked.
2. `xformers.ops.fmha` imports → wrapped in try/except. Only used in `torchscale/component/{flash_attention,custom_flash_attention}.py` (opt-in fast-paths); LongNet itself uses `DilatedAttention` which doesn't go through xformers.
3. `from torchscale.X import Y` (top-level absolute imports) → `from .torchscale.X import Y` (package-relative) — for compatibility with our `src.models._vendored_gigapath` package path.
4. Hard-coded `sys.path.append(...)` at top of `torchscale/model/LongNet.py` removed — the file now imports its siblings via relative imports.

## Why vendored (not installed)

The upstream `gigapath` package is not on PyPI, and its GitHub repo has no `setup.py`. Its dependencies pin `xformers==0.0.18` which is incompatible with `torch>=2.1` (our env). Vendoring + dep substitution is the cleanest cross-section.

## NOT vendored

- `gigapath/preprocessing/` — WSI tiling utilities, not needed for slide encoder
- `gigapath/pipeline.py` — inference orchestration, we have our own
- `gigapath/classification_head.py` — separate classifier, not needed

See `docs/02-design/features/slide-encoder-injection.design.md` for full design context.
