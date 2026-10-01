"""Phase 6 — Baseline MIL models.

Each baseline follows the same calling convention:

    forward(tile_embeddings: Tensor[N, 1536]) -> Tensor[num_classes]

so they are drop-in-comparable to :class:`src.models.VetGigaGraph` for
the experiment matrix in ``docs/02-design/04-experiment-design.md``.
The :data:`BASELINE_REGISTRY` maps the locked names used by
``configs/default.yaml model.name`` and the upcoming
``scripts/04_train.py --model`` flag to their classes.

References:
    Design: docs/02-design/03-architecture.md §6 (Phase 6 row)
    Tests:  docs/02-design/03-architecture.md §5.D rows 10–13
"""

from src.models.baselines.abmil import ABMIL
from src.models.baselines.acmil import ACMIL
from src.models.baselines.clam import CLAM_MB, CLAM_MB_Inst, CLAM_SB, CLAM_SB_Inst
from src.models.baselines.dsmil import DSMIL
from src.models.baselines.transmil import TransMIL
from src.models.baselines.wikg import WiKG

#: Locked names — match `configs/default.yaml model.name` literal.
BASELINE_REGISTRY: dict[str, type] = {
    "abmil": ABMIL,
    "dsmil": DSMIL,
    "transmil": TransMIL,
    "clam_sb": CLAM_SB,
    "clam_mb": CLAM_MB,
    "clam_sb_inst": CLAM_SB_Inst,
    "clam_mb_inst": CLAM_MB_Inst,
    "acmil": ACMIL,
    "wikg": WiKG,
}

__all__ = [
    "ABMIL",
    "ACMIL",
    "BASELINE_REGISTRY",
    "CLAM_MB",
    "CLAM_MB_Inst",
    "CLAM_SB",
    "CLAM_SB_Inst",
    "DSMIL",
    "TransMIL",
    "WiKG",
]
