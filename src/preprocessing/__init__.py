"""Stage 1 — WSI preprocessing.

Public surface re-exported for use from scripts and tests:

* :class:`WSITiler` — orchestrator: WSI → tiles + coords.csv + metadata.json.
* :class:`OpenSlideWSI` — thin OpenSlide reader wrapper.
* :class:`TileQualityFilter`, :class:`TileQualityMetrics`.
* :class:`StainNormalizer`.
* :func:`detect_tissue`, :func:`tissue_ratio`.

See ``docs/02-design/03-architecture.md`` §2 for the design contract.
"""

from src.preprocessing.quality_filter import TileQualityFilter, TileQualityMetrics
from src.preprocessing.stain_normalizer import StainNormalizer
from src.preprocessing.tissue_detector import detect_tissue, tissue_ratio
from src.preprocessing.wsi_tiler import (
    COORDS_CSV_COLUMNS,
    OpenSlideWSI,
    TileRecord,
    WSITiler,
)

__all__ = [
    "COORDS_CSV_COLUMNS",
    "OpenSlideWSI",
    "StainNormalizer",
    "TileQualityFilter",
    "TileQualityMetrics",
    "TileRecord",
    "WSITiler",
    "detect_tissue",
    "tissue_ratio",
]
