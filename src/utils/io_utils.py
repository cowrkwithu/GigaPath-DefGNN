"""IO helpers: HDF5 (features), PyG (graphs), CSV (splits/manifest), slide-filename parsing.

The slide-filename parser is the **single source of truth** for converting
raw-archive paths to ``slide_id`` and ``patient_id``. Every CV split, log
row, and checkpoint name flows from this function.

References:
    Design: docs/02-design/02-data-spec.md §0 (Storage Layout, Slide-ID derivation)
    Design: docs/02-design/03-architecture.md §3.B (HDF5 schema)
    Design: docs/02-design/features/vetgigagraph.design.md §3.3
"""

from __future__ import annotations

import os

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from src.utils.env import output_dir
from src.utils.errors import DataIntegrityError

logger = logging.getLogger(__name__)


# Locked mapping from raw class-directory name to label abbreviation.
# Mirror of `paths.class_dir_to_label` in configs/default.yaml — duplicated
# here so io_utils has no config dependency. Kept in sync by the test in
# tests/unit/test_io_utils.py::test_class_map_matches_config.
CLASS_DIR_TO_LABEL: dict[str, str] = {
    "Melanoma": "MEL",
    "MCT": "MCT",
    "SCC": "SCC",
    "PNST": "PNST",
    "Plasmacytoma": "PLC",
    "Trichoblastoma": "TRB",
    "Histiocytoma": "HIS",
}

LABEL_TO_INT: dict[str, int] = {
    "MEL": 0, "MCT": 1, "SCC": 2, "PNST": 3, "PLC": 4, "TRB": 5, "HIS": 6,
}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}

#: Filename pattern: <ClassDir>_<PatientID>_<SlotIdx>.svs
SLIDE_FILENAME_RE = re.compile(r"^(?P<klass>[A-Za-z]+)_(?P<patient>\d+)_(?P<slot>\d+)\.svs$")


@dataclass(frozen=True)
class SlideMetadata:
    """Parsed identifiers for a single CATCH WSI."""

    slide_id: str        # e.g. "MEL_01_1"
    patient_id: str      # e.g. "MEL_01" — same patient across slots
    tumor_class: str     # one of {MEL, MCT, SCC, PNST, PLC, TRB, HIS}
    label: int           # 0..6 per LABEL_TO_INT
    source_path: Path    # absolute path under /data/cancerImagingArchive
    raw_class_dir: str   # original directory name, e.g. "Melanoma"
    patient_num: str     # raw patient number, e.g. "01"
    slot: str            # slot index, e.g. "1"


def parse_slide_filename(svs_path: str | Path) -> SlideMetadata:
    """Parse a CATCH WSI path into a :class:`SlideMetadata`.

    Args:
        svs_path: Path under ``/data/cancerImagingArchive/<ClassDir>/...svs``.

    Returns:
        Parsed metadata.

    Raises:
        DataIntegrityError: If the filename or class directory is unrecognized.
    """
    path = Path(svs_path)
    name = path.name
    match = SLIDE_FILENAME_RE.match(name)
    if match is None:
        raise DataIntegrityError(
            f"Slide filename does not match expected pattern '<ClassDir>_<ID>_<Slot>.svs': {name}"
        )
    raw_class = match.group("klass")
    patient_num = match.group("patient")
    slot = match.group("slot")

    label_abbrev = CLASS_DIR_TO_LABEL.get(raw_class)
    if label_abbrev is None:
        raise DataIntegrityError(
            f"Unknown class directory '{raw_class}' in path {path}. "
            f"Expected one of {sorted(CLASS_DIR_TO_LABEL)}."
        )

    return SlideMetadata(
        slide_id=f"{label_abbrev}_{patient_num}_{slot}",
        patient_id=f"{label_abbrev}_{patient_num}",
        tumor_class=label_abbrev,
        label=LABEL_TO_INT[label_abbrev],
        source_path=path.resolve(),
        raw_class_dir=raw_class,
        patient_num=patient_num,
        slot=slot,
    )


def discover_slides(raw_root: str | Path) -> list[SlideMetadata]:
    """Scan ``raw_root`` and return one :class:`SlideMetadata` per ``.svs`` file.

    Skips files that don't match the locked filename pattern and logs the count.
    """
    root = Path(raw_root)
    slides: list[SlideMetadata] = []
    skipped: list[Path] = []
    for svs in sorted(root.rglob("*.svs")):
        try:
            slides.append(parse_slide_filename(svs))
        except DataIntegrityError as e:
            logger.warning("Skipping unrecognized slide: %s (%s)", svs, e)
            skipped.append(svs)
    logger.info("Discovered %d slides under %s (skipped %d)", len(slides), root, len(skipped))
    return slides


# ---------------------------------------------------------------------------
# HDF5 (per-slide GigaPath features)
# ---------------------------------------------------------------------------


def write_features_h5(
    out_path: str | Path,
    *,
    slide_id: str,
    embeddings: np.ndarray,   # [N, D]
    coordinates: np.ndarray,  # [N, 2]
    tile_indices: np.ndarray, # [N]
    metadata: dict[str, Any],
) -> Path:
    """Write a per-slide HDF5 feature file with the locked schema.

    Schema (locked, design `03-architecture.md` §3.B):
        /embeddings   float32 [N, D]
        /coordinates  int32   [N, 2]
        /tile_indices int32   [N]
        /metadata     str (JSON-encoded dict)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if embeddings.ndim != 2:
        raise DataIntegrityError(f"embeddings must be 2D [N, D], got {embeddings.shape}")
    if coordinates.shape != (embeddings.shape[0], 2):
        raise DataIntegrityError(
            f"coordinates shape mismatch: expected ({embeddings.shape[0]}, 2), got {coordinates.shape}"
        )
    if tile_indices.shape != (embeddings.shape[0],):
        raise DataIntegrityError(
            f"tile_indices shape mismatch: expected ({embeddings.shape[0]},), got {tile_indices.shape}"
        )

    metadata_json = json.dumps({"slide_id": slide_id, **metadata}, sort_keys=True)

    with h5py.File(out, "w") as f:
        f.create_dataset("embeddings", data=embeddings.astype(np.float32))
        f.create_dataset("coordinates", data=coordinates.astype(np.int32))
        f.create_dataset("tile_indices", data=tile_indices.astype(np.int32))
        # Metadata stored as scalar string (h5py handles bytes encoding)
        f.create_dataset("metadata", data=metadata_json)
    return out


def read_features_h5(path: str | Path) -> dict[str, Any]:
    """Read a per-slide HDF5 file. Returns dict with embeddings / coords / metadata."""
    p = Path(path)
    with h5py.File(p, "r") as f:
        for required in ("embeddings", "coordinates", "tile_indices", "metadata"):
            if required not in f:
                raise DataIntegrityError(f"HDF5 missing required dataset '{required}': {p}")
        emb = f["embeddings"][...]
        coords = f["coordinates"][...]
        idx = f["tile_indices"][...]
        meta_raw = f["metadata"][()]
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw)
    return {
        "embeddings": emb,
        "coordinates": coords,
        "tile_indices": idx,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# PyG (per-slide graph)
# ---------------------------------------------------------------------------


def save_pyg_data(data: Any, out_path: str | Path) -> Path:
    """Save a torch_geometric.data.Data object to disk.

    Stored via ``torch.save`` (Python pickle protocol). The file extension
    is conventionally ``.pt``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # weights_only=False is required for non-tensor PyG attrs (slide_id str, etc.)
    # Write to a sibling temp file and rename, so an interrupted writer never
    # leaves a truncated <slide>.pt that resume logic would treat as done.
    tmp = out.with_name(out.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, out)
    return out


def load_pyg_data(path: str | Path) -> Any:
    """Load a torch_geometric.data.Data object from disk."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# CATCH annotation — tile labels (µPDCA #8)
# ---------------------------------------------------------------------------


#: Default canonical path for per-WSI tile-label numpy artifacts produced by
#: the CATCH.json polygon → tile center mapping. Overrideable via the
#: ``paths.tile_labels`` config key (µPDCA #8 M1).
DEFAULT_TILE_LABELS_DIR = str(output_dir() / "annotations" / "tile_labels")

#: CATCH category id ranges (per CATCH.json).
#: Cats 1-6 = Tissue (Bone, Cartilage, Dermis, Epidermis, Subcutis,
#: Inflamm/Necrosis). Cats 7-13 = Tumor (MEL, PLC, MCT, PNST, SCC, TRB, HIS).
CATCH_TISSUE_RANGE = (1, 6)
CATCH_TUMOR_RANGE = (7, 13)
CATCH_INFLAMM_ID = 6  # Inflamm/Necrosis maps to "inflammation" in 3-way

#: Tile-label sentinel for tiles whose center did not fall inside any
#: polygon. ROI-only coverage in CATCH leaves ~64% of tiles unmapped on
#: average (see pre-analysis report).
TILE_LABEL_UNKNOWN = -1


def load_tile_labels(
    slide_id: str,
    *,
    labels_dir: str | Path | None = None,
):
    """Load per-tile CATCH category labels for one WSI.

    Returns
    -------
    ``numpy.ndarray`` of shape ``[N]``, dtype ``int32`` if the file exists,
    otherwise ``None``. Values:

    - ``1..6`` : Tissue categories (Bone, Cartilage, Dermis, Epidermis,
      Subcutis, Inflamm/Necrosis)
    - ``7..13`` : Tumor categories (MEL, PLC, MCT, PNST, SCC, TRB, HIS)
    - ``-1`` : Tile center did not fall inside any polygon (unmapped)

    Callers should treat ``None`` as a signal to fall back to the
    unsupervised path (e.g. k-means in :class:`HeterogeneousGraph`).

    See ``docs/04-report/_PRE-µPDCA-8-annotation-survey.md`` for the
    detailed coverage statistics (avg 36.2% of tiles mapped per WSI).
    """
    import numpy as np

    base = Path(labels_dir) if labels_dir is not None else Path(DEFAULT_TILE_LABELS_DIR)
    path = base / f"{slide_id}.npy"
    if not path.exists():
        return None
    return np.load(path)


def tile_labels_to_3way(labels):
    """Collapse 13-way CATCH category labels into the 3-way scheme used by
    :class:`HeterogeneousGraph` (``tumor / stroma / inflammation``).

    Mapping:
        - ``0`` (tumor) : category in ``[7, 13]``
        - ``1`` (stroma) : category in ``[1, 5]`` (Bone, Cartilage, Dermis,
          Epidermis, Subcutis)
        - ``2`` (inflammation) : category ``== 6`` (Inflamm/Necrosis)
        - ``-1`` (unknown) : category ``== -1`` (unmapped — caller decides
          whether to assign via fallback)

    Returns
    -------
    ``numpy.ndarray`` of shape ``[N]``, ``int32``.
    """
    import numpy as np

    out = np.full_like(labels, fill_value=-1, dtype=np.int32)
    tumor_mask = (labels >= CATCH_TUMOR_RANGE[0]) & (labels <= CATCH_TUMOR_RANGE[1])
    inflamm_mask = labels == CATCH_INFLAMM_ID
    stroma_mask = (
        (labels >= CATCH_TISSUE_RANGE[0])
        & (labels <= CATCH_TISSUE_RANGE[1])
        & (~inflamm_mask)
    )
    out[tumor_mask] = 0
    out[stroma_mask] = 1
    out[inflamm_mask] = 2
    return out
