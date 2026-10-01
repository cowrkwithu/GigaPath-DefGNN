"""Unit tests for src/utils/io_utils.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.config import load_config
from src.utils.errors import DataIntegrityError
from src.utils.io_utils import (
    CLASS_DIR_TO_LABEL,
    INT_TO_LABEL,
    LABEL_TO_INT,
    SLIDE_FILENAME_RE,
    discover_slides,
    parse_slide_filename,
    read_features_h5,
    save_pyg_data,
    load_pyg_data,
    write_features_h5,
)


# --- parse_slide_filename ---


def test_parse_melanoma():
    md = parse_slide_filename("/data/cancerImagingArchive/Melanoma/Melanoma_01_1.svs")
    assert md.slide_id == "MEL_01_1"
    assert md.patient_id == "MEL_01"
    assert md.tumor_class == "MEL"
    assert md.label == 0
    assert md.raw_class_dir == "Melanoma"
    assert md.patient_num == "01"
    assert md.slot == "1"


def test_parse_all_seven_classes():
    samples = {
        "Melanoma_01_1.svs": ("MEL", 0),
        "MCT_05_2.svs": ("MCT", 1),
        "SCC_10_1.svs": ("SCC", 2),
        "PNST_15_3.svs": ("PNST", 3),
        "Plasmacytoma_20_1.svs": ("PLC", 4),
        "Trichoblastoma_25_2.svs": ("TRB", 5),
        "Histiocytoma_30_1.svs": ("HIS", 6),
    }
    for fname, (label_abbrev, label_int) in samples.items():
        md = parse_slide_filename(f"/data/cancerImagingArchive/X/{fname}")
        assert md.tumor_class == label_abbrev
        assert md.label == label_int


def test_parse_same_patient_different_slots():
    """Slot index changes; patient_id stays the same."""
    md1 = parse_slide_filename("/x/Melanoma/Melanoma_07_1.svs")
    md2 = parse_slide_filename("/x/Melanoma/Melanoma_07_2.svs")
    assert md1.patient_id == md2.patient_id == "MEL_07"
    assert md1.slide_id != md2.slide_id


def test_parse_unknown_class_raises():
    with pytest.raises(DataIntegrityError, match="Unknown class directory"):
        parse_slide_filename("/x/Frog/Frog_01_1.svs")


def test_parse_bad_format_raises():
    with pytest.raises(DataIntegrityError, match="does not match expected pattern"):
        parse_slide_filename("/x/y/garbage.svs")


def test_label_int_mapping_complete():
    assert set(LABEL_TO_INT.values()) == {0, 1, 2, 3, 4, 5, 6}
    assert set(INT_TO_LABEL.keys()) == {0, 1, 2, 3, 4, 5, 6}
    for label, idx in LABEL_TO_INT.items():
        assert INT_TO_LABEL[idx] == label


def test_class_map_matches_config():
    """Locked class mapping in io_utils must match configs/default.yaml."""
    cfg = load_config()
    cfg_map = dict(cfg.paths.class_dir_to_label)
    assert cfg_map == CLASS_DIR_TO_LABEL


# --- discover_slides on the real /data/cancerImagingArchive ---


def test_discover_real_catch_dataset():
    """If the dataset is mounted, discovery finds 350 slides cleanly."""
    raw = Path("/data/cancerImagingArchive")
    if not raw.exists():
        pytest.skip("CATCH dataset not mounted at /data/cancerImagingArchive")
    slides = discover_slides(raw)
    assert len(slides) == 350, f"expected 350 slides, found {len(slides)}"
    # Class balance: 50 per class
    by_class: dict[str, int] = {}
    for s in slides:
        by_class[s.tumor_class] = by_class.get(s.tumor_class, 0) + 1
    for cls in ("MEL", "MCT", "SCC", "PNST", "PLC", "TRB", "HIS"):
        assert by_class.get(cls) == 50, f"class {cls} has {by_class.get(cls)} slides, expected 50"


# --- HDF5 write/read roundtrip ---


def test_hdf5_roundtrip(tmp_path: Path):
    n, d = 32, 1536
    embeds = np.random.randn(n, d).astype(np.float32)
    coords = np.random.randint(0, 50000, size=(n, 2)).astype(np.int32)
    indices = np.arange(n, dtype=np.int32)

    out = tmp_path / "MEL_01_1.h5"
    write_features_h5(
        out,
        slide_id="MEL_01_1",
        embeddings=embeds,
        coordinates=coords,
        tile_indices=indices,
        metadata={"num_tiles": n, "magnification": 40, "encoder_version": "test"},
    )

    loaded = read_features_h5(out)
    assert loaded["embeddings"].shape == (n, d)
    assert loaded["embeddings"].dtype == np.float32
    assert loaded["coordinates"].dtype == np.int32
    assert loaded["coordinates"].shape == (n, 2)
    assert np.array_equal(loaded["embeddings"], embeds)
    assert np.array_equal(loaded["coordinates"], coords)
    assert np.array_equal(loaded["tile_indices"], indices)
    assert loaded["metadata"]["slide_id"] == "MEL_01_1"
    assert loaded["metadata"]["num_tiles"] == n


def test_hdf5_shape_mismatch_raises(tmp_path: Path):
    out = tmp_path / "bad.h5"
    embeds = np.random.randn(10, 1536).astype(np.float32)
    coords = np.zeros((9, 2), dtype=np.int32)  # mismatched
    with pytest.raises(DataIntegrityError, match="coordinates shape mismatch"):
        write_features_h5(
            out, slide_id="X", embeddings=embeds, coordinates=coords,
            tile_indices=np.arange(10, dtype=np.int32), metadata={},
        )


def test_pyg_save_load_roundtrip(tmp_path: Path):
    pyg = pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, slide_id="MEL_01_1", y=torch.tensor([0]))

    out = tmp_path / "g.pt"
    save_pyg_data(data, out)
    loaded = load_pyg_data(out)
    assert torch.equal(loaded.x, x)
    assert torch.equal(loaded.edge_index, edge_index)
    assert loaded.slide_id == "MEL_01_1"
