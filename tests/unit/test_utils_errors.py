"""Unit tests for src/utils/errors.py."""
from __future__ import annotations

import pytest

from src.utils.errors import (
    ConfigError,
    DataIntegrityError,
    FrozenEncoderError,
    GraphDisconnectedError,
    InsufficientFoldsError,
    InsufficientTilesError,
    VetGigaGraphError,
)


def test_all_subclass_base():
    """Every custom exception inherits from VetGigaGraphError."""
    for cls in (
        InsufficientTilesError,
        GraphDisconnectedError,
        InsufficientFoldsError,
        ConfigError,
        DataIntegrityError,
        FrozenEncoderError,
    ):
        assert issubclass(cls, VetGigaGraphError), f"{cls.__name__} must subclass VetGigaGraphError"


def test_base_subclass_exception():
    assert issubclass(VetGigaGraphError, Exception)


def test_can_raise_and_catch():
    with pytest.raises(VetGigaGraphError):
        raise InsufficientTilesError("only 3 tiles, min is 16")


def test_message_preserved():
    msg = "graph has 0 edges"
    try:
        raise GraphDisconnectedError(msg)
    except GraphDisconnectedError as e:
        assert str(e) == msg
