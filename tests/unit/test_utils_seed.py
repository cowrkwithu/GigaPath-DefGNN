"""Unit tests for src/utils/seed.py."""
from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.seed import (
    LOCKED_FOLD_SEEDS,
    last_seed,
    seed_was_set,
    set_global_seed,
)


def test_locked_fold_seeds_value():
    """Seeds locked by design 08-statistics-reproducibility.md §2."""
    assert LOCKED_FOLD_SEEDS == (42, 123, 456, 789, 1024)


def test_python_random_reproducible():
    set_global_seed(42, deterministic=False)
    a = [random.random() for _ in range(5)]
    set_global_seed(42, deterministic=False)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_reproducible():
    set_global_seed(42, deterministic=False)
    a = np.random.rand(10)
    set_global_seed(42, deterministic=False)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_torch_cpu_reproducible():
    set_global_seed(42, deterministic=False)
    a = torch.rand(10)
    set_global_seed(42, deterministic=False)
    b = torch.rand(10)
    assert torch.equal(a, b)


def test_torch_cuda_reproducible_if_available():
    if not torch.cuda.is_available():
        return
    set_global_seed(42, deterministic=False)
    a = torch.rand(10, device="cuda")
    set_global_seed(42, deterministic=False)
    b = torch.rand(10, device="cuda")
    assert torch.equal(a, b)


def test_seed_was_set_flag():
    set_global_seed(42, deterministic=False)
    assert seed_was_set() is True
    assert last_seed() == 42


def test_different_seeds_produce_different_outputs():
    set_global_seed(42, deterministic=False)
    a = torch.rand(10)
    set_global_seed(123, deterministic=False)
    b = torch.rand(10)
    assert not torch.equal(a, b)
