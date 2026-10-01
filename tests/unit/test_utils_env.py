"""Unit tests for src/utils/env.py."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils import env as env_module
from src.utils.env import (
    DEFAULT_ENV_PATH,
    _is_placeholder,
    get_hf_token,
    get_wandb_api_key,
    load_env,
)
from src.utils.errors import ConfigError


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset known env vars per test and pin `_env_loaded=True` so the repo
    `.env` is not reloaded mid-test (which would re-set the vars we just
    deleted). Tests that exercise `load_env` directly flip `_env_loaded`
    back to False themselves.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "WANDB_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(env_module, "_env_loaded", True)
    yield


def test_default_env_path_at_repo_root():
    # Three parents up from src/utils/env.py is the repo root
    expected = Path(__file__).resolve().parents[2]
    assert DEFAULT_ENV_PATH.parent == expected


def test_load_env_missing_file_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env_module, "_env_loaded", False)
    result = load_env(tmp_path / "nope.env", silent=True)
    assert result is None


def test_load_env_with_real_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env_module, "_env_loaded", False)
    env_file = tmp_path / "test.env"
    env_file.write_text("HF_TOKEN=hf_realtoken_abc123\nWANDB_API_KEY=wandbkey\n")
    result = load_env(env_file)
    assert result == env_file
    assert os.environ.get("HF_TOKEN") == "hf_realtoken_abc123"
    assert os.environ.get("WANDB_API_KEY") == "wandbkey"


def test_load_env_does_not_override_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "shell_set_token")
    monkeypatch.setattr(env_module, "_env_loaded", False)
    env_file = tmp_path / "test.env"
    env_file.write_text("HF_TOKEN=dotenv_token\n")
    load_env(env_file, override=False)
    assert os.environ["HF_TOKEN"] == "shell_set_token"


def test_load_env_override_true(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "shell_set_token")
    monkeypatch.setattr(env_module, "_env_loaded", False)
    env_file = tmp_path / "test.env"
    env_file.write_text("HF_TOKEN=dotenv_token\n")
    load_env(env_file, override=True)
    assert os.environ["HF_TOKEN"] == "dotenv_token"


def test_load_env_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env_module, "_env_loaded", False)
    env_file = tmp_path / "test.env"
    env_file.write_text("HF_TOKEN=tok1\n")
    load_env(env_file)
    # Modify file but don't override
    env_file.write_text("HF_TOKEN=tok2\n")
    load_env(env_file)  # should be no-op
    assert os.environ["HF_TOKEN"] == "tok1"


# --- get_hf_token ---


def test_hf_token_returns_none_when_unset(monkeypatch):
    assert get_hf_token() is None


def test_hf_token_required_raises_when_unset(monkeypatch):
    with pytest.raises(ConfigError, match="No Hugging Face token"):
        get_hf_token(required=True)


def test_hf_token_picks_up_hf_token_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_real_token_xyz")
    assert get_hf_token() == "hf_real_token_xyz"


def test_hf_token_legacy_var_supported(monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy_token")
    assert get_hf_token() == "legacy_token"


def test_hf_token_third_alias_supported(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "third_alias_token")
    assert get_hf_token() == "third_alias_token"


def test_hf_token_priority_order(monkeypatch):
    """HF_TOKEN beats HUGGING_FACE_HUB_TOKEN beats HUGGINGFACE_TOKEN."""
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "third")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "second")
    monkeypatch.setenv("HF_TOKEN", "first")
    assert get_hf_token() == "first"


def test_hf_token_placeholder_treated_as_unset(monkeypatch):
    """Template placeholder must NOT be returned as a real token."""
    monkeypatch.setenv("HF_TOKEN", "hf_replace_this_with_your_actual_token")
    assert get_hf_token() is None
    with pytest.raises(ConfigError):
        get_hf_token(required=True)


# --- get_wandb_api_key ---


def test_wandb_key_returns_none_when_unset(monkeypatch):
    assert get_wandb_api_key() is None


def test_wandb_key_returns_real_value(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "wandb_xyz")
    assert get_wandb_api_key() == "wandb_xyz"


def test_wandb_key_placeholder_treated_as_unset(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "your_wandb_api_key_here")
    assert get_wandb_api_key() is None


# --- _is_placeholder ---


def test_placeholder_detection():
    assert _is_placeholder("hf_replace_this_with_your_actual_token")
    assert _is_placeholder("  hf_replace_this_with_your_actual_token  ")  # whitespace-trimmed
    assert _is_placeholder("your_wandb_api_key_here")
    assert not _is_placeholder("hf_realtokenfromhuggingface")
    assert not _is_placeholder("")
