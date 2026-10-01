"""Environment variable loading from .env files.

Loads the repo-root ``.env`` file (if present) into ``os.environ`` exactly
once per process. Imported by :mod:`src.utils.config`, so any code that
loads config also has env vars available.

Secrets that should live in ``.env`` (NOT in YAML, NOT in code):

* ``HF_TOKEN`` — Hugging Face access token (required for GigaPath gated model)
* ``WANDB_API_KEY`` — WandB authentication (optional)

Template: see ``.env.example`` at repo root.

References:
    Design: docs/02-design/features/vetgigagraph.design.md §7 (Security Considerations)
    Design: docs/02-design/05-software-architecture.md §3 (Coding Conventions, env vars)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.utils.errors import ConfigError

logger = logging.getLogger(__name__)

#: Repo root (3 levels up from this file: src/utils/env.py → repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default location of the .env file.
DEFAULT_ENV_PATH = _REPO_ROOT / ".env"

#: Tracks whether load_env has been called (idempotency).
_env_loaded: bool = False


def load_env(
    env_path: str | Path | None = None,
    *,
    override: bool = False,
    silent: bool = False,
) -> Path | None:
    """Load environment variables from a .env file into ``os.environ``.

    Args:
        env_path: Path to the .env file. Defaults to ``<repo>/.env``.
        override: If True, .env values overwrite already-set env vars.
            If False (default), existing env vars (e.g. set in shell) win.
        silent: If True, do not warn when the .env file is missing.

    Returns:
        The resolved path if the file was loaded, else None.
    """
    global _env_loaded
    if _env_loaded and not override:
        return None

    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if not path.exists():
        if not silent:
            logger.info(
                ".env not found at %s — using process env only. "
                "Copy .env.example to .env and fill in tokens to silence this.",
                path,
            )
        _env_loaded = True
        return None

    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise ConfigError(
            f"python-dotenv not installed but {path} exists. "
            "Add 'python-dotenv' to requirements.txt or remove the .env file."
        ) from e

    load_dotenv(dotenv_path=path, override=override)
    _env_loaded = True
    logger.info("Loaded env from %s (override=%s)", path, override)
    return path


def get_hf_token(*, required: bool = False) -> str | None:
    """Return the Hugging Face access token, if any.

    Looks up env vars in order:
        1. ``HF_TOKEN`` (current canonical)
        2. ``HUGGING_FACE_HUB_TOKEN`` (legacy, still honored by huggingface_hub)
        3. ``HUGGINGFACE_TOKEN`` (occasional alias in older docs)

    Args:
        required: If True, raise :class:`ConfigError` when no token is found
            and the placeholder template value is detected.

    Returns:
        The token string or None if unset.

    Raises:
        ConfigError: If ``required=True`` and no real token is configured.
    """
    load_env(silent=True)
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.environ.get(var)
        if tok and not _is_placeholder(tok):
            return tok

    if required:
        raise ConfigError(
            "No Hugging Face token found. "
            "Set HF_TOKEN in .env (see .env.example), or run `huggingface-cli login`. "
            "Required for GigaPath gated model access."
        )
    return None


def get_wandb_api_key() -> str | None:
    """Return the WandB API key from env (or None — WandB falls back to no-op)."""
    load_env(silent=True)
    key = os.environ.get("WANDB_API_KEY")
    if key and not _is_placeholder(key):
        return key
    return None


#: Default data locations; each is overridden by the environment variable
#: named next to it (shell or ``.env``).
DEFAULT_RAW_DIR = "/data/cancerImagingArchive"   # VETGIGA_RAW_DIR
DEFAULT_OUTPUT_DIR = "/data/cia_outputs"         # VETGIGA_OUTPUT_DIR


def raw_dir() -> Path:
    """Root of the CATCH whole-slide images (``VETGIGA_RAW_DIR``)."""
    load_env(silent=True)
    return Path(os.environ.get("VETGIGA_RAW_DIR", DEFAULT_RAW_DIR))


def output_dir() -> Path:
    """Root of all derived data: tiles, features, graphs, checkpoints (``VETGIGA_OUTPUT_DIR``)."""
    load_env(silent=True)
    return Path(os.environ.get("VETGIGA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


def defgnn_run_dir() -> Path:
    """Run directory of the original GigaPath-DefGNN training (``DEFGNN_RUN_DIR``),
    holding ``fold_<i>/checkpoints/``. Defaults to ``../vetgigagraph_v2/results/deformable``
    next to this repository."""
    load_env(silent=True)
    default = _REPO_ROOT.parent / "vetgigagraph_v2" / "results" / "deformable"
    return Path(os.environ.get("DEFGNN_RUN_DIR", default))


def v1_root() -> Path:
    """Checkout of the v1 (static-graph) repository (``V1_ROOT``). Defaults to
    ``../pw-vetGigagraph`` next to this repository."""
    load_env(silent=True)
    return Path(os.environ.get("V1_ROOT", _REPO_ROOT.parent / "pw-vetGigagraph"))


def portable_path(path: str | Path) -> str:
    """``path`` with a data-root prefix replaced by its variable name, e.g.
    ``$VETGIGA_OUTPUT_DIR/checkpoints/...``, for paths recorded in results files."""
    s = str(path)
    for var, root in (("DEFGNN_RUN_DIR", defgnn_run_dir()), ("VETGIGA_OUTPUT_DIR", output_dir()),
                      ("VETGIGA_RAW_DIR", raw_dir())):
        r = str(root).rstrip("/")
        if s == r or s.startswith(r + "/"):
            return "$" + var + s[len(r):]
    return s


def _is_placeholder(value: str) -> bool:
    """Detect dummy values still present from .env.example."""
    placeholders = {
        "hf_replace_this_with_your_actual_token",
        "your_wandb_api_key_here",
        "your_username_or_team",
    }
    return value.strip() in placeholders


# Auto-load on first import. Cheap, idempotent, and means downstream code
# (timm, huggingface_hub, wandb) sees the tokens without explicit calls.
load_env(silent=True)
