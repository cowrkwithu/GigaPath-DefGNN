"""No-op stand-ins for ``fairscale.nn.{checkpoint_wrapper, wrap}``.

The upstream GigaPath LongNet pulls these in but only invokes them when
``args.checkpoint_activations`` or ``args.fsdp`` is True — both default
False for the slide-encoder use case, so the stubs are never actually
hit at runtime. Vendoring no-ops avoids a 100+ MB ``fairscale``
dependency. If a downstream user enables either flag, the stubs will
silently no-op (which is mildly wrong but loudly logged at INFO).

See ``src/models/_vendored_gigapath/README.md`` adaptation §1.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def checkpoint_wrapper(module: Any, *args, **kwargs) -> Any:
    """No-op: returns the module unchanged."""
    if args or kwargs:
        logger.info(
            "_fairscale_stubs.checkpoint_wrapper called with args=%s kwargs=%s "
            "— ignoring (no-op stub)",
            args, kwargs,
        )
    return module


def wrap(module: Any, *args, **kwargs) -> Any:
    """No-op: returns the module unchanged."""
    if args or kwargs:
        logger.info(
            "_fairscale_stubs.wrap called with args=%s kwargs=%s "
            "— ignoring (no-op stub)",
            args, kwargs,
        )
    return module
