"""Abstract base class for slide-level graph constructors.

Every concrete constructor (spatial-kNN, feature-similarity, dual-edge,
hierarchical, heterogeneous) implements :meth:`_build_edges` and gets:

* Input coercion (numpy ↔ torch, dtype normalization).
* Locked PyG ``Data`` schema (``x``, ``pos``, ``edge_index``, ``edge_attr``,
  optional ``slide_id``, ``y``, plus any extra fields the variant emits).
* Common invariant guards: no implicit self-loops, every edge index in
  ``[0, N)``, finite ``edge_attr``, non-empty ``edge_index`` (raises
  :class:`GraphDisconnectedError` otherwise).
* :meth:`get_config` round-trip: ``cls(**ctor.get_config())`` rebuilds an
  identical constructor — the test in `test_graph_construction.py`
  enforces this.

Device contract:

* ``self.device`` (lazy property) defaults to ``'auto'`` → ``cuda`` when
  ``torch.cuda.is_available()`` else ``cpu``. Subclasses may expose a
  ``device`` kwarg and assign ``self.device = device`` to override.
* ``data.x`` and ``data.pos`` are always stored on CPU so saved ``.pt``
  files are device-agnostic and the ``test_node_features_preserved``
  invariant holds when the fixture is CPU.
* ``_build_edges`` receives ``x``/``pos`` on ``self.device``; whatever
  tensors it returns are moved to CPU before being placed on ``Data``.

References:
    Design: docs/02-design/03-architecture.md §4 (Module C)
    Spec:   docs/02-design/02-data-spec.md §0 (PyG schema)
    Tests:  docs/02-design/03-architecture.md §4.C (9 common invariants)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data

from src.utils.errors import GraphDisconnectedError

logger = logging.getLogger(__name__)

#: Names of the keys that subclass ``_build_edges`` may return.
#: ``edge_index`` and ``edge_attr`` are required; the others are optional
#: passthroughs (e.g. ``edge_type`` for dual-edge / heterogeneous).
ALLOWED_EDGE_KEYS = frozenset(
    {"edge_index", "edge_attr", "edge_type", "node_type"}
)

#: Row chunk size for GPU KNN / similarity matmul. For N=60k tiles and
#: D=1536, a [chunk, N] float32 matrix is ~470 MB at chunk=2048 — fits
#: comfortably alongside the [N, D] embedding (~365 MB) on a 24 GB GPU.
DEFAULT_CHUNK_SIZE = 2048


DeviceLike = Union[str, torch.device, None]


def _resolve_device(requested: DeviceLike) -> torch.device:
    """Resolve ``'auto' | None | str | torch.device`` to a concrete device.

    ``'auto'`` and ``None`` promote to CUDA when a GPU is available,
    matching the design contract that production runs are GPU-first.
    Explicit strings (``'cpu'``, ``'cuda'``, ``'cuda:0'``) are honored.
    """
    if isinstance(requested, torch.device):
        return requested
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class BaseGraphConstructor(ABC):
    """Abstract slide-level graph builder.

    Subclasses must:

    1. Override :meth:`_build_edges` with the variant-specific topology.
    2. Override :meth:`get_config` to expose the constructor kwargs (so
       round-trip rebuild is exact).

    Subclasses *may* accept a ``device`` kwarg and assign
    ``self.device = device``; if they don't, the lazy property below
    auto-resolves to CUDA when available.
    """

    # --- device ----------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        cached = getattr(self, "_resolved_device", None)
        if cached is not None:
            return cached
        req = getattr(self, "_device_request", "auto")
        resolved = _resolve_device(req)
        self._resolved_device = resolved
        return resolved

    @device.setter
    def device(self, value: DeviceLike) -> None:
        self._device_request = value
        self._resolved_device = _resolve_device(value)

    # --- public API ------------------------------------------------------- #

    def build_graph(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        coordinates: Union[np.ndarray, torch.Tensor],
        *,
        slide_id: Optional[str] = None,
        y: Optional[int] = None,
    ) -> Data:
        """Build a PyG ``Data`` object from per-tile embeddings + coords.

        Args:
            embeddings: Shape ``[N, D]``. Stored as ``data.x`` (on CPU,
                unchanged — no normalization, no projection — invariant
                required by ``test_node_features_preserved``).
            coordinates: Shape ``[N, 2]``. Stored as ``data.pos`` (CPU).
            slide_id: Optional metadata; stored verbatim on ``data``.
            y: Optional integer label; stored as ``data.y`` of shape
                ``[1]`` (matching PyG convention for graph-level labels).

        Returns:
            PyG ``Data`` with the locked schema; all tensor fields on CPU.

        Raises:
            GraphDisconnectedError: When the variant produces zero edges.
                Variants that can degenerate (e.g. high-τ feature-sim)
                detect this here rather than letting the model OOM later.
        """
        # CPU copies for storage on Data — keeps saved .pt device-agnostic.
        x_cpu = _to_float32_tensor(
            embeddings, name="embeddings", expected_ndim=2, device="cpu"
        )
        pos_cpu = _to_float32_tensor(
            coordinates, name="coordinates", expected_ndim=2, device="cpu"
        )
        if pos_cpu.shape[0] != x_cpu.shape[0]:
            raise ValueError(
                f"embeddings and coordinates disagree on N: "
                f"{x_cpu.shape[0]} vs {pos_cpu.shape[0]}"
            )

        # Compute copies on the resolved device — these feed _build_edges.
        device = self.device
        if device.type == "cpu":
            x, pos = x_cpu, pos_cpu
        else:
            x = x_cpu.to(device, non_blocking=True)
            pos = pos_cpu.to(device, non_blocking=True)

        edge_dict = self._build_edges(x=x, pos=pos)
        # Move every returned tensor back to CPU before validation and
        # Data assembly so on-disk artifacts don't carry CUDA state.
        edge_dict_cpu: dict[str, torch.Tensor] = {
            k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v)
            for k, v in edge_dict.items()
        }
        self._validate_edge_dict(edge_dict_cpu, n_nodes=x_cpu.shape[0])

        data_kwargs: dict[str, Any] = {"x": x_cpu, "pos": pos_cpu, **edge_dict_cpu}
        if slide_id is not None:
            data_kwargs["slide_id"] = slide_id
        if y is not None:
            data_kwargs["y"] = torch.tensor([int(y)], dtype=torch.long)

        return Data(**data_kwargs)

    @abstractmethod
    def _build_edges(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        """Return a dict containing at minimum ``edge_index`` and ``edge_attr``.

        ``x`` and ``pos`` arrive on ``self.device`` — variants should
        keep intermediate compute on the same device for the GPU path
        to pay off.

        Optional keys that downstream consumers know about:

        * ``edge_type`` — int64 ``[E]`` tagging spatial-vs-feature
          (dual-edge) or intra-vs-inter (heterogeneous).
        * ``node_type`` — int64 ``[N]`` for heterogeneous graphs.

        Subclasses that need richer outputs (e.g. hierarchical's
        ``level2_*`` fields, ``pool_assignment``) may include additional
        keys; they're passed through to ``Data`` verbatim. Use names with
        an unambiguous prefix to avoid colliding with PyG built-ins.
        """

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return constructor kwargs for round-trip ``cls(**get_config())``.

        ``device`` is intentionally **not** part of the graph spec — it's
        an execution detail. Round-trip rebuild must not bake the host's
        device into the saved config.
        """

    # --- validation ------------------------------------------------------- #

    def _validate_edge_dict(
        self,
        edge_dict: Mapping[str, torch.Tensor],
        *,
        n_nodes: int,
    ) -> None:
        if "edge_index" not in edge_dict or "edge_attr" not in edge_dict:
            raise KeyError(
                f"{type(self).__name__}._build_edges must return "
                "'edge_index' and 'edge_attr'."
            )

        edge_index = edge_dict["edge_index"]
        edge_attr = edge_dict["edge_attr"]

        if edge_index.numel() == 0:
            raise GraphDisconnectedError(
                f"{type(self).__name__} produced 0 edges on a graph with "
                f"{n_nodes} nodes. Hint: lower the threshold or increase k."
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                f"edge_index must be int64; got {edge_index.dtype}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must be [2, E]; got shape {tuple(edge_index.shape)}"
            )

        if edge_index.min() < 0 or edge_index.max() >= n_nodes:
            raise ValueError(
                f"edge_index out of bounds: min={int(edge_index.min())}, "
                f"max={int(edge_index.max())}, n_nodes={n_nodes}"
            )

        if (edge_index[0] == edge_index[1]).any():
            raise ValueError(
                f"{type(self).__name__} emitted self-loops; the design "
                "contract requires explicit removal."
            )

        if not torch.isfinite(edge_attr).all():
            raise ValueError(
                f"{type(self).__name__} produced non-finite edge_attr."
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_attr first dim ({edge_attr.shape[0]}) must match "
                f"E ({edge_index.shape[1]})"
            )


# --------------------------------------------------------------------------- #
# Helpers shared by all subclasses
# --------------------------------------------------------------------------- #


def _to_float32_tensor(
    arr: Union[np.ndarray, torch.Tensor],
    *,
    name: str,
    expected_ndim: int,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Coerce inputs to a contiguous float32 tensor, optionally on ``device``."""
    if isinstance(arr, np.ndarray):
        tensor = torch.from_numpy(arr).to(torch.float32)
    elif isinstance(arr, torch.Tensor):
        tensor = arr.to(torch.float32)
    else:
        raise TypeError(
            f"{name} must be numpy.ndarray or torch.Tensor; got {type(arr).__name__}"
        )
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D; got shape {tuple(tensor.shape)}"
        )
    if device is not None:
        target = device if isinstance(device, torch.device) else torch.device(device)
        if tensor.device != target:
            tensor = tensor.to(target)
    return tensor.contiguous()


def symmetrize_directed(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make a directed edge set bidirectional by concatenating its reverse.

    Note: deliberately keeps duplicates (so ``E_after == 2 * E_before``).
    The design contract for ``test_spatial_knn_edge_count`` requires
    ``E == 100 * 8 * 2`` after symmetrization — that's only the case if
    we don't dedup mutual k-NN edges.
    """
    rev = edge_index.flip(0)
    return (
        torch.cat([edge_index, rev], dim=1),
        torch.cat([edge_attr, edge_attr], dim=0),
    )


def knn_indices(
    points: torch.Tensor,
    k: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Return ``[N, k]`` indices of k-nearest neighbors (self excluded).

    Dispatches by ``points.device``:

    * **CPU** — uses scipy's KDTree (fast for 2D coords, exact ties).
    * **GPU** — uses chunked ``torch.cdist`` + ``torch.topk`` so we don't
      materialize a single ``[N, N]`` distance matrix (would OOM at
      N≈60k tiles, which CATCH WSIs actually reach).

    The returned tensor lives on ``points.device``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    n = points.shape[0]
    if k >= n:
        raise ValueError(
            f"k={k} must be < N={n} so each node has neighbors other than itself"
        )

    if points.device.type == "cpu":
        from scipy.spatial import cKDTree

        arr = points.detach().cpu().numpy()
        tree = cKDTree(arr)
        # query returns (distances, indices); we only need indices.
        # Request k+1 because the nearest neighbor is the point itself.
        _, idx = tree.query(arr, k=k + 1)
        return torch.as_tensor(idx[:, 1:], dtype=torch.long)

    return _torch_knn_indices(points, k=k, chunk_size=chunk_size)


def _torch_knn_indices(
    points: torch.Tensor,
    k: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """GPU-friendly k-NN via chunked ``torch.cdist`` + ``topk``.

    Memory: ``O(chunk_size * N * 4 bytes)`` per chunk. At chunk=2048,
    N=62k, that's ~500 MB — comfortably inside a 24 GB GPU even alongside
    a [N, D=1536] embedding (~365 MB).
    """
    n = points.shape[0]
    out = torch.empty((n, k), dtype=torch.long, device=points.device)
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        # [chunk, N] pairwise Euclidean distances.
        d = torch.cdist(points[i:j], points)
        rows = torch.arange(i, j, device=points.device).unsqueeze(-1)
        cols = torch.arange(n, device=points.device).unsqueeze(0)
        # Mask self-pairs so they never make the topk.
        d.masked_fill_(rows == cols, float("inf"))
        _, idx = torch.topk(d, k=k, dim=-1, largest=False)
        out[i:j] = idx
    return out


def knn_to_directed_edges(
    nbr_indices: torch.Tensor,
) -> torch.Tensor:
    """Convert ``[N, k]`` neighbor indices to a directed ``[2, N*k]`` edge index."""
    n, k = nbr_indices.shape
    src = torch.arange(n, dtype=torch.long, device=nbr_indices.device).repeat_interleave(k)
    dst = nbr_indices.flatten()
    return torch.stack([src, dst], dim=0)
