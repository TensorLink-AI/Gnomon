"""Pluggable model backends: third-party forecasting models as candidates.

A *model backend* contributes named point-forecast callables with the same
contract as the built-ins in ``models.py``: deterministic
``(history, horizon, season) -> list[float]``. Backends never bypass the
harness — their models enter the same rolling selection folds as every
other candidate and are selected only by beating the mandatory baselines
on the caller's data. Namespaced candidate names (``backend:model``)
carry provenance into scores, artifacts, and evidence.

Discovery is two-channel:

- **Entry points** — a package declares one in the ``aion.model_backends``
  group resolving to a zero-argument factory that returns a
  :class:`ModelBackend` (or ``None`` when its optional dependency is
  absent). Installing the package is all a user does; uninstalling it
  removes the candidates. The first-party statsforecast backend
  (``backends_statsforecast.py``, the ``statistical`` extra) registers
  this way.
- **Programmatic** — :func:`register_backend` for embedders and tests.

A backend that fails to import, construct, or predict is skipped or
scored ``None`` on the affected fold; it can never crash a run or win
without complete fold coverage. Installed backends are fingerprinted
(name and version) into the artifact ID payload, because a different
candidate set is a different task.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "aion.model_backends"

ModelFn = Callable[[list[float], int, int], list[float]]


@dataclass(frozen=True)
class ModelBackend:
    """A named provider of deterministic point-forecast models."""

    name: str
    version: str
    models: dict[str, ModelFn] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendModelAdapter:
    """One backend model shaped like a TSFM adapter for the evaluation lane."""

    name: str
    _predict: ModelFn

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        return self._predict(history, horizon, season)


_REGISTERED: dict[str, ModelBackend] = {}
_ENTRY_POINTS_LOADED = False


def register_backend(backend: ModelBackend) -> None:
    """Register a backend in-process (embedders and tests)."""
    _REGISTERED[backend.name] = backend


def clear_registered_backends() -> None:
    """Remove programmatically registered backends (test isolation)."""
    _REGISTERED.clear()


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
        discovered = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:
        logger.debug("Model-backend entry-point discovery failed", exc_info=True)
        return
    for entry in discovered:
        try:
            backend = entry.load()()
        except Exception:
            logger.debug("Model backend %s failed to load", entry.name, exc_info=True)
            continue
        if backend is None:
            logger.debug("Model backend %s declined (dependency absent)", entry.name)
            continue
        _REGISTERED.setdefault(backend.name, backend)


def loaded_backends() -> list[ModelBackend]:
    """Every usable backend, entry-point and programmatic, sorted by name."""
    _load_entry_points()
    return [_REGISTERED[name] for name in sorted(_REGISTERED)]


def backend_adapters() -> list[BackendModelAdapter]:
    """Adapter per backend model, named ``backend:model`` for provenance."""
    adapters: list[BackendModelAdapter] = []
    for backend in loaded_backends():
        for model_name in sorted(backend.models):
            adapters.append(BackendModelAdapter(
                f"{backend.name}:{model_name}", backend.models[model_name],
            ))
    return adapters


def resolve_backend_model(candidate: str) -> ModelFn | None:
    """The callable behind a namespaced candidate name, or None."""
    backend_name, _, model_name = candidate.partition(":")
    if not model_name:
        return None
    _load_entry_points()
    backend = _REGISTERED.get(backend_name)
    return backend.models.get(model_name) if backend else None


def backends_fingerprint() -> list[str] | None:
    """``name==version`` per installed backend, for artifact IDs — a
    different candidate set is a different task. None when no backend is
    installed, so IDs predating this layer are unchanged."""
    loaded = loaded_backends()
    return [f"{b.name}=={b.version}" for b in loaded] or None
