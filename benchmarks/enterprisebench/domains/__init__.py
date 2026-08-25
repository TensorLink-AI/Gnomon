"""Domain packs register themselves on import.

Every module in this package is imported automatically, and each one
calls :func:`benchmarks.enterprisebench.harness.register` at import
time. Adding a domain therefore means dropping a module here — the
harness is never edited, and a registry test enforces that.
"""

from __future__ import annotations

import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")
