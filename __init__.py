"""Root-level Hermes plugin entry point.

`hermes plugins install <repo-url>` expects `plugin.yaml` and `__init__.py`
at the repository root; the actual plugin implementation lives in
``integrations/hermes/`` and is re-exported here unchanged. This file is not
part of the ``gnomon`` Python package (see ``src/gnomon``).
"""

from .integrations.hermes import (  # noqa: F401
    check_gnomon_available,
    handle_gnomon_capabilities,
    handle_gnomon_forecast,
    handle_gnomon_inspect,
    make_propose_context_handler,
    register,
)
