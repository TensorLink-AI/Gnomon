"""Root-level Hermes plugin entry point.

`hermes plugins install <repo-url>` expects `plugin.yaml` and `__init__.py`
at the repository root; the actual plugin implementation lives in
``integrations/hermes/`` and is re-exported here unchanged. This file is not
part of the ``aion`` Python package (see ``src/aion``).
"""

from .integrations.hermes import (  # noqa: F401
    check_aion_available,
    handle_aion_capabilities,
    handle_aion_forecast,
    handle_aion_inspect,
    make_propose_context_handler,
    register,
)
