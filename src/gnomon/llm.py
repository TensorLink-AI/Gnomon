"""LLM adapter boundary: bring your own brain, keep the numbers deterministic.

Gnomon owns every LLM workflow end to end — the prompts, the structured
response schemas, and the deterministic validation applied to whatever the
model returns. The host supplies only the model call. Under Hermes the
adapter wraps the host's ``ctx.llm`` facade; standalone, a provider adapter
(for example OpenRouter) fills the same role; with no adapter configured,
every workflow degrades to deterministic templates.

The permanent invariant, restated from the system design: an adapter may
propose task fields, column mappings, context events, experiments, and
explanations. It may never generate, edit, or override forecast values,
evaluation metrics, uncertainty, model selection, or abstention. Nothing in
this module is reachable from the numerical pipeline.

Status: experimental. The protocol is stable in intent but not yet covered
by the release-gate suite, and no workflow in v0.1 calls it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .contracts import GnomonError


class LLMUnavailable(GnomonError):
    """Raised when a workflow needs an LLM and no adapter is configured.

    Callers must treat this as a signal to fall back to deterministic
    behaviour (templates, explicit user questions), never as a task error.
    """

    def __init__(self, message: str | None = None):
        super().__init__(
            code="LLM_UNAVAILABLE",
            message=message
            or "No LLM adapter is configured; falling back to deterministic behaviour.",
        )


@runtime_checkable
class LLMAdapter(Protocol):
    """Minimal surface a host must implement to power Gnomon's LLM workflows.

    ``complete`` returns a JSON-compatible object that the caller validates
    against ``response_schema`` (JSON Schema). Adapters must not retry into
    unvalidated free text: if the model cannot produce schema-conforming
    output, raise ``GnomonError('LLM_INVALID_RESPONSE', ...)`` so the workflow
    can degrade deterministically.
    """

    def complete(self, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]:
        ...


class NullLLMAdapter:
    """The no-LLM default: every completion raises ``LLMUnavailable``."""

    available = False

    def complete(self, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]:
        raise LLMUnavailable()
