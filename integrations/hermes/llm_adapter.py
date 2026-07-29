"""Aion's LLMAdapter implemented over Hermes's host-owned ``ctx.llm`` facade.

The host owns provider routing, auth, model choice, and budgets; this
adapter only carries an Aion-authored prompt and JSON schema across, and
returns the parsed object. No API keys, no provider config, no numbers.
"""

from __future__ import annotations

import json
from typing import Any


class HermesLLMAdapter:
    """Adapter satisfying ``aion.llm.LLMAdapter`` using ``ctx.llm``."""

    available = True

    def __init__(self, ctx: Any, purpose: str = "aion workflow"):
        self._ctx = ctx
        self._purpose = purpose

    def complete(self, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]:
        result = self._ctx.llm.complete_structured(
            instructions=prompt,
            input=[{"type": "text", "text": "Respond now with schema-conforming JSON only."}],
            json_schema=response_schema,
            schema_name="aion_workflow_response",
            purpose=self._purpose,
        )
        parsed = getattr(result, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        return json.loads(getattr(result, "text", "") or "{}")
