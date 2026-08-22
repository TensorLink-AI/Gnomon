"""Control condition for CiK: the official DirectPrompt baseline, with its
model routing generalised to any OpenRouter model.

The official ``cik_benchmark.baselines.direct_prompt.DirectPrompt`` already
implements the paper's LLM protocol — the exact prompt, the
``<forecast>``-tag output format, and rejection sampling until ``n_samples``
valid forecasts are collected. Its bundled OpenRouter client, however, only
recognises a hard-coded set of model prefixes (llama/mistral/qwen). This
subclass keeps every part of the official protocol verbatim and swaps in a
client that passes any OpenRouter model identifier through unchanged, so
the same control can be run with e.g. ``openai/gpt-4o`` or
``anthropic/claude-sonnet-4``.

Nothing about prompting, parsing, retry counts, or scoring is overridden
here. ``DirectPrompt`` configuration (notably ``fail_on_invalid``, official
default ``True``) is passed through from the runner; ``run_cik.py`` keeps
the official default and exposes ``--no-fail-on-invalid`` as a disclosed
deviation. ``constrained_decoding`` is forced off because OpenRouter offers
no equivalent of the official local-inference constrained decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402

try:
    from cik_benchmark.baselines.direct_prompt import DirectPrompt
except ModuleNotFoundError as error:  # pragma: no cover - environment only
    if error.name and not error.name.startswith("cik_benchmark"):
        raise
    raise ImportError(
        "The official CiK benchmark package is required for the control "
        "condition. Install it with:\n"
        "  pip install -r benchmarks/cik/requirements.txt"
    ) from error


class OpenRouterDirectPrompt(DirectPrompt):
    """DirectPrompt whose completions come from OpenRouter.

    Parameters
    ----------
    openrouter_model:
        Full OpenRouter model id (``provider/model``), passed through
        verbatim.
    """

    __version__ = DirectPrompt.__version__

    def __init__(self, openrouter_model: str, temperature: float = 1.0, **kwargs):
        self.openrouter_model = openrouter_model
        self._client = OpenRouterClient(openrouter_model, temperature=temperature)
        # The "openrouter-" prefix keeps DirectPrompt's own OpenRouter
        # behaviour (batch-of-1 retry budget, per-call cost accounting).
        super().__init__(
            model=f"openrouter-{openrouter_model}",
            temperature=temperature,
            constrained_decoding=False,
            **kwargs,
        )

    def _call(self, model, messages, n=1, max_tokens=10000, temperature=None):
        return self._client.chat(
            messages,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def get_client(self):
        # A bound method, not a closure. ``DirectPrompt.__init__`` assigns
        # this to ``self.client``, and ``evaluate_all_tasks`` pickles the
        # baseline to reach its worker pool — a local function is
        # unpicklable, so any ``--max-parallel > 1`` control run died at
        # launch with "Can't pickle local object". Signature and call path
        # are unchanged (``inspect.signature`` drops ``self``, so
        # DirectPrompt's ``future_timestamps`` probe still takes the same
        # branch); only the object's picklability differs.
        return self._call

    @property
    def usage_summary(self):
        return self._client.usage_summary

    @property
    def cache_name(self) -> str:
        # Keep the official naming scheme, but base it on the full
        # OpenRouter model id so distinct models never share a cache.
        args = [
            f"model=openrouter-{self.openrouter_model.replace('/', '-')}",
            f"use_context={self.use_context}",
            f"fail_on_invalid={self.fail_on_invalid}",
            f"n_retries={self.n_retries}",
            f"temperature={self.temperature}",
        ]
        return f"{self.__class__.__name__}_" + "_".join(args)
