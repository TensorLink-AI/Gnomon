"""The routed arm: an agent with Gnomon available, free to choose.

The existing arms measure the wrong deployment. `gnomon-agent` is
*mandatory* Gnomon — the model must use the tool even on tasks where
the context text carries the whole answer — and `control` is the model
alone. Real agents get a third posture: Gnomon as a tool they may call
or skip. This arm measures that posture, per task:

1. **Route.** The model reads the task's context and decides, before
   any forecasting: call Gnomon, or answer directly. The decision and
   its stated reason are recorded in ``extra_info`` for every run, so
   routing quality is analysable afterwards (was Gnomon skipped where
   it would have won?).
2. **Fallback on abstention.** If the Gnomon path abstains, the run
   falls back to answering directly instead of scoring an abstention —
   this is "reading the support signal", the behaviour a real agent
   should have, and the fallback is recorded as such.

An unparseable routing decision defaults to Gnomon (the conservative
arm-preserving choice) with the parse failure recorded.

Scores from this arm compare against both single-posture arms on
matched task-seeds. What this arm may NOT be quoted as: evidence about
Gnomon's own forecasting quality — a routed win can come entirely from
the model knowing when not to use the tool.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.gnomon_forecaster import (  # noqa: E402
    GnomonAbstained,
    GnomonForecaster,
    build_context_text,
)

#: Deliberately symmetric: the first version of this prompt described
#: gnomon in glowing terms and direct in two dismissive clauses, and the
#: measured result was a constant function — 68 of 68 runs chose gnomon.
#: A router that always picks one arm is not a router. The prompt now
#: names the concrete condition under which each choice wins and asks
#: for a one-sentence analysis of the context before the decision.
ROUTING_INSTRUCTIONS = """\
You are about to produce a probabilistic forecast for a time series.
Two methods are available. Decide which will produce the lower forecast
error for THIS task, then reply with a single JSON object.

"gnomon" — a statistical engine that forecasts from the numeric
history. Context sentences that STATE NUMBERS (bounds, levels,
multiples of the usual level, stated trend cessations) can influence
it, verified deterministically. Context that is qualitative — weather
states, unquantified events, cause-effect descriptions without numbers
— has ZERO influence on it, no matter how decisive it is.

"direct" — you write the forecast yourself from the full context and
history, with free-form reasoning but no calibration machinery.

Choose "direct" when the qualitative content of the context determines
the future shape (state changes like clear/cloudy, described regime
switches, causal explanations that carry the answer without stating
values). Choose "gnomon" when the future mostly follows the numeric
history, or when the decisive context states numbers it can verify.

Reply with exactly:
{"contains": "<one sentence: what the context contains>",
 "route": "gnomon" or "direct",
 "why": "<one sentence>"}
"""

_ROUTE_PATTERN = re.compile(r'"route"\s*:\s*"(gnomon|direct)"', re.IGNORECASE)
_WHY_PATTERN = re.compile(r'"why"\s*:\s*"([^"]{0,300})"')


def parse_route(completion: str) -> tuple[str, str, str | None]:
    """Return (route, rationale, note). Unparseable defaults to gnomon."""
    match = _ROUTE_PATTERN.search(completion or "")
    if not match:
        return "gnomon", "", "routing output unparseable; defaulted to gnomon"
    why = _WHY_PATTERN.search(completion or "")
    return match.group(1).lower(), (why.group(1) if why else ""), None


class RoutedForecaster:
    """CiK baseline: the model routes each task to Gnomon or itself."""

    __version__ = "0.1.0"

    def __init__(
        self,
        openrouter_model: str,
        *,
        temperature: float = 1.0,
        work_dir: str | None = None,
        future_context: bool = False,
        structural_context: bool = False,
        fail_on_invalid: bool = True,
        gnomon: Any = None,
        direct: Any = None,
        client: Any = None,
    ) -> None:
        self.openrouter_model = openrouter_model
        self.gnomon = gnomon or GnomonForecaster(
            mode="agent", openrouter_model=openrouter_model,
            temperature=temperature, work_dir=work_dir,
            future_context=future_context,
            structural_context=structural_context,
        )
        self.client = client or self.gnomon.client
        self._direct = direct
        self._direct_kwargs = {
            "temperature": temperature,
            "fail_on_invalid": fail_on_invalid,
        }

    @property
    def direct(self) -> Any:
        if self._direct is None:
            # Lazy: the official DirectPrompt base class needs the
            # cik_benchmark checkout, which unit tests do not have.
            from benchmarks.cik.openrouter_direct_prompt import (
                OpenRouterDirectPrompt,
            )
            self._direct = OpenRouterDirectPrompt(
                openrouter_model=self.openrouter_model,
                **self._direct_kwargs,
            )
        return self._direct

    @property
    def cache_name(self) -> str:
        return "RoutedForecaster_" + self.gnomon.cache_name.split(
            "GnomonForecaster_", 1)[-1]

    def __str__(self) -> str:
        return self.cache_name

    # -- internals ---------------------------------------------------------
    def _route(self, task_instance: Any) -> tuple[str, str, str | None]:
        context = build_context_text(task_instance)
        horizon = len(task_instance.future_time)
        prompt = (
            f"{ROUTING_INSTRUCTIONS}\n"
            f"Forecast horizon: {horizon} steps.\n\n"
            f"Task context:\n{context or '(no textual context)'}\n"
        )
        try:
            completion = self.client.completions(
                [{"role": "user", "content": prompt}], n=1,
            )[0]
        except Exception as error:
            return "gnomon", "", f"routing call failed ({error}); defaulted"
        return parse_route(completion)

    @staticmethod
    def _unpack(result: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], dict(result[1] or {})
        return result, {}

    def __call__(self, task_instance: Any, n_samples: int):
        started = time.time()
        route, why, note = self._route(task_instance)
        record: dict[str, Any] = {
            "adapter": self.cache_name,
            "route": route,
            "route_rationale": why,
            "fallback_from_abstention": False,
        }
        if note:
            record["route_note"] = note
        if route == "direct":
            samples, extra = self._unpack(self.direct(task_instance, n_samples))
        else:
            try:
                samples, extra = self._unpack(
                    self.gnomon(task_instance, n_samples))
            except GnomonAbstained as abstained:
                # The support signal read as a real agent would read it:
                # an honest refusal is the cue to answer yourself, and
                # the fallback is recorded, never hidden.
                record["fallback_from_abstention"] = True
                record["abstention_reasons"] = list(abstained.reasons)
                record["route"] = "direct"
                samples, extra = self._unpack(
                    self.direct(task_instance, n_samples))
        extra.update(record)
        extra["total_time"] = time.time() - started
        return samples, extra
