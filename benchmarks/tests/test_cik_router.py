"""The routed CiK arm: choose Gnomon or answer directly, per task.

Offline: sub-forecasters and the routing client are stubs; what is
under test is the routing decision, the abstention fallback, and the
per-run record that makes routing quality analysable afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.gnomon_forecaster import GnomonAbstained
from benchmarks.cik.router import RoutedForecaster, parse_route


class _Task:
    background = "Solar irradiance for a site."
    constraints = None
    scenario = "At 2022-07-15 06:00:00, the weather will become clear."
    future_time = list(range(7))


class _Client:
    def __init__(self, completion):
        self.completion = completion
        self.prompts = []

    def completions(self, messages, n=1):
        self.prompts.append(messages[0]["content"])
        if isinstance(self.completion, Exception):
            raise self.completion
        return [self.completion]


class _Gnomon:
    cache_name = "GnomonForecaster_mode=agent_model=x-y_future=on"

    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def __call__(self, task, n_samples):
        self.calls += 1
        if self.error:
            raise self.error
        return "gnomon-samples", {"future_context": {"admitted": []},
                                  "support": "supported"}


class _Direct:
    def __init__(self):
        self.calls = 0

    def __call__(self, task, n_samples):
        self.calls += 1
        return "direct-samples"  # official baselines may return bare samples


def _router(client, gnomon=None, direct=None):
    return RoutedForecaster(
        "x/y", gnomon=gnomon or _Gnomon(), direct=direct or _Direct(),
        client=client,
    )


def test_parse_route():
    assert parse_route('{"route": "direct", "why": "text is the answer"}') \
        == ("direct", "text is the answer", None)
    assert parse_route('{"route": "gnomon", "why": "numeric history"}')[0] \
        == "gnomon"
    route, _, note = parse_route("I think probably the engine?")
    assert route == "gnomon" and "defaulted" in note


def test_direct_route_skips_gnomon():
    gnomon, direct = _Gnomon(), _Direct()
    router = _router(
        _Client('{"route": "direct", "why": "the text carries the shape"}'),
        gnomon=gnomon, direct=direct,
    )
    samples, extra = router(_Task(), 25)
    assert samples == "direct-samples"
    assert gnomon.calls == 0 and direct.calls == 1
    assert extra["route"] == "direct"
    assert extra["route_rationale"] == "the text carries the shape"
    assert extra["fallback_from_abstention"] is False


def test_gnomon_route_carries_the_gate_record():
    gnomon, direct = _Gnomon(), _Direct()
    router = _router(_Client('{"route": "gnomon", "why": "numeric"}'),
                     gnomon=gnomon, direct=direct)
    samples, extra = router(_Task(), 25)
    assert samples == "gnomon-samples"
    assert direct.calls == 0
    assert extra["route"] == "gnomon"
    # the census classifier keeps working on routed dumps
    assert "future_context" in extra


def test_abstention_falls_back_to_direct_and_says_so():
    gnomon = _Gnomon(error=GnomonAbstained(["horizon exceeds history"]))
    direct = _Direct()
    router = _router(_Client('{"route": "gnomon", "why": "numeric"}'),
                     gnomon=gnomon, direct=direct)
    samples, extra = router(_Task(), 25)
    assert samples == "direct-samples"
    assert gnomon.calls == 1 and direct.calls == 1
    assert extra["route"] == "direct"
    assert extra["fallback_from_abstention"] is True
    assert extra["abstention_reasons"] == ["horizon exceeds history"]


def test_routing_failure_defaults_to_gnomon_with_a_note():
    gnomon = _Gnomon()
    router = _router(_Client(RuntimeError("network down")), gnomon=gnomon)
    samples, extra = router(_Task(), 25)
    assert samples == "gnomon-samples"
    assert extra["route"] == "gnomon"
    assert "defaulted" in extra["route_note"]


def test_the_task_context_reaches_the_routing_prompt():
    client = _Client('{"route": "gnomon", "why": "n"}')
    _router(client)(_Task(), 25)
    assert "weather will become clear" in client.prompts[0]
    assert "Forecast horizon: 7 steps" in client.prompts[0]


def test_cache_name_carries_the_flag_state():
    router = _router(_Client('{"route": "gnomon", "why": "n"}'))
    assert router.cache_name == \
        "RoutedForecaster_mode=agent_model=x-y_future=on"


def test_route_survey_tally_flags_a_constant_router():
    from benchmarks.cik.route_survey import tally

    constant = [{"task": f"T{i}", "seed": 1, "route": "gnomon", "why": "x"}
                for i in range(20)]
    summary = tally(constant)
    assert summary["degenerate"] is True
    assert summary["dominant_share"] == 1.0
    assert summary["tasks_with_split_decisions"] == []

    mixed = constant[:10] + [
        {"task": "T1", "seed": 2, "route": "direct", "why": "weather words"},
        {"task": "W", "seed": 1, "route": "direct", "why": "qualitative"},
    ]
    summary = tally(mixed)
    assert summary["degenerate"] is False
    assert summary["routes"] == {"gnomon": 10, "direct": 2}
    assert "T1" in summary["tasks_with_split_decisions"]
    assert summary["rationale_samples"]["direct"][0]["why"] == "weather words"
