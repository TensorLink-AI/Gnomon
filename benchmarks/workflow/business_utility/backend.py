"""Benchmark-only threshold overload around the existing isolated MCP backend.

The overload computes supplied marginals, not a shipped forecasting capability.
All existing calls still go to the actual service unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json

from benchmarks.workflow.bounded_agent import ToolReply
from benchmarks.workflow.service_backend import CombinedBackend
from .reference import threshold_probability

PROVIDER = "benchmark_supplied_quantiles"
OVERLOAD = {"type": "object", "additionalProperties": False,
    "required": ["provider", "threshold"], "properties": {
        "provider": {"const": PROVIDER},
        "threshold": {"type": "object", "additionalProperties": False,
            "required": ["level", "direction"], "properties": {
                "level": {"type": "number"}, "direction": {"enum": ["above", "below"]}}}}}


class ThresholdReferenceBackend:
    startup_cost_usd = 0

    def __init__(self, delegate, case):
        self.delegate = delegate
        raw = case.get("available_at_cutoff", {}).get("files", {}).get("quantiles.json")
        self.fixture = json.loads(raw) if raw is not None else None
        self.provenance = {"delegate": delegate.provenance,
                           "threshold_contract": "benchmark prototype; supplied continuous per-step marginals only",
                           "fixture_sha256": hashlib.sha256(raw.encode()).hexdigest() if raw else None}

    def tools(self):
        inventory = copy.deepcopy(self.delegate.tools())
        if self.fixture is not None:
            for tool in inventory:
                if tool["name"] == "gnomon_forecast":
                    tool["inputSchema"] = {"oneOf": [tool["inputSchema"], OVERLOAD]}
                    tool["description"] += (" BENCHMARK PROTOTYPE ONLY: provider=benchmark_supplied_quantiles "
                        "with threshold={level,direction} computes the supplied quantiles.json scenario's per-step "
                        "marginal using its declared continuous piecewise-linear CDF. No model fitting or path events.")
        return inventory

    def call(self, name, arguments, *, timeout):
        if name == "gnomon_forecast" and arguments.get("provider") == PROVIDER:
            try:
                if self.fixture is None or set(arguments) != {"provider", "threshold"}:
                    raise ValueError("prototype requires supplied quantile fixture and provider/threshold only")
                threshold = arguments["threshold"]
                if not isinstance(threshold, dict) or set(threshold) != {"level", "direction"}:
                    raise ValueError("only level and direction supported; path events refused")
                if self.fixture.get("distribution") != "continuous_piecewise_linear_cdf_finite_support":
                    raise ValueError("declared continuous finite-support CDF required")
                result = threshold_probability(self.fixture["knots"], **threshold)
                value = {"status": "ok", "provider": PROVIDER, "prototype": True,
                         "fixture_sha256": self.provenance["fixture_sha256"],
                         "per_step": [{"step": self.fixture["step"], **result}]}
                return ToolReply({"content": [{"type": "text", "text": json.dumps(value)}],
                                  "isError": False}, 0)
            except (ValueError, TypeError, KeyError):
                return ToolReply({"content": [{"type": "text", "text": "Invalid benchmark marginal request; see declared contract."}],
                                  "isError": True}, 0)
        return self.delegate.call(name, arguments, timeout=timeout)

    def close(self):
        self.delegate.close()


def lean(**kwargs):
    return CombinedBackend(arm="lean", **kwargs)


def full(**kwargs):
    delegate = CombinedBackend(arm="lean", **kwargs)
    try:
        return ThresholdReferenceBackend(delegate, kwargs["case"])
    except BaseException:
        delegate.close()
        raise
