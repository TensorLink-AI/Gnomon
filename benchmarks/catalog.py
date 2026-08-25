"""Authoritative benchmark claim boundaries.

Not every useful benchmark should drive an LLM through MCP. A model, compiler,
or safety invariant is clearer when agent behaviour is excluded. This catalog
prevents lower-layer evidence being presented as agent-reasoning evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkContract:
    layer: str
    gnomon_path: str
    claim_boundary: str
    harness_conditions: tuple[str, ...] = ()


CATALOG: dict[str, BenchmarkContract] = {
    "cik": BenchmarkContract("reasoning_harness", "matched LLM drives real MCP tools", "Context-aware agent forecasting and abstention; RCRPS is an engine outcome.", ("gnomon-mcp",)),
    "anomllm": BenchmarkContract("engine", "deterministic detection runtime", "Anomaly-detector quality only; not general agent reasoning."),
    "mtbench": BenchmarkContract("reasoning_harness", "matched LLM drives real MCP tools", "Agent temporal QA/forecast behaviour on supported forecasting families.", ("mcp",)),
    "timesage_mt": BenchmarkContract("reasoning_harness", "matched LLM drives the Gnomon tool loop", "Multi-turn reasoning on mechanically verifiable fields.", ("gnomon-tools",)),
    "temporalbench": BenchmarkContract("reasoning_harness", "matched LLM drives a real MCP profile", "All-tier agent behaviour; forecast and choice metrics stay separate.", ("gnomon-mcp",)),
    "compilerbench": BenchmarkContract("compiler", "production typed TemporalQuestion compiler", "Language-to-intent fidelity, not answer correctness."),
    "leaktrap": BenchmarkContract("safety_contract", "bitemporal runtime and immutable snapshots", "Structural non-leakage; mean error alone is not a trust result."),
    "workflow": BenchmarkContract("reasoning_harness", "matched LLM drives real MCP profiles", "End-to-end correctness, preservation, calls, tokens, and trust.", ("core", "describe", "evidence", "mega", "full")),
    "contextbench": BenchmarkContract("mixed", "engine arms and real MCP surface arms", "Engine arms test admission; surface arms test agent use and preservation.", ("core", "describe", "evidence", "mega", "full")),
    "reasoningbench": BenchmarkContract("reasoning_harness", "matched LLM receives the compact production evidence packet", "Evidence-assisted synthesis; field/property regressions remain first-class."),
    "volatilitybench": BenchmarkContract("engine", "fitted volatility executable", "Volatility estimation and calibration only."),
    "propertybench": BenchmarkContract("engine", "fitted temporal-property executables", "Independent property classification and calibration only."),
    "transitionbench": BenchmarkContract("engine", "production TemporalEvidence computation", "Observed-transition evidence quality only."),
    "modelbench": BenchmarkContract("engine", "immutable primary forecast pipeline", "Forecast-model admission/accuracy; agent behaviour excluded."),
    "contextcachebench": BenchmarkContract("safety_contract", "production context receipt/cache tools", "Replay parity, identity, and payload economics."),
    "adapterbench": BenchmarkContract("safety_contract", "production forecast-adapter protocol", "Backend conformance and rejection, not accuracy."),
    "admissionbench": BenchmarkContract("policy", "production model-admission policy", "Held-out admission regret and harmful-admission control."),
    "adjudicationbench": BenchmarkContract("policy", "production temporal-evidence adjudicator", "Authority/conflict invariants, not answer quality."),
    "effectbench": BenchmarkContract("safety_contract", "production effect registry, tracking, and decision runtime", "Effect transfer, false influence, calibration, and decision regret."),
    "boundarybench": BenchmarkContract("safety_contract", "production MCP response boundary", "Canonical immutability, fact traceability, sufficiency, rejection repair, and redundant-call attribution; not reasoning accuracy."),
    "discriminationbench": BenchmarkContract("policy", "production held-out hypothesis discrimination", "Known-truth accuracy, separation reliability, and truth-retention of the discriminating-evidence mechanism; not LLM uplift."),
    "dossierbench": BenchmarkContract("reasoning_harness", "matched LLM receives the conclusion packet or the evidence dossier with the selection repair loop", "Packet-design uplift and the transcription margin against deterministic references; mechanism accuracy lives in discriminationbench."),
    "breachbench": BenchmarkContract("reasoning_harness", "matched LLM receives the production forecast/threshold output on real telemetry", "The client job priced in decision cost and regret against realized breaches; not choice accuracy, not the packet mechanism."),
    "recallbench": BenchmarkContract("reasoning_harness", "matched LLM forecasts identical real windows raw and affine-anonymized", "Separates memorized recall of public series from transferable forecasting skill (MASE, affine-invariant); the gate for any LLM-forecast candidate lane. Not choice accuracy."),
}


def as_dict(name: str) -> dict[str, object]:
    contract = CATALOG[name]
    return {
        "layer": contract.layer,
        "gnomon_path": contract.gnomon_path,
        "claim_boundary": contract.claim_boundary,
        "harness_conditions": list(contract.harness_conditions),
    }
