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
    retired: bool = False
    retirement_reason: str | None = None


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
    "reasoningbench": BenchmarkContract(
        "retired",
        "historical instrument retained for reproducibility only",
        "No product or reasoning-uplift claim: the original published harness "
        "exposed answer-bearing fields and its headline was withdrawn.",
        retired=True,
        retirement_reason=(
            "Answer-bearing treatment packets made the historical headline "
            "a transcription result; use DossierBench and "
            "DiscriminationBench for the replacement claim boundaries."
        ),
    ),
    "volatilitybench": BenchmarkContract("engine", "fitted volatility executable", "Volatility estimation and calibration only."),
    "propertybench": BenchmarkContract("engine", "fitted temporal-property executables", "Independent property classification and calibration only."),
    "trendanswerbench": BenchmarkContract(
        "mixed",
        "production typed trend executable and bounded real-agent preservation probe",
        "Seasonally adjusted trend direction, numeric coherence, calibration, "
        "and abstention are engine claims; the agent shard tests exact contract "
        "preservation, not general reasoning uplift.",
        ("evidence",),
    ),
    "anomalyeventbench": BenchmarkContract(
        "mixed",
        "production investigation and graded anomaly-detection macros",
        "Event-level anomaly precision, recall, regime attribution, rebound "
        "deduplication, and detector-selection disclosure; not causal "
        "diagnosis or general agent reasoning.",
    ),
    "transitionbench": BenchmarkContract("engine", "production TemporalEvidence computation", "Observed-transition evidence quality only."),
    "outcomelearningbench": BenchmarkContract(
        "policy", "production publication and tracking outcome loop",
        "Prequential same-series candidate learning, cutoff discipline, "
        "proposer isolation, reversal response, and safety invariants; not "
        "LLM forecast accuracy."),
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
    "calibrationactionbench": BenchmarkContract(
        "policy", "production publication calibration gate",
        "Calibration/action authority matrix only; not forecast accuracy."),
    "claimbench": BenchmarkContract(
        "safety_contract", "production forecast runtime and renderers",
        "Claim, support, headline, and artifact coherence; not estimator uplift."),
    "decisioncontractbench": BenchmarkContract(
        "safety_contract", "production agent-response decision contract",
        "Typed conclusion preservation and authority invariants; not answer accuracy."),
    "decisioninputbench": BenchmarkContract(
        "safety_contract", "production decision-input evaluator",
        "Malformed/non-identifying input rejection and repair behavior only."),
    "hierarchybench": BenchmarkContract(
        "engine", "production bottom-up hierarchical forecast runtime",
        "Hierarchy coherence, leaf immutability, and forecast error by tier; agent behavior excluded."),
    "jitterbench": BenchmarkContract(
        "safety_contract", "production timestamp loading and repair path",
        "Bounded timestamp alignment, disclosure, and typed refusal only."),
    "jointhorizonbench": BenchmarkContract(
        "policy", "production horizon-event and breach-decision executables",
        "Joint-event calibration, decision regret, and authority; not point-forecast superiority."),
    "multivariatebench": BenchmarkContract(
        "engine", "production multivariate forecast path",
        "Held-out multivariate candidate admission and error; agent behavior excluded."),
    "recoverybench": BenchmarkContract(
        "safety_contract", "production MCP response and recovery boundary",
        "Executable recovery, canonical immutability, and bounded retry behavior only."),
    "reliabilitybench": BenchmarkContract(
        "safety_contract", "production artifact publication and MCP load path",
        "Atomicity, integrity, concurrency, and local-load reliability only."),
    "routingbench": BenchmarkContract(
        "policy", "production adapter outcome ledger and promotion policy",
        "Prequential promotion, drift response, and route authority; not model accuracy."),
    "seasonalbench": BenchmarkContract(
        "engine", "production season detection, evaluation, and support rendering",
        "Seasonal-period admission, harmful departure, and support precision; agent behavior excluded."),
    "capabilitybench": BenchmarkContract(
        "engine", "production temporal capability registry executables",
        "Capability classification, regression, and decomposition behavior; not agent reasoning."),
}


def as_dict(name: str) -> dict[str, object]:
    contract = CATALOG[name]
    return {
        "layer": contract.layer,
        "gnomon_path": contract.gnomon_path,
        "claim_boundary": contract.claim_boundary,
        "harness_conditions": list(contract.harness_conditions),
        "retired": contract.retired,
        "retirement_reason": contract.retirement_reason,
    }
