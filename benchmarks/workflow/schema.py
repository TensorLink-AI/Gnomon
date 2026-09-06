"""Versioned, dependency-free contracts for Workflow Bench cases and arms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
CASE_KINDS = {"synthetic", "frozen", "messy", "longitudinal", "multiseries"}
STATUSES = {"answered", "abstained", "error"}
SUPPORT = {"supported", "degraded", "best_effort", "abstained"}
STAGE_FIELDS = {"name", "revealed", "answer_schema", "oracle"}
CASE_FIELDS = {"schema_version", "id", "kind", "domain", "question",
               "available_at_cutoff", "answer_schema", "oracle", "tags", "stages", "episode"}
ORACLE_FIELDS = {"numbers", "tolerances", "choices", "required_disclosures",
                 "forbidden_claims", "allowed_support", "should_abstain",
                 "requires_repair", "requires_tracking",
                 "requires_publish_parity", "requires_quote_match",
                 "required_facts", "engine_required_facts", "choice_aliases",
                 "forecast"}
OBSERVATION_FIELDS = {"case_id", "status", "support", "numbers", "choices",
                      "disclosures", "claims", "temporal_leakage",
                      "publish_matches_evaluated", "repair_completed",
                      "tracking_completed", "quote_matches", "tool_calls",
                      "cumulative_tokens", "response_tokens", "latency_seconds",
                      "metadata", "facts", "evaluated_fingerprint",
                      "published_fingerprint", "headline_numbers",
                      "artifact_numbers", "stage_results"}
OBSERVATION_FIELDS |= {"engine_facts", "cost_usd"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _optional_bool(value: Any, name: str, case_id: str) -> bool | None:
    _require(value is None or isinstance(value, bool),
             f"case {case_id}: {name} must be boolean or null")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    _require(not unknown, f"{context}: unknown fields: {unknown}")


@dataclass(frozen=True)
class Oracle:
    numbers: dict[str, float] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    choices: dict[str, str] = field(default_factory=dict)
    required_disclosures: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    allowed_support: tuple[str, ...] = ("supported", "degraded", "best_effort")
    should_abstain: bool = False
    requires_repair: bool = False
    requires_tracking: bool = False
    requires_publish_parity: bool = False
    requires_quote_match: bool = False
    required_facts: dict[str, Any] = field(default_factory=dict)
    engine_required_facts: dict[str, Any] = field(default_factory=dict)
    choice_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    forecast: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Oracle":
        _reject_unknown(value, ORACLE_FIELDS, "oracle")
        numbers = {str(k): float(v) for k, v in (value.get("numbers") or {}).items()}
        tolerances = {str(k): float(v) for k, v in (value.get("tolerances") or {}).items()}
        _require(all(math.isfinite(v) for v in numbers.values()), "oracle numbers must be finite")
        _require(all(v >= 0 for v in tolerances.values()), "tolerances must be non-negative")
        forecast = value.get("forecast", {})
        _require(isinstance(forecast, dict), "forecast oracle must be an object")
        if forecast:
            _require(set(forecast) == {"keys", "scale", "max_mae"}, "forecast oracle requires keys/scale/max_mae")
            keys = forecast["keys"]
            _require(isinstance(keys, (list, tuple)) and 0 < len(keys) <= 512
                     and all(isinstance(key, str) and key in numbers for key in keys)
                     and len(set(keys)) == len(keys), "forecast keys must be unique numeric oracle keys")
            for key in ("scale", "max_mae"):
                item = forecast[key]
                try:
                    valid = type(item) in (int, float) and math.isfinite(item) and item >= 0
                except OverflowError:
                    valid = False
                _require(valid and (key != "scale" or item > 0), "forecast scale must be positive and max_mae nonnegative, both finite")
            forecast = {**forecast, "keys": list(keys)}
        allowed = tuple(value.get("allowed_support") or ("supported", "degraded", "best_effort"))
        _require(set(allowed) <= SUPPORT, f"unknown allowed_support: {allowed}")
        return cls(
            numbers=numbers,
            tolerances=tolerances,
            choices={str(k): str(v) for k, v in (value.get("choices") or {}).items()},
            required_disclosures=tuple(str(x) for x in value.get("required_disclosures", ())),
            forbidden_claims=tuple(str(x) for x in value.get("forbidden_claims", ())),
            allowed_support=allowed,
            should_abstain=bool(value.get("should_abstain", False)),
            requires_repair=bool(value.get("requires_repair", False)),
            requires_tracking=bool(value.get("requires_tracking", False)),
            requires_publish_parity=bool(value.get("requires_publish_parity", False)),
            requires_quote_match=bool(value.get("requires_quote_match", False)),
            required_facts=dict(value.get("required_facts") or {}),
            engine_required_facts=dict(value.get("engine_required_facts") or {}),
            choice_aliases={str(key): tuple(str(item) for item in aliases)
                            for key, aliases in
                            (value.get("choice_aliases") or {}).items()},
            forecast=forecast,
        )


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    domain: str
    question: str
    available_at_cutoff: dict[str, Any]
    answer_schema: dict[str, Any]
    oracle: Oracle
    tags: tuple[str, ...] = ()
    stages: tuple[dict[str, Any], ...] = ()
    schema_version: int = SCHEMA_VERSION
    episode: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Case":
        _reject_unknown(value, CASE_FIELDS, "case")
        version = int(value.get("schema_version", SCHEMA_VERSION))
        _require(version == SCHEMA_VERSION, f"unsupported case schema_version {version}")
        case_id = str(value.get("id", "")).strip()
        kind = str(value.get("kind", "")).strip()
        _require(bool(case_id), "case id is required")
        _require(kind in CASE_KINDS, f"case {case_id}: unknown kind {kind!r}")
        available = value.get("available_at_cutoff")
        _require(isinstance(available, dict), f"case {case_id}: available_at_cutoff must be an object")
        _require("future" not in available, f"case {case_id}: cutoff input may not contain a future key")
        question = str(value.get("question", "")).strip()
        _require(bool(question), f"case {case_id}: question is required")
        raw_answer_schema = value.get("answer_schema") or {}
        _require(isinstance(raw_answer_schema, dict),
                 f"case {case_id}: answer_schema must be an object")
        _reject_unknown(raw_answer_schema,
                        {"numbers", "choices", "facts", "choice_sources"},
                        f"case {case_id} answer_schema")
        answer_schema = {
            kind: tuple(str(item) for item in raw_answer_schema.get(kind, ()))
            for kind in ("numbers", "choices", "facts")
        }
        raw_choice_sources = raw_answer_schema.get("choice_sources") or {}
        _require(isinstance(raw_choice_sources, dict),
                 f"case {case_id}: choice_sources must be an object")
        answer_schema["choice_sources"] = {
            str(key): str(source) for key, source in raw_choice_sources.items()}
        oracle = Oracle.from_dict(value.get("oracle") or {})
        _require(set(oracle.numbers) <= set(answer_schema["numbers"]),
                 f"case {case_id}: numeric oracle keys missing from answer_schema")
        _require(set(oracle.choices) <= set(answer_schema["choices"]),
                 f"case {case_id}: choice oracle keys missing from answer_schema")
        _require(set(answer_schema["choice_sources"]) <=
                 set(answer_schema["choices"]),
                 f"case {case_id}: canonical choice source keys missing from answer_schema")
        _require(set(oracle.required_facts) <= set(answer_schema["facts"]),
                 f"case {case_id}: required fact keys missing from answer_schema")
        _require(set(oracle.engine_required_facts) <= set(answer_schema["facts"]),
                 f"case {case_id}: engine fact keys missing from answer_schema")
        stages = tuple(dict(stage) for stage in value.get("stages", ()))
        for stage in stages:
            _reject_unknown(stage, STAGE_FIELDS, f"case {case_id} stage")
        _require(all(stage.get("name") in {"repair", "outcome"} for stage in stages),
                 f"case {case_id}: stages must be named repair or outcome")
        _require(len({stage["name"] for stage in stages}) == len(stages),
                 f"case {case_id}: duplicate stage names")
        for stage in stages:
            stage_schema = stage.get("answer_schema") or {}
            _reject_unknown(stage_schema, {"numbers", "choices", "facts"},
                            f"case {case_id} {stage['name']} answer_schema")
            stage_oracle = Oracle.from_dict(stage.get("oracle") or {})
            _require(set(stage_oracle.numbers) <= set(stage_schema.get("numbers", ())),
                     f"case {case_id}: stage numeric oracle keys missing from answer_schema")
            _require(set(stage_oracle.choices) <= set(stage_schema.get("choices", ())),
                     f"case {case_id}: stage choice oracle keys missing from answer_schema")
            _require(set(stage_oracle.required_facts) <= set(stage_schema.get("facts", ())),
                     f"case {case_id}: stage fact oracle keys missing from answer_schema")
        _require(not oracle.requires_repair or any(s["name"] == "repair" for s in stages),
                 f"case {case_id}: repair oracle requires a repair stage")
        _require(not oracle.requires_tracking or any(s["name"] == "outcome" for s in stages),
                 f"case {case_id}: tracking oracle requires an outcome stage")
        episode = value.get("episode", ())
        _require(isinstance(episode, (list, tuple)), "episode must be an ordered phase list")
        if episode:
            _require(not stages and 2 <= len(episode) <= 8, "episode requires2..8 phases and no historical stages")
            names = []
            for phase in episode:
                _require(isinstance(phase, dict) and set(phase) == STAGE_FIELDS, "episode phase requires name/revealed/answer_schema/oracle")
                name = phase["name"]
                _require(isinstance(name, str) and 0 < len(name) <= 64 and name not in names, "episode phase names must be unique bounded strings")
                names.append(name)
                _require(isinstance(phase["revealed"], dict), "phase revealed input must be an object")
                # Reuse real answer/oracle validation without treating these as
                # historical host-compiled repair/outcome stages.
                cls.from_dict({"id": case_id, "kind": kind, "question": question,
                               "available_at_cutoff": {}, "answer_schema": phase["answer_schema"],
                               "oracle": phase["oracle"]})
            _require(episode[0]["revealed"] == {}, "initial episode data belongs in available_at_cutoff")
            _require(Oracle.from_dict(episode[-1]["oracle"]) == oracle,
                     "case oracle must equal final episode oracle")
        return cls(
            id=case_id, kind=kind, domain=str(value.get("domain", "unknown")),
            question=question, available_at_cutoff=available,
            answer_schema=answer_schema, oracle=oracle,
            tags=tuple(str(x) for x in value.get("tags", ())), stages=stages,
            schema_version=version,
            episode=tuple(dict(phase) for phase in episode),
        )


@dataclass(frozen=True)
class Observation:
    case_id: str
    status: str
    support: str
    numbers: dict[str, float] = field(default_factory=dict)
    choices: dict[str, str] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    temporal_leakage: bool | None = None
    publish_matches_evaluated: bool | None = None
    repair_completed: bool | None = None
    tracking_completed: bool | None = None
    quote_matches: bool | None = None
    tool_calls: int = 0
    cumulative_tokens: int = 0
    response_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    # Deterministically harvested from the tool response.  Keep separate from
    # ``facts`` (the agent's final envelope) so engine-contract completeness
    # and agent preservation cannot be conflated.
    engine_facts: dict[str, Any] = field(default_factory=dict)
    evaluated_fingerprint: str | None = None
    published_fingerprint: str | None = None
    headline_numbers: dict[str, float] = field(default_factory=dict)
    artifact_numbers: dict[str, float] = field(default_factory=dict)
    stage_results: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Observation":
        _reject_unknown(value, OBSERVATION_FIELDS, "observation")
        case_id = str(value.get("case_id", "")).strip()
        status, support = str(value.get("status", "")), str(value.get("support", ""))
        _require(bool(case_id), "observation case_id is required")
        _require(status in STATUSES, f"case {case_id}: unknown status {status!r}")
        _require(support in SUPPORT, f"case {case_id}: unknown support {support!r}")
        _require(status != "abstained" or support == "abstained",
                 f"case {case_id}: abstained status requires abstained support")
        _require(support != "abstained" or status in {"abstained", "error"},
                 f"case {case_id}: abstained support requires abstained/error status")
        raw_numbers = {str(k): float(v) for k, v in (value.get("numbers") or {}).items()}
        _require(all(math.isfinite(v) for v in raw_numbers.values()),
                 f"case {case_id}: numbers must be finite")
        temporal_leakage = _optional_bool(value.get("temporal_leakage"),
                                          "temporal_leakage", case_id)
        from .accounting import FIELDS
        metadata = dict(value.get("metadata") or {})
        known = metadata.get("resource_fields", [key for key in FIELDS if value.get(key) is not None])
        _require(isinstance(known, list) and all(isinstance(key, str) and key in FIELDS for key in known),
                 "resource_fields must list known resource names")
        resources = {}
        for key in FIELDS:
            item = value.get(key)
            if item is not None:
                try:
                    valid = type(item) in (int, float) and math.isfinite(item) and item >= 0
                except OverflowError:
                    valid = False
                _require(valid,
                         f"{key} must be a finite nonnegative number")
                if key in FIELDS[:3]:
                    _require(type(item) is int, f"{key} must be an integer")
            resources[key] = item if item is not None else (None if key == "cost_usd" else 0)
        metadata["resource_fields"] = sorted(key for key in known if value.get(key) is not None)
        return cls(
            case_id=case_id, status=status, support=support,
            numbers=raw_numbers,
            choices={str(k): str(v) for k, v in (value.get("choices") or {}).items()},
            disclosures=tuple(str(x) for x in value.get("disclosures", ())),
            claims=tuple(str(x) for x in value.get("claims", ())),
            temporal_leakage=temporal_leakage,
            publish_matches_evaluated=_optional_bool(value.get("publish_matches_evaluated"), "publish_matches_evaluated", case_id),
            repair_completed=_optional_bool(value.get("repair_completed"), "repair_completed", case_id),
            tracking_completed=_optional_bool(value.get("tracking_completed"), "tracking_completed", case_id),
            quote_matches=_optional_bool(value.get("quote_matches"), "quote_matches", case_id),
            **resources,
            metadata=metadata,
            facts=dict(value.get("facts") or {}),
            engine_facts=dict(value.get("engine_facts") or {}),
            evaluated_fingerprint=(str(value["evaluated_fingerprint"])
                                   if value.get("evaluated_fingerprint") is not None else None),
            published_fingerprint=(str(value["published_fingerprint"])
                                   if value.get("published_fingerprint") is not None else None),
            headline_numbers={str(k): float(v) for k, v in
                              (value.get("headline_numbers") or {}).items()},
            artifact_numbers={str(k): float(v) for k, v in
                              (value.get("artifact_numbers") or {}).items()},
            stage_results=dict(value.get("stage_results") or {}),
        )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    from benchmarks.workflow.agent_metrics import _read_records
    yield from _read_records(path)


def load_cases(path: str | Path) -> list[Case]:
    cases = [Case.from_dict(row) for row in _read_jsonl(Path(path))]
    ids = [case.id for case in cases]
    _require(len(ids) == len(set(ids)), "duplicate case ids")
    _require(bool(cases), "case bundle is empty")
    return cases


def load_observations(path: str | Path) -> list[Observation]:
    observations = []
    for row in _read_jsonl(Path(path)):
        # Old serialized defaults do not prove measured zero or complete history.
        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("resource_fields", [])
        observations.append(Observation.from_dict({**row, "metadata": metadata}))
    ids = [row.case_id for row in observations]
    _require(len(ids) == len(set(ids)), "duplicate observation case_ids")
    return observations
