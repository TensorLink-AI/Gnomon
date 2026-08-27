"""BreachBench diagnostic for prior-before-evidence reconciliation.

This runner does not ask the model to rediscover its unaided view. It consumes
the request-bound control row from a completed BreachBench run, host-seals that
pre-evidence answer as a non-authoritative prior, and then asks the same model
to reconcile it with the current immutable Gnomon packet. The realized future
remains scorer-only. This isolates whether one-call evidence anchoring erases
useful model priors without weakening forecast or automation contracts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import threading
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.breachbench.run_breachbench import (
    ARMS, SYSTEM, _arm_metrics, _git_sha, _score, base_prompt, exact_sign_p,
    generate_cases, parse_answer, product_packet, request_identity,
)
from benchmarks.common.envfile import load_env_file
from benchmarks.common.openrouter import OpenRouterClient, extract_json_objects
from gnomon.agent_context import (
    build_temporal_decision_reconciliation,
    seal_temporal_decision_selection,
    seal_temporal_decision_prior,
)


ARM = "reconciled"


def _source_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[(row["case_id"], row["arm"])] = row
    return rows


def _question_sha(case: Any) -> str:
    return hashlib.sha256(base_prompt(case).encode("utf-8")).hexdigest()


def _prior_from_control(row: dict[str, Any], case: Any) -> dict[str, Any]:
    return seal_temporal_decision_prior({
        "breach_expected": bool(row["breach_expected"]),
        "breach_probability": row.get("breach_probability"),
        "first_breach_step": row.get("first_breach_step"),
        "action": row["action"],
    }, question_sha256=_question_sha(case),
       proposer_id="breachbench:pre_evidence_control",
       model=str(row["model"]))


def reconciliation_prompt(case: Any, packet: dict[str, Any],
                          prior: dict[str, Any]) -> str:
    reconciliation = build_temporal_decision_reconciliation(
        packet, prior, question_sha256=_question_sha(case))
    return base_prompt(case) + (
        "\nThe host captured your independent answer before showing Gnomon. "
        "It is a prior, not evidence and never automation authority:\n"
        + json.dumps(prior, separators=(",", ":"))
        + "\nComputed immutable Gnomon evidence:\n"
        + json.dumps(packet, separators=(",", ":"))
        + "\nDeterministic reconciliation contract:\n"
        + json.dumps(reconciliation, separators=(",", ":"))
        + "\nReturn one JSON object: {\"breach_expected\":true|false,"
          "\"first_breach_step\":<1-24 or null>,"
          "\"action\":\"act\"|\"monitor\","
          "\"automation_action\":\"act\"|\"monitor\"|\"withhold\","
          "\"evidence_assessment\":\"breach\"|\"no_breach\"|"
          "\"indeterminate\",\"breach_probability\":<0..1 or null>,"
          "\"selected_source\":\"independent_prior\"|\"immutable_primary\"|"
          "\"synthesis\",\"counterevidence_source\":\"independent_prior\"|"
          "\"immutable_primary\"|\"synthesis\","
          "\"confidence\":\"low\"|\"medium\"|\"high\","
          "\"what_would_change\":\"brief falsifiable condition\"}. "
          "Quote Gnomon's probability_any_breach when available. Reconcile "
          "the independent prior and evidence for the human action; state no "
          "greater support than Gnomon earned. automation_action must be "
          "withhold unless Gnomon explicitly marks automation_eligible true."
    )


def parse_reconciled_answer(
    text: str, case: Any, reconciliation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    answer = parse_answer(text, case.horizon)
    if not answer.get("valid"):
        return answer, None, "base_answer_invalid"
    for payload in extract_json_objects(text):
        if (payload.get("breach_expected") is answer.get("breach_expected")
                and str(payload.get("action") or "").lower()
                == answer.get("action")):
            try:
                selection = seal_temporal_decision_selection(
                    reconciliation, payload)
            except ValueError as error:
                return {"valid": False}, None, str(error)
            return answer, selection, None
    return {"valid": False}, None, "selection_payload_missing"


def _request_sha(case: Any, packet: dict[str, Any], prior: dict[str, Any],
                 args: argparse.Namespace) -> str:
    body = {
        "system": SYSTEM,
        "user": reconciliation_prompt(case, packet, prior),
        "model": args.model, "base_url": args.base_url,
        "temperature": 0, "initial_max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(args: argparse.Namespace, client: Any = None) -> dict[str, Any]:
    source = Path(args.source_run)
    source_summary = json.loads((source / "summary.json").read_text())
    if (source_summary.get("model") != args.model
            or source_summary.get("seed") != args.seed
            or source_summary.get("cases") != args.cases):
        raise ValueError("source run model/seed/case identity does not match")
    cases, corpus, futures = generate_cases(args.seed, args.cases)
    packets = {case.case_id: product_packet(case) for case in cases}
    source_rows = _source_rows(source / "rows.jsonl")
    source_args = SimpleNamespace(
        model=args.model,
        base_url=((source_summary.get("usage") or {}).get("base_url")
                  or args.base_url),
        max_tokens=(source_summary.get("design") or {}).get(
            "initial_max_tokens", 400),
        reasoning_effort=(source_summary.get("design") or {}).get(
            "reasoning_effort"),
    )
    for case in cases:
        for arm in ARMS:
            row = source_rows.get((case.case_id, arm))
            if row is None:
                raise ValueError(f"source run lacks {case.case_id}/{arm}")
            expected = request_identity(case, arm, packets[case.case_id],
                                        source_args)
            if row.get("request_sha256") != expected:
                raise ValueError(
                    f"source row request identity failed for {case.case_id}/{arm}")
        # The held-out future is scorer-only. Hash search is sufficient here
        # because the source harness already performed its stronger marker
        # preflight and this arm adds only source answers + current evidence.
        shown = reconciliation_prompt(
            case, packets[case.case_id],
            _prior_from_control(source_rows[(case.case_id, "control")], case))
        future_blob = json.dumps(futures[case.case_id], separators=(",", ":"))
        if future_blob in shown:
            raise ValueError(f"held-out future leaked for {case.case_id}")

    if client is None:
        load_env_file()
        client = OpenRouterClient(
            args.model, api_key=os.environ.get(args.api_key_env),
            base_url=args.base_url, temperature=0,
            max_tokens=args.max_tokens, max_retries=4,
            reasoning_effort=args.reasoning_effort)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case = next((item for item in cases
                         if item.case_id == row.get("case_id")), None)
            if case is None or row.get("model") != args.model:
                continue
            prior = _prior_from_control(
                source_rows[(case.case_id, "control")], case)
            if row.get("request_sha256") == _request_sha(
                    case, packets[case.case_id], prior, args):
                completed[case.case_id] = row
    lock = threading.Lock()

    def one(case: Any) -> dict[str, Any]:
        prior = _prior_from_control(
            source_rows[(case.case_id, "control")], case)
        reconciliation = build_temporal_decision_reconciliation(
            packets[case.case_id], prior, question_sha256=_question_sha(case))
        text = client.completions([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": reconciliation_prompt(
                case, packets[case.case_id], prior)},
        ], n=1)[0]
        answer, selection, selection_error = parse_reconciled_answer(
            text, case, reconciliation)
        return {
            "case_id": case.case_id, "arm": ARM, "model": args.model,
            "origin": case.origin, "history_length": case.history_length,
            "history_band": case.history_band,
            "truth_breach": case.truth_breach,
            "truth_first_step": case.truth_first_step,
            "request_sha256": _request_sha(
                case, packets[case.case_id], prior, args),
            "selection_valid": selection is not None,
            "selection_error": selection_error,
            "selected_source": (
                selection.get("selected_source") if selection else None),
            "counterevidence_source": (
                selection.get("counterevidence_source") if selection else None),
            "selection_confidence": (
                selection.get("confidence") if selection else None),
            "selection_seal_sha256": (
                selection.get("seal_sha256") if selection else None),
            **_score(answer, case),
        }

    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(one, case): case for case in cases
                   if case.case_id not in completed}
        for future in as_completed(pending):
            case = pending[future]
            try:
                row = future.result()
            except Exception as error:
                failures.append((case.case_id, repr(error)[:300]))
                continue
            with lock:
                completed[case.case_id] = row
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} reconciliation calls failed; "
                           f"rerun --resume; first={failures[0]}")

    reconciled = [completed[case.case_id] for case in cases]
    metrics = {"control": _arm_metrics([
        source_rows[(case.case_id, "control")] for case in cases]),
        "gnomon": _arm_metrics([
            source_rows[(case.case_id, "gnomon")] for case in cases]),
        ARM: _arm_metrics(reconciled)}
    deltas = [source_rows[(case.case_id, "control")]["regret"]
              - completed[case.case_id]["regret"] for case in cases]
    better = sum(delta > 0 for delta in deltas)
    worse = sum(delta < 0 for delta in deltas)
    by_origin: dict[str, list[Any]] = {}
    for case in cases:
        by_origin.setdefault(case.origin, []).append(case)
    origins = sorted(by_origin)
    rng = random.Random(args.seed + 1991)
    bootstrap = []
    for _ in range(2000):
        sampled = [origins[rng.randrange(len(origins))] for _ in origins]
        bootstrap.append(statistics.mean(
            source_rows[(case.case_id, "control")]["regret"]
            - completed[case.case_id]["regret"]
            for origin in sampled for case in by_origin[origin]))
    bootstrap.sort()
    summary = {
        "schema_version": "0.1", "model": args.model,
        "seed": args.seed, "cases": args.cases,
        "source_run": str(source), "evaluated_commit": _git_sha(),
        "corpus": corpus, "metrics": metrics,
        "paired": {
            "reconciled_better": better, "control_better": worse,
            "exact_sign_p": exact_sign_p(better, worse),
            "regret_reduction_vs_control": statistics.mean(deltas),
            "cluster_bootstrap_95": {
                "lower": bootstrap[int(.025 * len(bootstrap))],
                "upper": bootstrap[int(.975 * len(bootstrap))],
                "cluster": "origin_series", "replicates": len(bootstrap)},
        },
        "selection": {
            "valid_rate": statistics.mean(
                row.get("selection_valid") is True for row in reconciled),
            "selected_source_counts": {
                source: sum(row.get("selected_source") == source
                            for row in reconciled)
                for source in ("independent_prior", "immutable_primary",
                               "synthesis")},
            "confidence_counts": {
                level: sum(row.get("selection_confidence") == level
                           for row in reconciled)
                for level in ("low", "medium", "high")},
        },
        "invariants": {
            "primary_forecast_unchanged": True,
            "prior_support": "prior_assisted",
            "automation_eligible": False,
            "source_requests_verified": True,
            "held_out_future_absent": True,
        },
    }
    if hasattr(client, "usage_summary"):
        summary["usage"] = client.usage_summary
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--base-url", default="https://api.engy.ai/v1")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY")
    parser.add_argument("--cases", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--reasoning-effort",
                        choices=("none", "low", "medium", "high"),
                        default="none")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
