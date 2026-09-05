"""Run or score Gnomon Workflow Bench.

An arm is either an existing JSONL submission (``--submission``) or an
executable (``--arm-command``) that receives one case JSON object on stdin and
returns one observation JSON object on stdout.  This tiny protocol keeps the
benchmark neutral: raw LLM, evidence injection, MCP profiles, and deterministic
Gnomon can all be adapters without the scorer knowing their implementation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmarks.common.checkpoint import prepare_run_identity
from benchmarks.common.manifest import code_revision, write_manifest
from benchmarks.workflow.schema import Case, Observation, load_cases, load_observations
from benchmarks.workflow.scoring import score_run
from benchmarks.workflow.provenance import atomic_write_text, corpus_sha256, write_observations
from benchmarks.workflow.accounting import AttemptJournal, attach, receipt, receipts, reported_cost_limit, FIELDS
from benchmarks.workflow.matched import prepare as prepare_experiment, public_context, normalized_rows, artifact_hashes
from benchmarks.workflow.process import ProcessLimit, run_process

DEFAULT_CASES = Path(__file__).with_name("cases") / "smoke.jsonl"


def _run_identity(args: argparse.Namespace, cases: list[Case]) -> dict[str, Any]:
    submission_sha = None
    if args.submission:
        submission_sha = hashlib.sha256(
            Path(args.submission).read_bytes()).hexdigest()
    identity = {
        "schema_version": 1,
        "benchmark": "gnomon-workflow",
        "code_revision": code_revision(),
        "corpus_sha256": corpus_sha256(cases),
        "case_ids": [case.id for case in cases],
        "arm": args.arm,
        "arm_command": args.arm_command,
        "submission_sha256": submission_sha,
        "timeout_seconds": args.timeout,
        "jobs": args.jobs,
        "infrastructure_retries": args.infrastructure_retries,
    }
    if getattr(args, "experiment", None):
        identity["matched_experiment"] = prepare_experiment(
            args.experiment, cases, command=args.arm_command, arm=args.arm,
            timeout=args.timeout, jobs=args.jobs, retries=args.infrastructure_retries)
    return identity


def _prepare_run_identity(output_dir: Path, identity: dict[str, Any],
                          *, resume: bool) -> None:
    """Prevent a checkpoint from being reused by a different arm or corpus."""
    prepare_run_identity(
        output_dir, identity, resume=resume,
        state_paths=[output_dir / "observations.jsonl",
                     output_dir / "observations.attempts.sqlite3",
                     output_dir / "summary.json"],
    )


def case_payload(case: Case) -> dict[str, Any]:
    """Public arm input. The oracle is deliberately excluded."""
    result = {
        "schema_version": case.schema_version, "id": case.id,
        "kind": case.kind, "domain": case.domain, "question": case.question,
        "available_at_cutoff": case.available_at_cutoff,
        "answer_schema": {
            key: (dict(value) if key == "choice_sources" else list(value))
            for key, value in case.answer_schema.items()},
        "tags": list(case.tags),
    }
    if case.episode:
        result["answer_schema"] = case.episode[0]["answer_schema"]
        result["episode_phase"] = case.episode[0]["name"]
    return result


def _invoke(payload: dict[str, Any], case_id: str, argv: list[str],
            timeout: float, retries: int = 0, stage: str = "initial",
            journal: AttemptJournal | None = None) -> Observation:
    if journal is None:
        owned = AttemptJournal()
        try:
            return _invoke(payload, case_id, argv, timeout, retries, stage, owned)
        finally:
            owned.close()
    last: Observation | None = None
    if "_spending_allowance_usd" in payload and retries:
        raise ValueError("reported-cost controlled attempts cannot be retried")
    if "_private_episode" in payload and retries:
        raise ValueError("episodes cannot be retried after a possible reveal")
    if "_private_episode" in payload and journal.path is None:
        raise ValueError("episodes require a persistent attempt journal")
    items = []
    for attempt in range(retries + 1):
        identifier = journal.start(case_id, stage)
        started = time.monotonic()
        delivered = payload
        if "_private_episode" in payload:
            delivered = {**payload, "_episode_journal": {"path": str(journal.path.resolve()), "attempt_id": identifier}}
        last = _invoke_once(delivered, case_id, argv, timeout, stage)
        if "_private_episode" in payload:
            checkpoints = journal.checkpoints(identifier)
            partial = checkpoints[-1]["usage"] if checkpoints else {}
            # A killed driver may have incurred more work after its last commit.
            # Retain positive lower bounds without pretending they are full totals.
            values = {key: max(getattr(last, key), partial.get(key) or 0) for key in (
                "tool_calls", "cumulative_tokens", "response_tokens", "latency_seconds")}
            values["cost_usd"] = max(last.cost_usd or 0, partial.get("cost_usd") or 0) or last.cost_usd
            last = replace(last, **values, metadata={**last.metadata, "episode_checkpoints": checkpoints})
        item = {**receipt(last, stage), "attempt_id": identifier,
                "harness_wall_seconds": time.monotonic() - started}
        journal.finish(identifier, item)
        items.append(item)
        if last.status != "error" or last.metadata.get("error") not in {
            "timeout", "subprocess_failure", "empty_stdout", "provider_timeout",
            "provider_error", "model_error", "model_submission_error",
        }:
            break
    assert last is not None
    return attach(replace(last, metadata={**last.metadata,
                   "attempts": len(items), "retries_used": len(items) - 1}), items)


def _invoke_once(payload: dict[str, Any], case_id: str, argv: list[str],
                 timeout: float, stage: str) -> Observation:
    try:
        result = run_process(argv, input=(json.dumps(payload, allow_nan=False) + "\n").encode(), timeout=timeout)
    except ProcessLimit as error:
        return Observation(case_id=case_id, status="error", support="abstained",
                           metadata={"error": str(error), "timeout_seconds": timeout,
                                     "budget_exceeded": True,
                                     "failed_stage": stage})
    except OSError as error:
        return Observation(case_id=case_id, status="error", support="abstained",
                           metadata={"error": type(error).__name__, "detail": str(error),
                                     "failed_stage": stage})
    if result.returncode != 0:
        return Observation(
            case_id=case_id, status="error", support="abstained",
            metadata={"error": "subprocess_failure",
                      "returncode": result.returncode,
                      "failed_stage": stage,
                      "stderr": result.stderr[-1000:].decode("utf-8", errors="replace")},
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return Observation(case_id=case_id, status="error", support="abstained",
                           metadata={"error": "empty_stdout", "failed_stage": stage})
    try:
        from gnomon.agent_eval import _decode_record
        payload = _decode_record(lines[-1].decode("utf-8"))
        payload.setdefault("case_id", case_id)
        if payload["case_id"] != case_id:
            raise ValueError(f"returned case_id {payload['case_id']!r}")
        return Observation.from_dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return Observation(case_id=case_id, status="error", support="abstained",
                           metadata={"error": "invalid_observation", "detail": str(error),
                                     "failed_stage": stage})


def _run_one(case: Case, argv: list[str], timeout: float, retries: int = 0,
             journal: AttemptJournal | None = None, experiment: dict | None = None) -> Observation:
    if journal is None:
        owned = AttemptJournal()
        try:
            return _run_one(case, argv, timeout, retries, owned, experiment)
        finally:
            owned.close()
    payload = case_payload(case)
    if experiment is not None:
        payload["experiment"] = public_context(experiment)
        limit = reported_cost_limit(experiment["pinned"]["common"].get("budget", {}))
        if limit is not None:
            spend = journal.reported_spend()
            if not spend["complete"] or spend["total"] >= limit:
                stopped = Observation(case_id=case.id, status="error", support="abstained", cost_usd=0,
                    metadata={"resource_fields": list(FIELDS), "error": "spending_usage_unmeasured" if not spend["complete"] else "cap:cost",
                              "budget_exceeded": None if not spend["complete"] else True,
                              "not_dispatched": True, "arm_reported_spend": spend,
                              "max_reported_cost_usd": limit})
                item = {**receipt(stopped, "spending_stop"), "provenance": "operator_preflight_no_external_dispatch"}
                journal.import_receipt(item)
                return attach(stopped, journal.for_case(case.id))
            payload["_spending_allowance_usd"] = limit - spend["total"]
    if case.episode:
        if experiment is None:
            raise ValueError("episodes require the matched shared driver")
        payload["_private_episode"] = [{key: value for key, value in phase.items() if key != "oracle"}
                                        for phase in case.episode]
    initial = _invoke(payload, case.id, argv, timeout, retries, "initial", journal)
    if case.episode:
        return attach(initial, journal.for_case(case.id))
    if initial.status == "error":
        return attach(initial, journal.for_case(case.id))
    stage_results = dict(initial.stage_results)
    required_calls = int(initial.metadata.get("surface_required_calls", 0))
    total_retries = int(initial.metadata.get("retries_used", 0))
    stage_infrastructure_failures: list[dict[str, Any]] = []
    total_calls, total_tokens = initial.tool_calls, initial.cumulative_tokens
    total_response, total_latency = initial.response_tokens, initial.latency_seconds
    for stage in case.stages:
        name = stage["name"]
        if name == "repair" and not case.oracle.requires_repair:
            continue
        if name == "outcome" and not case.oracle.requires_tracking:
            continue
        public = case_payload(case)
        public["workflow_stage"] = name
        public["revealed"] = stage.get("revealed") or {}
        public["answer_schema"] = stage.get("answer_schema") or {
            "numbers": [], "choices": [], "facts": []}
        public["prior_observation"] = {
            "status": initial.status, "support": initial.support,
            "numbers": initial.numbers, "choices": initial.choices,
            "published_fingerprint": initial.published_fingerprint,
            "artifact_id": initial.metadata.get("artifact_id"),
        }
        public["question"] = (
            "The target ambiguity is now resolved. Complete the original request."
            if name == "repair" else
            "The outcome is now revealed. Compare it with the saved prediction and record tracking."
        )
        if name == "outcome":
            # Outcome tracking is deterministic bookkeeping over the immutable
            # initial answer, not another forecasting task. Compiling it here
            # prevents a fresh agent process from losing the prior artifact or
            # inventing different arithmetic. Absence of a published identity
            # remains a real tracking failure (notably for the raw control).
            actual = (stage.get("revealed") or {}).get("actual")
            predicted = initial.numbers.get("next")
            absolute_error = (abs(float(actual) - predicted)
                              if actual is not None and predicted is not None else None)
            followup = Observation(
                case_id=case.id,
                status="answered" if absolute_error is not None else "error",
                support="supported" if absolute_error is not None else "abstained",
                numbers=({"absolute_error": absolute_error}
                         if absolute_error is not None else {}),
                facts=({"tracked_forecast_id": initial.published_fingerprint}
                       if initial.published_fingerprint is not None else {}),
                temporal_leakage=False,
                metadata={"compiled_stage": "outcome"},
            )
        else:
            followup = _invoke(public, case.id, argv, timeout, retries, name, journal)
        total_calls += followup.tool_calls
        total_tokens += followup.cumulative_tokens
        total_response += followup.response_tokens
        total_latency += followup.latency_seconds
        required_calls += int(followup.metadata.get("surface_required_calls", 0))
        total_retries += int(followup.metadata.get("retries_used", 0))
        if followup.status == "error" and followup.metadata.get("error") in {
            "timeout", "subprocess_failure", "empty_stdout", "provider_timeout",
            "provider_error", "model_error", "model_submission_error",
        }:
            failure = {
                "stage": name, "error": followup.metadata.get("error")}
            if followup.metadata.get("returncode") is not None:
                failure["returncode"] = followup.metadata["returncode"]
            stderr = str(followup.metadata.get("stderr") or "").strip()
            if stderr:
                # Keep the actionable final traceback/provider line without
                # copying a whole subprocess transcript into every score row.
                failure["diagnostic"] = stderr.splitlines()[-1][:300]
            stage_infrastructure_failures.append(failure)
        if name == "repair":
            stage_results[name] = {
                "completed": initial.status == "abstained" and followup.status == "answered",
                "followup_status": followup.status,
                "numbers": followup.numbers, "choices": followup.choices,
                "facts": followup.facts,
                "support": followup.support,
                "published_fingerprint": followup.published_fingerprint,
            }
        else:
            actual = (stage.get("revealed") or {}).get("actual")
            predicted = initial.numbers.get("next")
            bound_id = initial.published_fingerprint
            stage_results[name] = {
                "tracked": (followup.status == "answered" and predicted is not None
                            and bound_id is not None
                            and followup.facts.get("tracked_forecast_id")
                            == bound_id),
                "actual": actual, "prediction": predicted,
                "absolute_error": (abs(float(actual) - predicted)
                                   if actual is not None and predicted is not None else None),
                "reported_absolute_error": followup.numbers.get("absolute_error"),
            }
        stage_results[name]["economics"] = {
            "tool_calls": followup.tool_calls,
            "cumulative_tokens": followup.cumulative_tokens,
            "response_tokens": followup.response_tokens,
            "latency_seconds": followup.latency_seconds,
            "resource_accounting": followup.metadata.get("resource_accounting"),
        }
    stage_results["initial"] = {"economics": {
        "tool_calls": initial.tool_calls, "cumulative_tokens": initial.cumulative_tokens,
        "response_tokens": initial.response_tokens,
        "latency_seconds": initial.latency_seconds,
        "resource_accounting": initial.metadata.get("resource_accounting"),
    }}
    return attach(replace(initial, stage_results=stage_results, tool_calls=total_calls,
                   cumulative_tokens=total_tokens, response_tokens=total_response,
                   latency_seconds=total_latency,
                   metadata={**initial.metadata,
                             "surface_required_calls": required_calls,
                             "retries_used": total_retries,
                             "stage_infrastructure_failures":
                                 stage_infrastructure_failures}), journal.for_case(case.id))


def run_command(cases: list[Case], command: str, timeout: float,
                jobs: int = 1, retries: int = 0,
                prior: list[Observation] | None = None,
                checkpoint_path: Path | None = None, experiment: dict | None = None) -> list[Observation]:
    journal = AttemptJournal(checkpoint_path.with_suffix(".attempts.sqlite3") if checkpoint_path is not None else None)
    try:
        for observation in prior or []:
            items = observation.metadata.get("attempt_receipts")
            if items:
                for item in receipts(observation):
                    journal.import_receipt(item)
            elif not journal.for_case(observation.case_id):
                journal.import_receipt(receipt(observation, "legacy_resume", historical=True))
        return _run_command(cases, command, timeout, jobs, retries, prior, checkpoint_path, journal, experiment)
    finally:
        journal.close()


def _run_command(cases, command, timeout, jobs, retries, prior, checkpoint_path, journal, experiment=None):
    argv = shlex.split(command)
    if not argv:
        raise ValueError("arm command is empty")
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if retries < 0:
        raise ValueError("retries must be non-negative")
    limit = reported_cost_limit(experiment["pinned"]["common"].get("budget", {})) if experiment else None
    if limit is not None and (jobs != 1 or retries or journal.path is None):
        raise ValueError("reported-cost control requires one worker, zero retries and a persistent attempt journal")
    retained = {
        row.case_id: row for row in (prior or [])
        if row.status != "error"
        and not row.metadata.get("stage_infrastructure_failures")
    }
    previous = {row.case_id: row for row in prior or []}
    for case in cases:
        if not case.episode or case.id in retained:
            continue
        attempts = journal.for_case(case.id)
        if attempts:
            # A previous attempt may already have seen later actuals. Never turn
            # checkpoint resume into another forecast attempt after that reveal.
            row = previous.get(case.id) or Observation(case_id=case.id, status="error", support="abstained",
                                                       metadata={"error": "episode_interrupted"})
            row = replace(row, metadata={**row.metadata, "episode_resume": "not_replayed_after_prior_start",
                                        "episode_checkpoints": journal.checkpoints(attempts[-1]["attempt_id"])})
            retained[case.id] = attach(row, attempts)
    pending = [case for case in cases if case.id not in retained]
    fresh: dict[str, Observation] = {}

    def checkpoint() -> None:
        if checkpoint_path is None:
            return
        available = {**retained, **fresh}
        write_observations(checkpoint_path, [available[case.id] for case in cases
                                             if case.id in available])

    if jobs == 1:
        for case in pending:
            fresh[case.id] = _run_one(case, argv, timeout, retries, journal, experiment)
            checkpoint()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_run_one, case, argv, timeout, retries, journal, experiment): case.id
                       for case in pending}
            for future in concurrent.futures.as_completed(futures):
                fresh[futures[future]] = future.result()
                checkpoint()
    merged = {**retained, **fresh}
    return [attach(merged[case.id], journal.for_case(case.id)) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--submission", help="normalized observation JSONL to score")
    source.add_argument("--arm-command", help="executable implementing the stdin/stdout arm protocol")
    parser.add_argument("--arm", required=True, help="stable arm name recorded in results")
    parser.add_argument("--experiment", help="common matched ordinary/lean/full experiment JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per arm case")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel arm processes (default: 1)")
    parser.add_argument("--infrastructure-retries", type=int, default=2,
                        help="bounded retries for timeout/provider/subprocess failures")
    parser.add_argument("--resume", action="store_true",
                        help="reuse successful observations and rerun only failed case IDs")
    args = parser.parse_args()
    if args.experiment and args.submission:
        parser.error("matched experiments require an executed arm command, not an unbound submission")

    cases = load_cases(args.cases)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = _run_identity(args, cases)
    _prepare_run_identity(output_dir, identity, resume=args.resume)
    prior_path = output_dir / "observations.jsonl"
    prior = (load_observations(prior_path)
             if args.resume and prior_path.is_file() and not args.submission else None)
    observations = load_observations(args.submission) if args.submission else \
        run_command(cases, args.arm_command, args.timeout, args.jobs,
                    args.infrastructure_retries, prior, prior_path, identity.get("matched_experiment"))
    result = score_run(cases, observations, arm=args.arm)
    if args.experiment:
        # Retain raw observations/journal on failure, but never publish a matched
        # summary under stale code/configuration after files changed during a run.
        checked = _run_identity(args, load_cases(args.cases))
        if checked != identity:
            raise ValueError("experiment inputs changed during execution; observations retained, use a new run")
        result["matched_experiment"] = identity["matched_experiment"]
    write_observations(output_dir / "observations.jsonl", observations)
    if args.experiment:
        result["matched_artifacts"] = artifact_hashes(output_dir)
    atomic_write_text(output_dir / "summary.json",
                      json.dumps(result, indent=2, sort_keys=True) + "\n")
    normalized = [json.dumps(row, sort_keys=True) + "\n" for row in normalized_rows(result)]
    atomic_write_text(output_dir / "gnomonbench.jsonl", "".join(normalized))
    write_manifest(output_dir, benchmark="gnomon-workflow", condition=args.arm,
                   target={"case_ids": [case.id for case in cases], "schema_version": 1,
                           "corpus_sha256": corpus_sha256(cases)},
                   arm_command=args.arm_command, jobs=args.jobs, timeout=args.timeout,
                   infrastructure_retries=args.infrastructure_retries,
                   run_identity=identity,
                   resumed_successful_cases=len([
                       row for row in (prior or [])
                       if row.status != "error"
                       and not row.metadata.get(
                           "stage_infrastructure_failures")]))
    console_summary = {key: value for key, value in result.items() if key != "rows"}
    if "matched_experiment" in console_summary:
        console_summary["matched_experiment"] = {
            key: identity["matched_experiment"][key] for key in ("experiment_id", "arm")}
    print(json.dumps(console_summary, indent=2))
    return 0 if result["release_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
