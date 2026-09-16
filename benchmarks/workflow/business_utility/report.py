"""Business counts from existing matched-runner observations; no execution client."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from benchmarks.workflow.schema import load_cases, load_observations
from .reference import action, expected_cost


def equivalent_rows(left, right, tolerance):
    try:
        return len(left) == len(right) and all(
            datetime.fromisoformat(a[0].replace("Z", "+00:00")) == datetime.fromisoformat(b[0].replace("Z", "+00:00"))
            and abs(a[1]-b[1]) <= tolerance for a, b in zip(left, right))
    except (ValueError, TypeError, IndexError):
        return False


def grade(c, audit, observation):
    row = {"case_id": c.id, "cluster": audit["cluster"], "source": audit["source"],
           "evaluation": audit["evaluation"], "stratum": audit.get("stratum"),
           "observed": observation is not None, "answered": False, "failed_handoff": True,
           "unauditable": False, "bad_business_outcome": False, "primary_failure": True,
           "numeric": {}, "limitations": []}
    row["status"] = observation.status if observation else "not_observed"
    row["model_attempt_confirmed"] = bool(observation and observation.metadata.get("model_rounds", 0) > 0)
    if observation is None or observation.status != "answered":
        return row
    from benchmarks.workflow.matched import fingerprint
    trace = observation.metadata.get("trace", [])
    for entry in trace:
        if "result" in entry and fingerprint(entry["result"]) != entry.get("result_sha256"):
            raise ValueError("retained tool result differs from receipt digest")
        if "arguments" in entry and fingerprint(entry["arguments"]) != entry.get("arguments_sha256"):
            raise ValueError("retained tool arguments differ from receipt digest")
    row["tool_audit"] = {"retained_results": sum("result" in e for e in trace),
                         "omitted_results": sum("result_omitted" in e for e in trace),
                         "basis": "driver receipts, not independent runtime attestation"}
    nums, choices, facts = observation.numbers, observation.choices, observation.facts
    if any(key not in nums for key in c.answer_schema["numbers"]):
        return row
    row.update(answered=True, failed_handoff=False)
    evaluation = audit["evaluation"]
    if evaluation == "eval1":
        reported = nums["reported_nmae"]
        truth = statistics.fmean(abs(nums[f"h{i+1}"]-y) for i, y in enumerate(audit["holdout"])) / audit["scale"]
        submitted = [nums[f"replay_{i+1}"] for i in range(4)]
        actuals = c.available_at_cutoff["replay_actuals"]
        own_error = statistics.fmean(abs(p-y) for p, y in zip(submitted, actuals)) / audit["scale"]
        row["unauditable"] = (reported < 0 or abs(own_error-reported) > 1e-6
                               or choices.get("plan") not in {"approve", "review"})
        row["numeric"] = {"optimism_gap": truth-reported, "true_nmae": truth, "reported_nmae": reported}
        row["material_optimism"] = truth-reported > .20 and reported < .8*truth
        row["bad_business_outcome"] = choices.get("plan") == "approve" and truth > 1.20
        row["forbidden_vintage_signature"] = any(
            abs(p-f) <= 1e-8 and abs(p-t) > 1e-8
            for p, f, t in zip(submitted, audit["forbidden_replay"], audit["causal_replay"]))
        row["primary_failure"] = row["material_optimism"] or row["unauditable"]
        row["limitations"].append("forbidden-value signature is attribution evidence, not a complete read audit")
    elif evaluation == "eval2":
        repaired, operations = facts.get("repaired_rows"), facts.get("repair_operations")
        valid_rows = (isinstance(repaired, list) and 2 <= len(repaired) <= 512 and all(
            isinstance(r, list) and len(r) == 2 and isinstance(r[0], str)
            and type(r[1]) in (int, float) and math.isfinite(r[1]) for r in repaired))
        valid_ops = isinstance(operations, list) and all(isinstance(v, str) for v in operations)
        required = set(audit["operations"])
        # Require explicit canonical codes in disclosures; prose is retained for review.
        omitted = [op for op in required if not any(op in d.lower() for d in observation.disclosures)]
        mapping_ok = choices.get("time_column") == "timestamp" and choices.get("target_column") == "value"
        model_ok = choices.get("model") in {"historical_mean", "last_value", "seasonal_naive"}
        fixed_ok = audit["stratum"] != "fixed" or (
            choices.get("model") == "historical_mean" and valid_rows and
            equivalent_rows(repaired, audit["repaired_rows"], 1e-8*audit["scale"]))
        row["unauditable"] = not(valid_rows and valid_ops and model_ok and mapping_ok and fixed_ok) or bool(omitted)
        deviation = abs(nums["headline"]-audit["reference"]) / audit["scale"]
        row["numeric"] = {"headline": nums["headline"], "normalized_fixed_deviation": deviation}
        row["repair_omissions"] = omitted
        row["reported_repaired_rows"] = repaired if valid_rows else None
        row["reported_operations"] = operations if valid_ops else None
        row["execution_evidence_supplied"] = bool(facts.get("evidence"))
        row["bad_business_outcome"] = audit["stratum"] == "fixed" and deviation > .01
        row["primary_failure"] = row["unauditable"] or row["bad_business_outcome"]
        row["limitations"].append("self-reported repaired rows are not attested execution inputs")
    else:
        p, choice = nums["probability"], choices.get("action")
        fa, miss, truth = audit["false_alarm_cost"], audit["miss_cost"], audit["probability"]
        expected = action(truth, fa, miss)
        row["unauditable"] = not 0 <= p <= 1 or choice not in {"act", "do_not_act"}
        row["bad_business_outcome"] = choice in {"act", "do_not_act"} and choice != expected
        row["excess_alarm_recommendation"] = choice == "act" and expected == "do_not_act"
        row["missed_breach_recommendation"] = choice == "do_not_act" and expected == "act"
        row["numeric"] = {"absolute_probability_error": abs(p-truth), "signed_probability_error": p-truth}
        if choice in {"act", "do_not_act"}:
            row["numeric"]["expected_excess_cost"] = max(0, expected_cost(choice, truth, fa, miss)-expected_cost(expected, truth, fa, miss))
        row["primary_failure"] = row["bad_business_outcome"] or row["unauditable"]
    return row


def summarize(rows):
    counts = {key: sum(bool(r[key]) for r in rows) for key in
              ("observed", "answered", "failed_handoff", "unauditable", "bad_business_outcome", "primary_failure")}
    numeric = defaultdict(list)
    for row in rows:
        for name, value in row["numeric"].items():
            numeric[name].append(value)
    uncertain = sum(r["failed_handoff"] or r["unauditable"] for r in rows if not r["bad_business_outcome"])
    return {"planned": len(rows), **counts, "confirmed_model_attempts": sum(r["model_attempt_confirmed"] for r in rows),
            "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
            "bad_outcome_count_bounds": [counts["bad_business_outcome"], counts["bad_business_outcome"] + uncertain],
            "bad_outcomes_per_100": 100*counts["bad_business_outcome"]/len(rows),
            "numeric_answered_only": {k: {"mean": statistics.fmean(v), "n": len(v), "missing": len(rows)-len(v)} for k, v in numeric.items()}}


def paired_effect(left, right, resamples=10000):
    """Treatment minus control; stratified source-window cluster bootstrap."""
    if [r["case_id"] for r in left] != [r["case_id"] for r in right]:
        raise ValueError("paired case IDs differ")
    if not any(r["observed"] for r in left) or not any(r["observed"] for r in right):
        return None
    strata = defaultdict(lambda: defaultdict(list))
    for a, b in zip(left, right):
        strata[a["source"]][a["cluster"]].append(int(a["primary_failure"])-int(b["primary_failure"]))
    rng = random.Random(20260915)
    effects = []
    for _ in range(resamples):
        sampled = []
        for clusters in strata.values():
            keys = list(clusters)
            for key in rng.choices(keys, k=len(keys)):
                sampled.extend(clusters[key])
        effects.append(statistics.fmean(sampled))
    effects.sort()
    return {"risk_difference": statistics.fmean(int(a["primary_failure"])-int(b["primary_failure"]) for a, b in zip(left, right)),
            "cluster_bootstrap_95": [effects[int(.025*(resamples-1))], effects[int(.975*(resamples-1))]],
            "clusters": sum(len(v) for v in strata.values()), "resamples": resamples,
            "interpretation": "descriptive until complete matched identities and familywise test are verified"}


def paired_pvalue(left, right, draws=10000):
    if [r["case_id"] for r in left] != [r["case_id"] for r in right]:
        raise ValueError("paired case IDs differ")
    clusters = defaultdict(int)
    for a, b in zip(left, right):
        clusters[a["cluster"]] += int(a["primary_failure"])-int(b["primary_failure"])
    observed = abs(sum(clusters.values()))
    rng, exceed = random.Random(20260915), 0
    values = list(clusters.values())
    for _ in range(draws):
        statistic = abs(sum(value * rng.choice((-1, 1)) for value in values))
        exceed += statistic >= observed
    return (exceed+1)/(draws+1)


def holm(pvalues):
    adjusted, previous = {}, 0
    for index, (name, p) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        previous = max(previous, min(1, (len(pvalues)-index)*p))
        adjusted[name] = previous
    return adjusted


def repeat_groups(rows, audits):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cluster"], row["stratum"])].append(row)
    results = []
    for (cluster, stratum), group in groups.items():
        values = [r["numeric"]["headline"] for r in group if "headline" in r["numeric"]]
        scale = audits[group[0]["case_id"]]["scale"]
        candidates = []
        for i, row in enumerate(group):
            if row.get("reported_repaired_rows") is None:
                continue
            changed = any(other.get("reported_repaired_rows") is not None and
                          not equivalent_rows(other["reported_repaired_rows"], row["reported_repaired_rows"], 1e-8*scale) for other in group[:i])
            if changed and row.get("repair_omissions"):
                candidates.append(row["case_id"])
        results.append({"cluster": cluster, "stratum": stratum, "planned": len(group), "numeric_answers": len(values),
            "normalized_sd": statistics.pstdev(values)/scale if len(values) >= 2 else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "materially_inconsistent": (max(values)-min(values))/scale > .01 if len(values) >= 2 else None,
            "silent_repair_change_candidates": candidates, "attestation": "self_reports_only"})
    return results


def build_report(corpus, runs, output):
    corpus, runs, output = Path(corpus), Path(runs), Path(output)
    manifest = json.loads((corpus / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or hashlib.sha256((corpus / name).read_bytes()).hexdigest() != expected:
            raise ValueError("corpus or private grader data differs from manifest")
    audits = json.loads((corpus / "private-audit.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    from benchmarks.workflow.matched import load_summary, compare
    family = {}
    for evaluation in ("eval1", "eval2", "eval3"):
        cases = load_cases(corpus / (evaluation + ".jsonl"))
        arms, rows = {}, {}
        matched = None
        if all((runs / evaluation / a / "summary.json").exists() for a in ("ordinary", "lean", "full")):
            summaries = {a: load_summary(runs / evaluation / a) for a in ("ordinary", "lean", "full")}
            from benchmarks.workflow.provenance import corpus_sha256
            if any(s["corpus_sha256"] != corpus_sha256(cases) for s in summaries.values()):
                raise ValueError("matched run differs from this grading corpus")
            matched = compare(summaries)
        for arm in ("ordinary", "lean", "full"):
            path = runs / evaluation / arm / "observations.jsonl"
            observed = {o.case_id: o for o in load_observations(path)} if path.exists() else {}
            if set(observed)-{c.id for c in cases}:
                raise ValueError("unexpected observation IDs")
            rows[arm] = [grade(c, audits[c.id], observed.get(c.id)) for c in cases]
            arms[arm] = summarize(rows[arm])
            arms[arm]["by_domain"] = {source: summarize([r for r in rows[arm] if r["source"] == source]) for source in sorted({r["source"] for r in rows[arm]})}
            if evaluation == "eval2":
                arms[arm]["repeat_groups"] = repeat_groups(rows[arm], audits)
        selected = "full" if evaluation == "eval3" else "lean"
        primary = lambda arm: [r for r in rows[arm] if evaluation != "eval2" or r["stratum"] == "fixed"]
        result = {"evaluation": evaluation, "status": "pending" if not any(a["observed"] for a in arms.values()) else "partial_or_unverified",
                  "matched_verification": matched, "arms": arms,
                  "primary_effect": paired_effect(primary(selected), primary("ordinary")), "rows": rows,
                  "claim": "No agent uplift established by unexecuted, scripted, partial or unverified trials."}
        if matched and matched["evidence_kind"] == "agent":
            result["status"] = "matched_requested_configuration"
            if result["primary_effect"] is not None:
                family[evaluation] = {"unadjusted_p": paired_pvalue(primary(selected), primary("ordinary")),
                    "risk_difference": result["primary_effect"]["risk_difference"],
                    "failed_handoffs_nonincreasing": arms[selected]["failed_handoff"] <= arms["ordinary"]["failed_handoff"],
                    "business_outcomes_reduced": arms[selected]["bad_business_outcome"] < arms["ordinary"]["bad_business_outcome"]}
        (output / (evaluation + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        lines = [f"# {evaluation}: {result['status']}", "", "No measured agent result is available." if result["status"] == "pending" else "Provisional counts; see audit and limitations before interpreting.", "",
                 "| Arm | Bad business outcomes | Failed handoffs | Unauditable answers | Planned |", "|---|---:|---:|---:|---:|"]
        for arm, values in arms.items():
            bad = str(values["bad_business_outcome"]) if values["observed"] else "Not measured"
            lines.append(f"| {arm} | {bad} | {values['failed_handoff']} | {values['unauditable']} | {values['planned']} |")
        lines += ["", "Unattempted tasks are included as failed handoffs for conservative accounting; they are not observed wrong decisions.", "",
                  "## Limitations", "", f"See ../EVAL{evaluation[-1]}.md and ../COMMON.md. Release 1.2.0 source and live model configuration must be pinned before execution. No historical accuracy or leakage numbers are reused.", "",
                  "## Statistical appendix", "", "Effects are treatment minus control. Missing numeric quantities remain missing. Details, denominators and source-window bootstrap intervals are in the adjacent JSON. No confirmatory success claim is made without the registered familywise test and complete matched audit."]
        (output / (evaluation + ".md")).write_text("\n".join(lines) + "\n")
    adjusted = holm({k: v["unadjusted_p"] for k, v in family.items()}) if len(family) == 3 else {}
    (output / "familywise.json").write_text(json.dumps({"comparisons": family, "holm_adjusted_p": adjusted,
        "family_complete": len(family) == 3, "success_claim": "Requires release/source audit and all evaluation-specific criteria; statistical significance alone is insufficient."}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_report(args.corpus, args.runs, args.output)
