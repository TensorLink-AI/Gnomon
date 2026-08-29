"""Per-channel TemporalBench scoring — THE way to compare arms here.

The official metric aggregates all channels of a MIMIC record: its
``_align_series_dict`` returns "not aligned" the moment one ground-truth
channel is absent from a prediction, and the whole record scores nothing
(``metric_flag: missing_channel``). That rule is right for the official
leaderboard — a forecast of 5 of 6 vitals is not a forecast of the record
— and each arm's ``summary.json`` keeps reporting that official
all-channels number: it is the headline the leaderboard uses and it
never disappears.

But it is the wrong rule for *comparing* an arm that abstains against
one that never does. In the DeepSeek V4 Flash run Gnomon forecast 218 of
288 channels, but abstained on ``temperature_c`` in 44 of 48 records
(MIMIC charts temperature every few hours, so its history is far shorter
than heart rate's). One sparse channel voided 38 otherwise-complete
records, and the official metric could compare exactly one record across
arms. So cross-arm comparison goes through this script (see the README's
"Comparing arms" section): the same official metrics, computed on a
declared subset — for each record, the intersection of channels both
arms forecast. The metric code is the dataset's own
``_compute_ow_metrics`` — nothing is reimplemented. Coverage (how many
records and channels each number rests on) is reported next to every
figure, because a number over a subset means nothing without it; quote
the two together or not at all.

Output is a paired per-channel table plus the paired record-level OW
metrics over the intersection. Channels either arm skipped are counted
and named, never dropped silently. Where an arm's details records carry
per-channel support labels (Gnomon arms write ``channel_support``; a
run with ``--best-effort`` labels its disclosed fallback rows
``best_effort``), the support mix over the compared channels is printed
beside the scores — best_effort rows carry a NO RELIABLE FORECAST
disclosure and are not supported forecasts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from statistics import median

from benchmarks.common.manifest import incompatibilities, read_manifest
from benchmarks.report import sign_test
from .gnomon_runner import forecast_target_map
from .tasks import load_official_metrics, prompt_input_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score TemporalBench arms per channel on the "
                    "channels both arms forecast.",
    )
    parser.add_argument("--data-dir", required=True,
                        help="TemporalBench data dir (holds the labeled "
                             "split and forecast_metrics_utils.py)")
    parser.add_argument("--baseline", required=True,
                        help="Baseline arm output dir (with details/)")
    parser.add_argument("--treatment", required=True,
                        help="Treatment arm output dir (with details/)")
    parser.add_argument("--json", action="store_true",
                        help="Emit the full result as JSON")
    parser.add_argument("--output", help="Write the full JSON result to this path")
    return parser.parse_args()


def load_forecasts(arm_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Map task id -> the arm's per-channel forecast dict, plus task id ->
    per-channel support labels where the arm recorded them (Gnomon arms
    write ``channel_support``; LLM control answers carry no labels)."""
    out: dict[str, dict] = {}
    support: dict[str, dict] = {}
    details = arm_dir / "details"
    for path in sorted(details.glob("*.json")):
        record = json.loads(path.read_text())
        forecast = (record.get("answer") or {}).get("forecast") or {}
        out[path.stem] = {
            name: values for name, values in forecast.items()
            if isinstance(values, (list, tuple)) and values
        }
        labels = record.get("channel_support")
        if isinstance(labels, dict):
            support[path.stem] = {str(k): str(v) for k, v in labels.items()}
    return out, support


def load_truth(data_dir: Path) -> dict[str, dict]:
    """Map task id -> {ground_truth, history} for forecast tiers."""
    path = data_dir / "task_merged_dev_with_labels_tiers.jsonl"
    out: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            truth = row.get("ground_truth")
            if not isinstance(truth, dict) or not truth:
                continue
            arrays = prompt_input_arrays(row)
            target_map = forecast_target_map(row, arrays)
            # Truth may call one output `future_main` while its historical
            # input is named `pressure_downwind`. The benchmark forecaster
            # already resolves this explicit alias; scoring must use the same
            # mapping or silently lose the history-dependent denominator.
            history = {
                output_key: arrays[input_key]
                for output_key, input_key in target_map.items()
                if input_key in arrays
            }
            out[row["id"]] = {"truth": truth, "history": history}
    return out


def subset_metrics(metrics_module, truth, history, forecast, channels):
    """Official OW metrics restricted to `channels`."""
    gt = OrderedDict((name, truth[name]) for name in channels)
    pred = OrderedDict((name, forecast[name]) for name in channels)
    hist = None
    if isinstance(history, dict):
        hist = OrderedDict(
            (name, history[name]) for name in channels if name in history
        )
    gt_parsed = metrics_module._series_dict_from_obj(gt)
    pred_parsed = metrics_module._series_dict_from_obj(pred)
    if gt_parsed is None or pred_parsed is None:
        return None, OrderedDict()
    aligned_ok, aligned = metrics_module._align_series_dict(
        gt_parsed, pred_parsed
    )
    if not aligned_ok:
        return None, OrderedDict()
    hist_parsed = (
        metrics_module._series_dict_from_obj(hist) if hist else None
    )
    return metrics_module._compute_ow_metrics(
        gt_parsed, aligned, hist_parsed,
        metrics_module.DEFAULT_OW_WEIGHT_MODE, 1,
        metrics_module.DEFAULT_OW_SMAPE_EPS,
    )


def stable_scaled_error_denominator(history: object) -> bool:
    """Whether a naive absolute-change scale is numerically informative.

    MASE can explode for an almost constant series even when absolute errors
    are tiny.  We retain the official score, but identify those denominators
    and report a stable-history subset beside it.
    """
    if not isinstance(history, (list, tuple)) or len(history) < 2:
        return False
    values = []
    for item in history:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return False
    level = max(1.0, median(abs(item) for item in values))
    scale = median(abs(values[index] - values[index - 1])
                   for index in range(1, len(values)))
    return scale > 1e-6 * level


def summarise_pairs(pairs: list[tuple[float, float]]) -> dict[str, object]:
    """Summarise paired errors, including an exact directional test."""
    baseline = [pair[0] for pair in pairs]
    treatment = [pair[1] for pair in pairs]
    paired_baseline = {str(index): pair[0] for index, pair in enumerate(pairs)}
    paired_treatment = {str(index): pair[1] for index, pair in enumerate(pairs)}
    test = sign_test(paired_baseline, paired_treatment, lower_is_better=True)
    return {
        "n": len(pairs),
        "baseline_median": round(median(baseline), 4),
        "treatment_median": round(median(treatment), 4),
        "treatment_wins": test["treatment_wins"],
        "treatment_losses": test["treatment_losses"],
        "ties": test["ties"],
        "paired_sign_p_value": test["p_value"],
        "near_constant_denominators_excluded": 0,
    }


def coverage_bucket(baseline: bool, treatment: bool) -> str:
    """Name the mutually exclusive two-arm coverage state."""
    return ("both" if baseline and treatment else
            "base_only" if baseline else
            "treat_only" if treatment else "neither")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    metrics_module = load_official_metrics(data_dir)

    truth_by_id = load_truth(data_dir)
    baseline_dir = Path(args.baseline).expanduser()
    treatment_dir = Path(args.treatment).expanduser()
    problems = incompatibilities(read_manifest(baseline_dir),
                                 read_manifest(treatment_dir))
    if problems:
        raise ValueError("incompatible runs: " + "; ".join(problems))
    base, base_support = load_forecasts(baseline_dir)
    treat, treat_support = load_forecasts(treatment_dir)

    shared_ids = sorted(set(base) & set(treat) & set(truth_by_id))
    per_channel: dict[str, list[tuple[float, float]]] = {}
    per_channel_smape: dict[str, list[tuple[float, float]]] = {}
    per_channel_stable: dict[str, list[tuple[float, float]]] = {}
    versus_naive: dict[str, list[tuple[float, float]]] = {}
    versus_naive_smape: dict[str, list[tuple[float, float]]] = {}
    abstention_priced: list[tuple[float, float]] = []
    abstention_priced_smape: list[tuple[float, float]] = []
    record_rows = []
    paired_channel_records = []
    coverage = {"base_only": 0, "treat_only": 0, "both": 0, "neither": 0}
    scoring_coverage = {
        "base_only": 0, "treat_only": 0, "both": 0, "neither": 0,
    }
    # Support-label mix over the channels actually scored, per arm:
    # best_effort rows are disclosed fallbacks and must never blend
    # invisibly into a compared number.
    base_mix: dict[str, int] = {}
    treat_mix: dict[str, int] = {}

    for task_id in shared_ids:
        entry = truth_by_id[task_id]
        truth, history = entry["truth"], entry["history"]
        b_fc, t_fc = base[task_id], treat[task_id]

        both = [c for c in truth if c in b_fc and c in t_fc]
        for channel in truth:
            in_b, in_t = channel in b_fc, channel in t_fc
            key = coverage_bucket(in_b, in_t)
            coverage[key] += 1
            channel_history = (history or {}).get(channel) \
                if isinstance(history, dict) else None
            if not isinstance(channel_history, (list, tuple)) or not channel_history:
                continue
            finite_history = []
            for value in channel_history:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    finite_history.append(numeric)
            if not finite_history:
                continue
            naive = {channel: [finite_history[-1]] * len(truth[channel])}
            naive_summary, naive_detail = subset_metrics(
                metrics_module, truth, history, naive, [channel])
            if not naive_summary:
                continue
            naive_mase = (naive_detail.get(channel) or {}).get("MASE")
            naive_smape = (naive_detail.get(channel) or {}).get("sMAPE")
            if naive_mase is None:
                continue
            base_scorable = False
            base_mase = None
            if in_b:
                base_summary, base_detail = subset_metrics(
                    metrics_module, truth, history,
                    {channel: b_fc[channel]}, [channel])
                base_mase = ((base_detail or {}).get(channel) or {}).get("MASE")
                base_smape = ((base_detail or {}).get(channel) or {}).get("sMAPE")
                base_scorable = bool(base_summary and base_mase is not None)
            else:
                base_smape = None
            treatment_scorable = False
            treatment_mase = None
            if in_t:
                treatment_summary, treatment_detail = subset_metrics(
                    metrics_module, truth, history,
                    {channel: t_fc[channel]}, [channel])
                treatment_mase = ((treatment_detail or {}).get(channel) or {}).get("MASE")
                treatment_smape = ((treatment_detail or {}).get(channel) or {}).get(
                    "sMAPE")
                if treatment_summary and treatment_mase is not None:
                    treatment_scorable = True
                    versus_naive.setdefault(channel, []).append(
                        (naive_mase, treatment_mase))
                    abstention_priced.append((naive_mase, treatment_mase))
                    if naive_smape is not None and treatment_smape is not None:
                        versus_naive_smape.setdefault(channel, []).append(
                            (naive_smape, treatment_smape))
                        abstention_priced_smape.append(
                            (naive_smape, treatment_smape))
            else:
                # Pricing an abstention at the registered robust fallback
                # prevents a policy from improving its score by suppressing
                # difficult channels.
                abstention_priced.append((naive_mase, naive_mase))
                if naive_smape is not None:
                    abstention_priced_smape.append((naive_smape, naive_smape))
                treatment_smape = None
            scoring_key = coverage_bucket(base_scorable, treatment_scorable)
            scoring_coverage[scoring_key] += 1

            # Score each channel independently.  Record-level official metrics
            # are all-or-nothing, so using their detail payload here would
            # silently discard a valid sibling whenever another channel has a
            # malformed horizon.
            if base_scorable and treatment_scorable:
                b_label = (base_support.get(task_id) or {}).get(
                    channel, "unlabeled")
                t_label = (treat_support.get(task_id) or {}).get(
                    channel, "unlabeled")
                base_mix[b_label] = base_mix.get(b_label, 0) + 1
                treat_mix[t_label] = treat_mix.get(t_label, 0) + 1
                outcome = (
                    "safety_preservation"
                    if math.isclose(treatment_mase, naive_mase,
                                    rel_tol=0, abs_tol=1e-12)
                    else "uplift" if treatment_mase < naive_mase
                    else "regression"
                )
                paired_channel_records.append({
                    "task_id": task_id, "channel": channel,
                    "baseline_mase": base_mase,
                    "treatment_mase": treatment_mase,
                    "last_value_mase": naive_mase,
                    "baseline_smape": base_smape,
                    "treatment_smape": treatment_smape,
                    "last_value_smape": naive_smape,
                    "treatment_vs_last_value": outcome,
                    "baseline_support": b_label,
                    "treatment_support": t_label,
                })
                per_channel.setdefault(channel, []).append(
                    (base_mase, treatment_mase))
                if base_smape is not None and treatment_smape is not None:
                    per_channel_smape.setdefault(channel, []).append(
                        (base_smape, treatment_smape))
                if stable_scaled_error_denominator(channel_history):
                    per_channel_stable.setdefault(channel, []).append(
                        (base_mase, treatment_mase))
        if not both:
            continue

        b_sum, _ = subset_metrics(
            metrics_module, truth, history, b_fc, both)
        t_sum, _ = subset_metrics(
            metrics_module, truth, history, t_fc, both)
        if not b_sum or not t_sum:
            continue

        record_rows.append({
            "task_id": task_id,
            "channels": both,
            "baseline": b_sum,
            "treatment": t_sum,
        })

    def stable_summary(channel, pairs):
        result = summarise_pairs(pairs)
        stable_pairs = per_channel_stable.get(channel, [])
        result["near_constant_denominators_excluded"] = (
            len(pairs) - len(stable_pairs))
        result["stable_history"] = (summarise_pairs(stable_pairs)
                                    if stable_pairs else None)
        return result

    all_pairs = [p for pairs in per_channel.values() for p in pairs]
    all_smape_pairs = [p for pairs in per_channel_smape.values() for p in pairs]
    if len(all_pairs) != scoring_coverage["both"]:
        raise AssertionError(
            "paired score count diverged from independently scorable "
            f"coverage: {len(all_pairs)} != {scoring_coverage['both']}")
    stable_all = [pair for pairs in per_channel_stable.values()
                  for pair in pairs]

    def mix_line(mix: dict[str, int]) -> str:
        return ", ".join(f"{label} {count}"
                         for label, count in sorted(mix.items())) or "(none)"

    result = {
        "baseline_code_revision": read_manifest(baseline_dir).get(
            "code_revision"),
        "treatment_code_revision": read_manifest(treatment_dir).get(
            "code_revision"),
        "coverage": coverage,
        # Presence alone is insufficient: a model may return a named array
        # with the wrong horizon or non-numeric values. Keep the historical
        # presence accounting, but make the denominator behind actual scores
        # explicit so malformed forecasts cannot masquerade as coverage.
        "scoring_coverage": scoring_coverage,
        "support_mix": {"baseline": base_mix, "treatment": treat_mix},
        "per_channel": {c: stable_summary(c, p)
                        for c, p in per_channel.items()},
        "per_channel_smape": {
            c: summarise_pairs(pairs)
            for c, pairs in per_channel_smape.items()
        },
        "treatment_vs_last_value": {
            c: summarise_pairs(pairs) for c, pairs in versus_naive.items()
        },
        "treatment_vs_last_value_smape": {
            c: summarise_pairs(pairs)
            for c, pairs in versus_naive_smape.items()
        },
        "publication": {
            "eligible_channel_slots_with_naive_scale": len(abstention_priced),
            "treatment_published": coverage["both"] + coverage["treat_only"],
            "treatment_abstained_where_baseline_published": coverage["base_only"],
            "abstention_priced_as_last_value": (
                summarise_pairs(abstention_priced) if abstention_priced else None),
            "abstention_priced_as_last_value_smape": (
                summarise_pairs(abstention_priced_smape)
                if abstention_priced_smape else None),
        },
        "overall": summarise_pairs(all_pairs) if all_pairs else None,
        "overall_smape": (summarise_pairs(all_smape_pairs)
                           if all_smape_pairs else None),
        "overall_stable_history": (summarise_pairs(stable_all)
                                   if stable_all else None),
        "records": record_rows,
        "raw_paired_channel_records": paired_channel_records,
        "outcome_counts_vs_last_value": {
            label: sum(row["treatment_vs_last_value"] == label
                       for row in paired_channel_records)
            for label in ("uplift", "safety_preservation", "regression",
                          "unclassified")
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    if args.output:
        return 0

    total_channels = sum(coverage.values())
    print(f"records compared: {len(record_rows)} "
          f"(of {len(shared_ids)} with truth in both arms)")
    print(f"channel coverage over {total_channels} channel-slots: "
          f"both {coverage['both']}, baseline-only {coverage['base_only']}, "
          f"treatment-only {coverage['treat_only']}, "
          f"neither {coverage['neither']}")
    print()
    print(f"{'channel':16s} {'n':>4s} {'base MASE':>11s} "
          f"{'treat MASE':>11s} {'treat wins':>11s}")
    for channel, pairs in sorted(per_channel.items()):
        s = stable_summary(channel, pairs)
        print(f"{channel:16s} {s['n']:4d} {s['baseline_median']:11.4f} "
              f"{s['treatment_median']:11.4f} "
              f"{s['treatment_wins']:>6d}/{s['n']:<4d}")
    for label, summary in (("ALL", result["overall"]),
                           ("ALL stable-scale",
                            result["overall_stable_history"])):
        if summary:
            print(f"{label:16s} {summary['n']:4d} "
                  f"{summary['baseline_median']:11.4f} "
                  f"{summary['treatment_median']:11.4f} "
                  f"{summary['treatment_wins']:>6d}/{summary['n']:<4d}")
    print()
    print("support-label mix over the compared channel scores "
          "(best_effort = disclosed fallback rows carrying NO RELIABLE "
          "FORECAST, not supported forecasts; unlabeled = the arm "
          "records no support labels):")
    print(f"  baseline:  {mix_line(base_mix)}")
    print(f"  treatment: {mix_line(treat_mix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
