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
from collections import OrderedDict
from pathlib import Path
from statistics import median

from .tasks import load_official_metrics


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
            history = (row.get("input") or {}).get("history")
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


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    metrics_module = load_official_metrics(data_dir)

    truth_by_id = load_truth(data_dir)
    base, base_support = load_forecasts(Path(args.baseline).expanduser())
    treat, treat_support = load_forecasts(Path(args.treatment).expanduser())

    shared_ids = sorted(set(base) & set(treat) & set(truth_by_id))
    per_channel: dict[str, list[tuple[float, float]]] = {}
    record_rows = []
    coverage = {"base_only": 0, "treat_only": 0, "both": 0, "neither": 0}
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
            key = ("both" if in_b and in_t else
                   "base_only" if in_b else
                   "treat_only" if in_t else "neither")
            coverage[key] += 1
        if not both:
            continue

        b_sum, b_detail = subset_metrics(
            metrics_module, truth, history, b_fc, both)
        t_sum, t_detail = subset_metrics(
            metrics_module, truth, history, t_fc, both)
        if not b_sum or not t_sum:
            continue

        for channel in both:
            b_label = (base_support.get(task_id) or {}).get(channel, "unlabeled")
            t_label = (treat_support.get(task_id) or {}).get(channel, "unlabeled")
            base_mix[b_label] = base_mix.get(b_label, 0) + 1
            treat_mix[t_label] = treat_mix.get(t_label, 0) + 1
        record_rows.append({
            "task_id": task_id,
            "channels": both,
            "baseline": b_sum,
            "treatment": t_sum,
        })
        for channel in both:
            b_ch = (b_detail or {}).get(channel) or {}
            t_ch = (t_detail or {}).get(channel) or {}
            b_val, t_val = b_ch.get("MASE"), t_ch.get("MASE")
            if b_val is not None and t_val is not None:
                per_channel.setdefault(channel, []).append((b_val, t_val))

    def summarise(pairs):
        b = [p[0] for p in pairs]
        t = [p[1] for p in pairs]
        wins = sum(1 for x, y in pairs if y < x)
        return {
            "n": len(pairs),
            "baseline_median": round(median(b), 4),
            "treatment_median": round(median(t), 4),
            "treatment_wins": wins,
        }

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
        s = summarise(pairs)
        print(f"{channel:16s} {s['n']:4d} {s['baseline_median']:11.4f} "
              f"{s['treatment_median']:11.4f} "
              f"{s['treatment_wins']:>6d}/{s['n']:<4d}")

    all_pairs = [p for pairs in per_channel.values() for p in pairs]
    if all_pairs:
        s = summarise(all_pairs)
        print(f"\n{'ALL':16s} {s['n']:4d} {s['baseline_median']:11.4f} "
              f"{s['treatment_median']:11.4f} "
              f"{s['treatment_wins']:>6d}/{s['n']:<4d}")

    def mix_line(mix: dict[str, int]) -> str:
        return ", ".join(f"{label} {count}"
                         for label, count in sorted(mix.items())) or "(none)"

    print()
    print("support-label mix over the compared channel scores "
          "(best_effort = disclosed fallback rows carrying NO RELIABLE "
          "FORECAST, not supported forecasts; unlabeled = the arm "
          "records no support labels):")
    print(f"  baseline:  {mix_line(base_mix)}")
    print(f"  treatment: {mix_line(treat_mix)}")

    if args.json:
        print(json.dumps({
            "coverage": coverage,
            "support_mix": {"baseline": base_mix, "treatment": treat_mix},
            "per_channel": {c: summarise(p) for c, p in per_channel.items()},
            "records": record_rows,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
