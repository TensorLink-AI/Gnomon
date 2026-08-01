"""Run the AnomLLM benchmark's Aion treatment (and optionally the LLM
control) against an official AnomLLM checkout.

The datasets, prompt variants, and metrics are the official ones; this
script writes Aion's predictions into the official results tree and can
invoke the official ``online_api.py`` for the control condition.

Examples
--------
Aion treatment on the ``point`` dataset::

    python -m benchmarks.anomllm.run_anomllm \
        --anomllm-root ~/AnomLLM --data point

LLM control via OpenRouter (uses the official runner unchanged; first
copy benchmarks/anomllm/credentials.example.yml to the checkout as
credentials.yml and fill in your key)::

    python -m benchmarks.anomllm.run_anomllm \
        --anomllm-root ~/AnomLLM --data point \
        --control-model openai/gpt-4o-mini --control-variant 0shot-text

Official scoring (from the AnomLLM checkout)::

    python src/result_agg.py --data_name point \
        --label_name point-exp --table_caption "Point anomalies"

The resulting table contains one row per condition — Aion appears as
``aion`` next to every LLM variant, scored by identical code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.anomllm.aion_detector import run_aion_condition  # noqa: E402

DATASETS = (
    "point", "range", "freq", "trend", "flat-trend",
    "noisy-point", "noisy-freq",
)


def run_control(anomllm_root: Path, data: str, model: str, variant: str) -> int:
    """Invoke the official online runner for the LLM control condition."""
    credentials = anomllm_root / "credentials.yml"
    if not credentials.exists():
        raise SystemExit(
            f"{credentials} not found. Copy "
            "benchmarks/anomllm/credentials.example.yml there and add your "
            "OpenRouter key for the model you want to run."
        )
    command = [
        sys.executable, "src/online_api.py",
        "--data", data, "--model", model, "--variant", variant,
    ]
    print("Running official control:", " ".join(command))
    return subprocess.call(command, cwd=anomllm_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--anomllm-root", required=True,
                        help="Path to a clone of rose-stl-lab/AnomLLM")
    parser.add_argument("--data", required=True,
                        help=f"Dataset name, e.g. one of {', '.join(DATASETS)}")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Aion detection threshold (default: Aion's)")
    parser.add_argument("--variant-name", default="detect",
                        help="Variant label for Aion's results file")
    parser.add_argument("--skip-aion", action="store_true",
                        help="Only run the control condition")
    parser.add_argument("--control-model", default=None,
                        help="OpenRouter model id for the official LLM control")
    parser.add_argument("--control-variant", default="0shot-text",
                        help="Official prompt variant for the control")
    args = parser.parse_args()

    anomllm_root = Path(args.anomllm_root).expanduser().resolve()
    if not (anomllm_root / "src" / "result_agg.py").exists():
        raise SystemExit(
            f"{anomllm_root} does not look like an AnomLLM checkout "
            "(src/result_agg.py missing)."
        )

    exit_code = 0
    if not args.skip_aion:
        summary = run_aion_condition(
            anomllm_root, args.data,
            threshold=args.threshold, variant_name=args.variant_name,
        )
        print(json.dumps(summary, indent=2))

    if args.control_model:
        exit_code = run_control(
            anomllm_root, args.data, args.control_model, args.control_variant
        )

    print(
        "\nScore all conditions with the official aggregator:\n"
        f"  cd {anomllm_root} && python src/result_agg.py "
        f"--data_name {args.data} --label_name {args.data}-exp "
        f'--table_caption "{args.data} anomalies"'
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
