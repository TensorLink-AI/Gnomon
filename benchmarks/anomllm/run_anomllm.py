"""Run the AnomLLM benchmark's Gnomon treatment (and optionally the LLM
control) against an official AnomLLM checkout.

The datasets, prompt variants, and metrics are the official ones; this
script writes Gnomon's predictions into the official results tree and can
invoke the official ``online_api.py`` for the control condition.

Examples
--------
Gnomon treatment on the ``point`` dataset::

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

The resulting table contains one row per condition — Gnomon appears as
``gnomon`` next to every LLM variant, scored by identical code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.anomllm.gnomon_detector import (  # noqa: E402
    RESCALED_VARIANT,
    default_records_path,
    run_gnomon_condition,
)
from benchmarks.common.manifest import write_manifest  # noqa: E402

DATASETS = (
    "point", "range", "freq", "trend", "flat-trend",
    "noisy-point", "noisy-freq",
)

#: Variant names the official aggregator postprocesses: for exactly these
#: names (upstream ``postprocess_configs`` in src/config.py), every
#: integer in a stored response is multiplied by 1/scale before scoring,
#: because those variants prompt the LLM on a 0.3-subsampled series.


def gnomon_variant_name(name: str) -> str:
    """Argparse type for ``--variant-name``: rejects names upstream
    silently rescales.

    Gnomon's predictions are full-resolution indices. A results file
    named like a scaled prompt variant would have every index multiplied
    by the inverse scale at aggregation time — no error, just wrong
    scores — so the collision is refused up front.
    """
    if RESCALED_VARIANT.match(name):
        raise argparse.ArgumentTypeError(
            f"variant name {name!r} collides with an upstream rescaling "
            "variant: postprocess_configs (src/config.py in the AnomLLM "
            "checkout) multiplies every integer in files with that exact "
            "name by the inverse scale before scoring, which would "
            "silently corrupt Gnomon's full-resolution indices. Pick a "
            "name outside the [01]shot-text-sN[-cot] pattern."
        )
    return name


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
                        help="Gnomon detection threshold (default: Gnomon's)")
    parser.add_argument("--variant-name", default="detect",
                        type=gnomon_variant_name,
                        help="Variant label for Gnomon's results file "
                             "(names upstream would rescale are rejected)")
    parser.add_argument("--skip-gnomon", action="store_true",
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
    command = " ".join(
        [sys.executable, "-m", "benchmarks.anomllm.run_anomllm"]
        + sys.argv[1:]
    )
    if not args.skip_gnomon:
        summary = run_gnomon_condition(
            anomllm_root, args.data,
            threshold=args.threshold, variant_name=args.variant_name,
        )
        print(json.dumps(summary, indent=2))
        # Same provenance run_all.py records, next to the sidecar, so a
        # later comparison can refuse arms that answered other questions.
        write_manifest(
            Path(summary["gnomonbench_file"]).parent,
            benchmark="anomllm",
            run_name=f"anomllm-{args.data}",
            condition=f"gnomon/{args.variant_name}",
            target=args.data,
            command=command,
            status="ok",
        )

    if args.control_model:
        exit_code = run_control(
            anomllm_root, args.data, args.control_model, args.control_variant
        )
        # The control's predictions live in the official results tree,
        # owned by upstream code; its provenance still gets an
        # adapter-owned home in the gnomonbench/ tree.
        control_dir = default_records_path(
            anomllm_root, args.data,
            args.control_model.replace("/", "--"), args.control_variant,
        ).parent
        write_manifest(
            control_dir,
            benchmark="anomllm",
            run_name=f"anomllm-{args.data}",
            condition=f"control/{args.control_variant}",
            target=args.data,
            model=args.control_model,
            command=command,
            status="ok" if exit_code == 0 else f"exit {exit_code}",
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
