"""Conformance benchmark for model-neutral forecast adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gnomon.forecast_adapter import (  # noqa: E402
    ForecastAdapterError, ForecastRequest, ForecastResult,
    LegacyModelAdapter, StatisticalAdapter, conformance_report,
)
from gnomon.models import MODELS, predict  # noqa: E402
from gnomon.tsfm import get_tsfm, installed_tsfms  # noqa: E402
from gnomon.tsfm_sandbox import (  # noqa: E402
    SubprocessAdapter, sandbox_available_tsfms,
)


class _Adversarial:
    kind = "test"
    revision = "fixture"
    def __init__(self, name: str):
        self.name = name
    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if self.name == "short_horizon":
            return ForecastResult(tuple([1.0] * (request.horizon - 1)))
        if self.name == "nonfinite":
            return ForecastResult(tuple([float("inf")] * request.horizon))
        if self.name == "nondeterministic":
            self.counter = getattr(self, "counter", 0) + 1
            return ForecastResult(tuple([float(self.counter)] * request.horizon))
        raise ForecastAdapterError(self.name)


def run() -> dict[str, object]:
    valid = {name: conformance_report(StatisticalAdapter(name, predict))
             for name in sorted(MODELS)}
    adversarial = {name: conformance_report(_Adversarial(name))
                   for name in ("short_horizon", "nonfinite", "nondeterministic")}
    installed = {
        f"in_process:{name}": conformance_report(
            LegacyModelAdapter(get_tsfm(name)))
        for name in installed_tsfms()
    }
    installed.update({
        f"sandbox:{name}": conformance_report(
            LegacyModelAdapter(SubprocessAdapter(name)))
        for name in sandbox_available_tsfms()
    })
    gates = {
        "all_statistical_adapters_conform": all(
            item["conformant"] for item in valid.values()),
        "all_adversarial_adapters_rejected": all(
            not item["conformant"] for item in adversarial.values()),
        "all_installed_external_adapters_conform": all(
            item["conformant"] for item in installed.values()),
    }
    return {"schema_version": "0.1", "valid": valid,
            "installed_external": installed,
            "external_coverage": {
                "tested": len(installed),
                "note": ("No external TSFM is installed in this environment."
                         if not installed else
                         "Every locally installed external adapter was executed.")},
            "adversarial": adversarial, "gates": gates,
            "graduated": all(gates.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run()
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["graduated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
