"""Sandboxed TSFM execution: each foundation model runs in its own isolated venv.

The core problem: TSFM libraries have conflicting transitive dependencies
(e.g. momentfm pins numpy==1.25.2 while uni2ts needs numpy>=1.26.0).
They cannot coexist in a single Python environment.

The solution: each TSFM adapter gets its own isolated venv managed by ``uv``.
Gnomon's main process stays zero-dependency and communicates with each sandbox
via a subprocess that speaks JSON over stdin/stdout.

Architecture::

    Gnomon main process (no TSFM deps)
        │
        ├─ gnomon.tsfm.SubprocessAdapter("chronos_bolt_mini")
        │     │ spawns: uv run --with chronos-forecasting --with torch python -m gnomon.tsfm._worker chronos_bolt_mini
        │     │ stdin:  {"history": [...], "horizon": 24, "season": 24, "quantiles": [0.1, 0.5, 0.9]}
        │     │ stdout: {"point": [...], "quantiles": [{"0.1": ..., "0.5": ..., "0.9": ...}, ...]}
        │     └─ isolated venv (chronos + its own torch + numpy)
        │
        ├─ gnomon.tsfm.SubprocessAdapter("moment_small")
        │     │ spawns: uv run --with momentfm --with torch python -m gnomon.tsfm._worker moment_small
        │     └─ isolated venv (momentfm + its own numpy==1.25.2 + torch)
        │
        └─ gnomon.tsfm.SubprocessAdapter("moirai2_small")
              │ spawns: uv run --with uni2ts --with torch python -m gnomon.tsfm._worker moirai2_small
              └─ isolated venv (uni2ts + its own numpy>=1.26.0 + torch)

Each venv is created lazily on first use and cached. The worker script is a
tiny self-contained Python module that imports the TSFM library, loads the
model, reads a JSON request from stdin, and writes a JSON response to stdout.

This preserves Gnomon's core invariants:
  - Zero dependencies for the base install (no torch, no numpy pinning).
  - TSFMs are candidates — they compete against baselines on identical folds.
  - Missing or broken sandboxes are a soft skip, not an error.
  - The adapter protocol (predict / predict_quantiles) is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from .tsfm import TSFMAdapter, TSFMError, TSFMUnavailable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: where to store sandbox venvs
# ---------------------------------------------------------------------------

SANDBOX_ROOT = Path(
    os.environ.get("GNOMON_TSFM_SANDBOX_ROOT", "")
) or Path.home() / ".cache" / "gnomon-tsfm-venvs"

# Each TSFM's pip-install spec. The key must match the adapter name in
# tsfm.py. Every spec is exact — ``==`` for PyPI, a tag for git.
#
# Unpinned specs made a sandbox's numbers a function of the day it was
# built, which a content-addressed forecast_id cannot express: two runs
# with the same id could disagree, and first-write-wins would discard the
# second. The module docstring already cites numpy conflicts as a known
# sensitivity, so this surface was never safely floating.
#
# Resolved 2026-08-02 from PyPI and from granite-tsfm's tag list. To move a
# pin, bump it here and re-run the TSFM benchmark: a library change can move
# published numbers exactly as a weight change can.
TSFM_PIP_SPECS: dict[str, list[str]] = {
    "chronos_bolt_mini": [
        "chronos-forecasting==2.3.1",
        "torch==2.13.0",
    ],
    "chronos_bolt_small": [
        "chronos-forecasting==2.3.1",
        "torch==2.13.0",
    ],
    "toto2_22m": [
        "toto-models==1.0.0",
        "torch==2.13.0",
    ],
    "flowstate": [
        # tsfm_public is not on PyPI — install from GitHub at a fixed tag.
        "git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.7",
        "torch==2.13.0",
    ],
    "ttm": [
        "git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.7",
        "torch==2.13.0",
    ],
    "moirai2_small": [
        "uni2ts==2.0.0",
        "torch==2.13.0",
    ],
    "moment_small": [
        "momentfm==0.1.4",
        "torch==2.13.0",
    ],
}

# Which Python to use for sandboxes (defaults to current interpreter)
SANDBOX_PYTHON = os.environ.get("GNOMON_TSFM_PYTHON", sys.executable)


# ---------------------------------------------------------------------------
# Sandbox management
# ---------------------------------------------------------------------------

def _sandbox_dir(name: str) -> Path:
    """Return the directory for a TSFM's sandbox venv."""
    safe_name = name.replace("/", "_").replace(":", "_")
    return SANDBOX_ROOT / safe_name


def _uv_available() -> bool:
    return shutil.which("uv") is not None


def ensure_sandbox(name: str, force: bool = False) -> Path:
    """Create or verify a sandboxed venv for the given TSFM.

    Returns the path to the venv directory.

    Raises:
        TSFMUnavailable: If uv is not installed or the TSFM is unknown.
        TSFMError: If venv creation or dependency installation fails.
    """
    if name not in TSFM_PIP_SPECS:
        raise TSFMUnavailable(f"Unknown TSFM for sandboxing: {name}")

    if not _uv_available():
        raise TSFMUnavailable(
            "uv is required for sandboxed TSFM execution but is not installed. "
            "Install it from https://docs.astral.sh/uv/"
        )

    venv_dir = _sandbox_dir(name)
    marker = venv_dir / ".gnomon-sandbox-ready"

    if marker.exists() and not force:
        logger.debug("Sandbox for %s already exists at %s", name, venv_dir)
        return venv_dir

    # Create venv directory
    venv_dir.mkdir(parents=True, exist_ok=True)

    # Create the venv using uv
    specs = TSFM_PIP_SPECS[name]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    logger.info("Creating sandbox venv for %s at %s", name, venv_dir)

    # Step 1: create the venv
    result = subprocess.run(
        [
            "uv", "venv",
            "--python", python_version,
            str(venv_dir / "venv"),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise TSFMError(
            f"Failed to create venv for {name}: {result.stderr.strip()}"
        )

    # Step 2: install dependencies into the venv
    pip_args = [
        "uv", "pip", "install",
        "--python", str(venv_dir / "venv" / "bin" / "python"),
    ] + specs

    logger.info("Installing %s deps: %s", name, " ".join(specs))
    result = subprocess.run(
        pip_args,
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise TSFMError(
            f"Failed to install dependencies for {name}: {result.stderr.strip()[:500]}"
        )

    # Step 3: write marker file
    marker.write_text(f"tsfm={name}\ncreated_by=gnomon\n")
    logger.info("Sandbox for %s ready", name)
    return venv_dir


def sandbox_exists(name: str) -> bool:
    """Check if a sandbox venv has been created for the given TSFM."""
    return _sandbox_dir(name).joinpath(".gnomon-sandbox-ready").exists()


def remove_sandbox(name: str) -> None:
    """Remove a sandbox venv."""
    venv_dir = _sandbox_dir(name)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        logger.info("Removed sandbox for %s", name)


def list_sandboxes() -> list[str]:
    """Return names of TSFMs with existing sandboxes."""
    if not SANDBOX_ROOT.exists():
        return []
    return sorted(
        d.name for d in SANDBOX_ROOT.iterdir()
        if d.is_dir() and (d / ".gnomon-sandbox-ready").exists()
    )


# ---------------------------------------------------------------------------
# Worker script (runs inside the sandbox venv)
# ---------------------------------------------------------------------------

# This script is written into the sandbox dir and run by the sandbox Python.
# It must be self-contained — it imports the TSFM library, loads the model,
# reads a JSON request from stdin, and writes a JSON response to stdout.
WORKER_SCRIPT = textwrap.dedent("""\
    '''Sandbox worker: load a TSFM, read JSON from stdin, write JSON to stdout.'''

    import json
    import sys
    import traceback


    def main():
        request = json.load(sys.stdin)
        tsfm_name = request["tsfm_name"]
        mode = request.get("mode", "predict")
        history = request["history"]

        try:
            if mode == "reconstruct":
                result = run_reconstruct(tsfm_name, history, request.get("mask"))
            elif mode == "embed":
                result = run_embed(tsfm_name, history)
            elif mode == "predict":
                result = run_tsfm(
                    tsfm_name, history, request["horizon"], request["season"],
                    request.get("quantiles", [0.1, 0.5, 0.9]),
                    request.get("frequency", "h"),
                    request.get("want_quantiles", False),
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            json.dump(result, sys.stdout)
        except Exception as exc:
            json.dump({"error": str(exc), "traceback": traceback.format_exc()}, sys.stdout)
            sys.exit(1)


    def run_tsfm(name, history, horizon, season, quantiles, frequency, want_quantiles):
        import torch

        if name == "chronos_bolt_mini" or name == "chronos_bolt_small":
            return run_chronos(name, history, horizon, quantiles, want_quantiles)
        elif name == "toto2_22m":
            return run_toto(history, horizon, quantiles, want_quantiles)
        elif name == "flowstate":
            return run_flowstate(history, horizon, quantiles, frequency, want_quantiles)
        elif name == "ttm":
            return run_ttm(history, horizon, want_quantiles)
        elif name == "moirai2_small":
            return run_moirai(history, horizon, quantiles, want_quantiles)
        elif name == "moment_small":
            return run_moment(history, horizon, want_quantiles)
        else:
            raise ValueError(f"Unknown TSFM: {name}")


    def run_chronos(name, history, horizon, quantiles, want_quantiles):
        import torch
        from chronos import BaseChronosPipeline

        model_id = {
            "chronos_bolt_mini": "amazon/chronos-bolt-mini",
            "chronos_bolt_small": "amazon/chronos-bolt-small",
        }[name]

        pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32,
        )
        context = torch.tensor(history, dtype=torch.float32)
        forecast = pipeline.predict(
            context=context,
            prediction_length=horizon,
            quantile_levels=list(quantiles),
        )
        import numpy as np
        arr = forecast.numpy()
        if arr.ndim == 3:
            arr = arr[0]
        # arr shape: [num_quantiles, horizon]
        median_idx = arr.shape[0] // 2
        point = arr[median_idx].tolist()

        if not want_quantiles:
            return {"point": point}

        steps = []
        for s in range(arr.shape[1]):
            row = {str(q): float(arr[i, s]) for i, q in enumerate(quantiles)}
            steps.append(row)
        return {"point": point, "quantiles": steps}


    def run_toto(history, horizon, quantiles, want_quantiles):
        import torch
        from toto2 import Toto2Model

        model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m")
        model = model.to("cpu").eval()

        target = torch.tensor(history, dtype=torch.float32)
        target = target.unsqueeze(0).unsqueeze(0)  # (batch, n_var, time)
        mask = torch.ones_like(target, dtype=torch.bool)
        ids = torch.zeros(1, 1, dtype=torch.long)

        q = model.forecast(
            {"target": target, "target_mask": mask, "series_ids": ids},
            horizon=horizon,
            has_missing_values=False,
        )
        arr = q.detach().cpu().numpy().squeeze()  # (9, horizon)
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        median_idx = 4
        point = arr[median_idx].tolist()

        if not want_quantiles:
            return {"point": point}

        steps = []
        for s in range(arr.shape[1]):
            row = {}
            for rq in quantiles:
                idx = min(range(len(levels)), key=lambda i: abs(levels[i] - rq))
                row[str(rq)] = float(arr[idx, s])
            steps.append(row)
        return {"point": point, "quantiles": steps}


    def run_flowstate(history, horizon, quantiles, frequency, want_quantiles):
        import torch
        from tsfm_public import FlowStateForPrediction

        predictor = FlowStateForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-flowstate-r1", revision="r1.1",
        ).to("cpu")

        scale_map = {"h": 1.0, "D": 3.43, "W": 0.46, "MS": 2.0}
        scale = scale_map.get(frequency, 1.0)

        ts = torch.tensor(history, dtype=torch.float32)
        ts = ts.unsqueeze(1).unsqueeze(2)  # (context, batch, n_ch)
        result = predictor(ts, scale_factor=scale, prediction_length=horizon, batch_first=False)
        arr = result.prediction_outputs.detach().cpu().numpy().squeeze()

        median_idx = arr.shape[0] // 2
        point = arr[median_idx].tolist()

        if not want_quantiles:
            return {"point": point}

        fs_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        steps = []
        for s in range(arr.shape[1]):
            row = {}
            for rq in quantiles:
                idx = min(range(len(fs_levels)), key=lambda i: abs(fs_levels[i] - rq))
                row[str(rq)] = float(arr[idx, s])
            steps.append(row)
        return {"point": point, "quantiles": steps}


    def run_ttm(history, horizon, want_quantiles):
        import torch
        from tsfm_public import TinyTimeMixerForPrediction

        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2",
        )

        ctx_len = min(len(history), 512)
        ctx = history[-ctx_len:]
        ts = torch.tensor(ctx, dtype=torch.float32)
        ts = ts.unsqueeze(0).unsqueeze(0)

        output = model(ts, prediction_length=horizon)
        arr = output.prediction_outputs.detach().cpu().numpy().squeeze()
        if arr.ndim > 1:
            arr = arr[-1]
        point = arr[:horizon].tolist()

        return {"point": point, "quantiles": None}


    def run_moirai(history, horizon, quantiles, want_quantiles):
        import torch
        import pandas as pd
        from gluonts.dataset.pandas import PandasDataset
        from gluonts.dataset.split import split
        from uni2ts.model.moirai import Moirai2Forecast, Moirai2Module

        module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        model = Moirai2Forecast(
            module=module,
            prediction_length=horizon,
            context_length=min(512, 1000),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to("cpu")

        idx = pd.date_range(start="2000-01-01", periods=len(history), freq="h")
        df = pd.DataFrame({"target": history}, index=idx)
        ds = PandasDataset(dict(df))

        _, test_template = split(ds, offset=-horizon)
        test_data = test_template.generate_instances(
            prediction_length=horizon, windows=1, distance=horizon,
        )

        predictor = model.create_predictor(batch_size=1)
        forecasts = predictor.predict(test_data.input)
        forecast = next(iter(forecasts))
        point = forecast.mean.tolist()

        if not want_quantiles:
            return {"point": point}

        steps = []
        for s in range(horizon):
            row = {str(q): float(forecast.quantile(q)[s]) for q in quantiles}
            steps.append(row)
        return {"point": point, "quantiles": steps}


    def run_moment(history, horizon, want_quantiles):
        import torch
        from momentfm import MOMENTPipeline

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={"task_name": "forecasting", "forecast_horizon": horizon},
        )
        model.init()

        device = next(model.model.parameters()).device
        ctx_len = min(len(history), 512)
        ctx = history[-ctx_len:]
        if len(ctx) < 512:
            ctx = [0.0] * (512 - len(ctx)) + ctx
        ts = torch.tensor(ctx, dtype=torch.float32, device=device)
        ts = ts.unsqueeze(0).unsqueeze(0)

        model.model.forecast_horizon = horizon
        output = model.model(ts)
        arr = output.reconstruction.detach().cpu().numpy().squeeze()
        point = arr[:horizon].tolist()

        return {"point": point, "quantiles": None}


    def _moment_window(history):
        import torch
        ctx_len = min(len(history), 512)
        ctx = history[-ctx_len:]
        padding = 512 - len(ctx)
        padded = [0.0] * padding + ctx
        ts = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_mask = torch.tensor(
            [0.0] * padding + [1.0] * len(ctx), dtype=torch.float32,
        ).unsqueeze(0)
        return ts, input_mask, padding, ctx_len


    def run_reconstruct(name, history, mask):
        if name != "moment_small":
            raise ValueError(f"TSFM {name} does not implement reconstruction")
        import torch
        from momentfm import MOMENTPipeline

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={"task_name": "reconstruction"},
        )
        model.init()
        ts, input_mask, padding, ctx_len = _moment_window(history)
        if mask is not None:
            tail_mask = mask[-ctx_len:]
            for offset, observed in enumerate(tail_mask):
                if not observed:
                    input_mask[0, padding + offset] = 0.0
                    ts[0, 0, padding + offset] = 0.0
        with torch.no_grad():
            output = model.model(x_enc=ts, input_mask=input_mask)
        values = output.reconstruction.detach().cpu().numpy().squeeze()
        return {"reconstruction": values[padding:].tolist()}


    def run_embed(name, history):
        if name != "moment_small":
            raise ValueError(f"TSFM {name} does not implement embedding")
        import torch
        from momentfm import MOMENTPipeline

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={"task_name": "embedding"},
        )
        model.init()
        ts, input_mask, _, _ = _moment_window(history)
        with torch.no_grad():
            output = model.model(x_enc=ts, input_mask=input_mask)
        embedding = output.embeddings.detach().cpu().numpy().squeeze()
        if embedding.ndim > 1:
            embedding = embedding.mean(axis=0)
        return {"embedding": embedding.tolist()}


    if __name__ == "__main__":
        main()
""")


def _ensure_worker_script(sandbox_dir: Path) -> Path:
    """Write the worker script into the sandbox dir, refreshing it when the
    packaged script has changed (e.g. new verbs) since the sandbox was built."""
    script_path = sandbox_dir / "worker.py"
    if not script_path.exists() or script_path.read_text() != WORKER_SCRIPT:
        script_path.write_text(WORKER_SCRIPT)
    return script_path


# ---------------------------------------------------------------------------
# SubprocessAdapter: implements TSFMAdapter protocol via subprocess
# ---------------------------------------------------------------------------

class SubprocessAdapter:
    """A TSFM adapter that runs inference in an isolated sandbox venv.

    This implements the same ``TSFMAdapter`` protocol as the in-process
    adapters in ``tsfm.py``, but delegates all heavy work to a subprocess
    running in a dedicated venv with its own dependency tree.

    The subprocess communicates via JSON over stdin/stdout:
      - stdin:  {"tsfm_name": "...", "history": [...], "horizon": N, ...}
      - stdout: {"point": [...], "quantiles": [...]} or {"error": "..."}
    """

    def __init__(
        self,
        name: str,
        frequency: str = "h",
        timeout: int = 300,
    ):
        self.name = name
        self.frequency = frequency
        self.timeout = timeout
        # Infer metadata from the registration in tsfm.py
        self._params_m = self._lookup_params(name)
        self._supports_quantiles = self._lookup_supports_quantiles(name)

    @property
    def params_m(self) -> float:
        return self._params_m

    @property
    def supports_quantiles(self) -> bool:
        return self._supports_quantiles

    def _lookup_params(self, name: str) -> float:
        params_map = {
            "chronos_bolt_mini": 21.0,
            "chronos_bolt_small": 48.0,
            "toto2_22m": 22.0,
            "flowstate": 18.5,
            "ttm": 3.0,
            "moirai2_small": 14.0,
            "moment_small": 38.0,
        }
        return params_map.get(name, 0.0)

    def _lookup_supports_quantiles(self, name: str) -> bool:
        no_quantiles = {"ttm", "moment_small"}
        return name not in no_quantiles

    def _run_subprocess(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run the worker script in the sandbox venv and return the result."""
        try:
            sandbox_dir = ensure_sandbox(self.name)
        except TSFMUnavailable:
            raise
        except TSFMError as exc:
            raise TSFMError(f"Sandbox for {self.name} is not ready: {exc}") from exc

        worker = _ensure_worker_script(sandbox_dir)
        venv_python = sandbox_dir / "venv" / "bin" / "python"

        if not venv_python.exists():
            raise TSFMError(
                f"Sandbox venv for {self.name} exists but Python binary not found at {venv_python}"
            )

        request["tsfm_name"] = self.name
        request["frequency"] = self.frequency

        try:
            proc = subprocess.run(
                [str(venv_python), str(worker)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise TSFMError(
                f"Sandbox for {self.name} timed out after {self.timeout}s"
            )
        except FileNotFoundError as exc:
            raise TSFMError(f"Failed to start sandbox process: {exc}") from exc

        if proc.returncode != 0:
            # Try to parse error from stdout
            try:
                err_response = json.loads(proc.stdout)
                if "error" in err_response:
                    raise TSFMError(
                        f"Sandbox {self.name} failed: {err_response['error']}"
                    )
            except json.JSONDecodeError:
                pass
            raise TSFMError(
                f"Sandbox {self.name} exited with code {proc.returncode}: {proc.stderr[:500]}"
            )

        try:
            response = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise TSFMError(
                f"Sandbox {self.name} produced invalid JSON output: {exc}"
            ) from exc

        if "error" in response:
            raise TSFMError(f"Sandbox {self.name} returned error: {response['error']}")

        return response

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        request = {
            "history": history,
            "horizon": horizon,
            "season": season,
            "want_quantiles": False,
        }
        response = self._run_subprocess(request)
        point = response.get("point")
        if point is None:
            raise TSFMError(f"Sandbox {self.name} returned no point forecast")
        return point

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]] | None:
        if not self._supports_quantiles:
            return None
        request = {
            "history": history,
            "horizon": horizon,
            "season": season,
            "quantiles": list(quantiles),
            "want_quantiles": True,
        }
        response = self._run_subprocess(request)
        return response.get("quantiles")

    def _require_task(self, task: str) -> None:
        from .tsfm import tsfm_capabilities
        try:
            capabilities = tsfm_capabilities(self.name)
        except KeyError:
            raise TSFMError(f"Unknown TSFM adapter: {self.name}")
        if task not in capabilities.tasks:
            raise TSFMError(
                f"TSFM {self.name} does not implement task {task!r} "
                f"(verified tasks: {', '.join(capabilities.tasks)})"
            )

    def reconstruct(
        self, history: list[float], mask: list[int] | None = None,
    ) -> list[float]:
        """Masked reconstruction via the sandbox (see TSFMAdapter docs)."""
        self._require_task("detect_anomalies")
        request: dict[str, Any] = {"mode": "reconstruct", "history": history}
        if mask is not None:
            request["mask"] = mask
        response = self._run_subprocess(request)
        reconstruction = response.get("reconstruction")
        if reconstruction is None:
            raise TSFMError(f"Sandbox {self.name} returned no reconstruction")
        return reconstruction

    def embed(self, history: list[float]) -> list[float]:
        """Series embedding via the sandbox (see TSFMAdapter docs)."""
        self._require_task("embed")
        request: dict[str, Any] = {"mode": "embed", "history": history}
        response = self._run_subprocess(request)
        embedding = response.get("embedding")
        if embedding is None:
            raise TSFMError(f"Sandbox {self.name} returned no embedding")
        return embedding


# ---------------------------------------------------------------------------
# Registry integration: override check_tsfm and tsfm_candidates for sandbox mode
# ---------------------------------------------------------------------------

def sandbox_check_tsfm(name: str) -> bool:
    """Check if a TSFM sandbox is ready (venv created and deps installed)."""
    return sandbox_exists(name)


def sandbox_available_tsfms() -> list[str]:
    """Return names of TSFMs with ready sandboxes."""
    return list_sandboxes()


def sandbox_tsfm_candidates(
    requested: list[str] | None = None,
    frequency: str = "h",
) -> list[TSFMAdapter]:
    """Return SubprocessAdapter instances for all ready sandboxes.

    Unlike ``tsfm_candidates`` in tsfm.py, this returns adapters that
    run in isolated venvs, avoiding dependency conflicts.
    """
    ready = list_sandboxes()
    if requested:
        ready = [name for name in ready if name in requested]
    candidates: list[TSFMAdapter] = []
    for name in ready:
        try:
            adapter = SubprocessAdapter(name, frequency=frequency)
            candidates.append(adapter)
            logger.info("Sandbox TSFM '%s' ready", name)
        except Exception:
            logger.debug("Failed to create SubprocessAdapter for %s", name, exc_info=True)
    return candidates
