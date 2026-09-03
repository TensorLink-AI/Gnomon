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
import atexit
import select
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any, Callable

from .tsfm import (
    TSFMAdapter,
    TSFMError,
    TSFMUnavailable,
    resolved_weights,
    tsfm_capabilities,
    tsfm_parameter_count,
    tsfm_supports_quantiles,
)

logger = logging.getLogger(__name__)
_ADAPTER_POOL: dict[tuple[str, str, str], "SubprocessAdapter"] = {}
_ADAPTER_POOL_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Configuration: where to store sandbox venvs
# ---------------------------------------------------------------------------

# `Path("") or default` never falls through — Path("") is PosixPath('.'),
# which is truthy — so the old expression rooted every sandbox in whatever
# directory the process happened to be started from, and `gnomon
# capabilities` answered differently per cwd. Test the string, not the Path.
_SANDBOX_ROOT_OVERRIDE = os.environ.get("GNOMON_TSFM_SANDBOX_ROOT", "")
SANDBOX_ROOT = (
    Path(_SANDBOX_ROOT_OVERRIDE) if _SANDBOX_ROOT_OVERRIDE
    else Path.home() / ".cache" / "gnomon-tsfm-venvs"
)

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
    "toto2_4m": [
        "toto-models==1.0.0",
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
    # A locally trained checkpoint, so there is no model package to install:
    # the export ships its own model.py and forecast_wrapper.py and needs
    # only the tensor stack to run them.
    "cascade": [
        "torch==2.13.0",
        "numpy==2.5.2",
        "safetensors==0.8.0",
    ],
}

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
        # The ready marker covers an executable sandbox, not only installed
        # wheels. Older installers wrote worker.py lazily on first inference,
        # which made a supposedly ready model fail in read-only production
        # images. Repair legacy installs while the operator is explicitly
        # checking/installing; current installs already have an exact worker.
        _ensure_worker_script(venv_dir)
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

    # Step 3: materialize the executable worker. The readiness marker is
    # deliberately last: once it exists, inference must not need to create a
    # file inside the installed environment.
    _ensure_worker_script(venv_dir)

    # Step 4: write marker file
    marker.write_text(f"tsfm={name}\ncreated_by=gnomon\n")
    logger.info("Sandbox for %s ready", name)
    return venv_dir


def sandbox_exists(name: str) -> bool:
    """Check if a sandbox venv has been created for the given TSFM."""
    return _sandbox_dir(name).joinpath(".gnomon-sandbox-ready").exists()


# --- Background installation, for callers that cannot block ---------------
#
# `ensure_sandbox` is synchronous and can legitimately run for minutes
# (torch is most of it). An MCP tool call blocking that long looks like a
# hang to every interactive host, so the agent-facing path starts the
# install as a detached `gnomon tsfm install` process and polls: the
# `.gnomon-sandbox-ready` marker is the single source of truth for
# success, the pid file distinguishes "still running" from "died", and
# the log tail is the evidence when it died.

def _install_log_path(name: str) -> Path:
    return _sandbox_dir(name) / "install.log"


def _install_pid_path(name: str) -> Path:
    return _sandbox_dir(name) / "install.pid"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def install_status(name: str) -> dict[str, Any]:
    """The state of a TSFM's sandbox: ready, installing, failed, or absent.

    Raises:
        TSFMUnavailable: If the TSFM name is unknown.
    """
    if name not in TSFM_PIP_SPECS:
        raise TSFMUnavailable(f"Unknown TSFM for sandboxing: {name}")
    if sandbox_exists(name):
        return {"state": "ready", "sandbox_path": str(_sandbox_dir(name))}
    pid_path = _install_pid_path(name)
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
        if pid is not None and _pid_alive(pid):
            return {"state": "installing", "pid": pid,
                    "log_path": str(_install_log_path(name))}
        log_path = _install_log_path(name)
        tail = ""
        if log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        return {"state": "failed", "log_tail": tail,
                "log_path": str(log_path)}
    return {"state": "absent"}


def start_install(name: str) -> dict[str, Any]:
    """Start a detached sandbox install and return without waiting.

    Idempotent: a sandbox that is ready or already installing is reported
    as such, never rebuilt. A previous failure is retried.

    Raises:
        TSFMUnavailable: If the TSFM name is unknown or uv is missing.
    """
    status = install_status(name)
    if status["state"] in ("ready", "installing"):
        return status
    if not _uv_available():
        raise TSFMUnavailable(
            "uv is required for sandboxed TSFM execution but is not installed. "
            "Install it from https://docs.astral.sh/uv/"
        )
    venv_dir = _sandbox_dir(name)
    venv_dir.mkdir(parents=True, exist_ok=True)
    log_path = _install_log_path(name)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "gnomon", "tsfm", "install", name],
            stdout=log_handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    _install_pid_path(name).write_text(str(process.pid), encoding="utf-8")
    logger.info("Started background install for %s (pid %d)", name, process.pid)
    return {"state": "installing", "pid": process.pid,
            "log_path": str(log_path)}


def remove_sandbox(name: str) -> bool:
    """Remove a known model sandbox and report whether it existed.

    Removal is intentionally stricter than path construction: an unknown name
    must never be transformed into a filesystem target, and a symlink must not
    be followed as a directory tree.
    """
    if name not in TSFM_PIP_SPECS:
        raise TSFMUnavailable(f"Unknown TSFM for sandboxing: {name}")
    venv_dir = _sandbox_dir(name)
    root = SANDBOX_ROOT.resolve()
    if venv_dir.parent.resolve() != root:
        raise TSFMError("Refusing to remove a sandbox outside GNOMON_TSFM_SANDBOX_ROOT")
    if venv_dir.is_symlink():
        venv_dir.unlink()
        logger.info("Removed sandbox link for %s", name)
        return True
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        logger.info("Removed sandbox for %s", name)
        return True
    return False


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

    # {model_id: commit sha} from the parent's TSFM_REVISIONS, delivered in
    # the request. Loading without it would fetch whatever the Hub serves
    # today — numbers the parent's content-addressed forecast_id (which
    # records the *pinned* revision) could not honestly cover.
    REVISIONS = {}
    MODELS = {}


    def pinned(model_id):
        revision = REVISIONS.get(model_id)
        if not revision:
            raise RuntimeError(
                "no pinned revision supplied for %r; refusing an unpinned "
                "weight load" % model_id
            )
        return revision


    def handle(request):
        tsfm_name = request["tsfm_name"]
        mode = request.get("mode", "predict")
        history = request.get("history", [])
        REVISIONS.update(request.get("revisions") or {})

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
            elif mode == "predict_batch":
                result = {"points": [
                    run_tsfm(
                        tsfm_name, item, request["horizon"], request["season"],
                        request.get("quantiles", [0.1, 0.5, 0.9]),
                        request.get("frequency", "h"), False,
                    )["point"]
                    for item in request["histories"]
                ]}
            else:
                raise ValueError(f"Unknown mode: {mode}")
            return result
        except Exception as exc:
            return {"error": str(exc), "traceback": traceback.format_exc()}


    def main():
        # A sandbox is a dependency boundary, not a per-call lifecycle.
        # Keep the process (and therefore the pinned model weights) alive and
        # exchange one JSON object per line.  EOF remains a clean shutdown.
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                result = handle(request)
            except Exception as exc:
                result = {"error": str(exc), "traceback": traceback.format_exc()}
            sys.stdout.write(json.dumps(result) + "\\n")
            sys.stdout.flush()


    def run_tsfm(name, history, horizon, season, quantiles, frequency, want_quantiles):
        import torch

        if name == "chronos_bolt_mini" or name == "chronos_bolt_small":
            return run_chronos(name, history, horizon, quantiles, want_quantiles)
        elif name == "toto2_4m" or name == "toto2_22m":
            return run_toto(name, history, horizon, quantiles, want_quantiles)
        elif name == "flowstate":
            return run_flowstate(history, horizon, quantiles, frequency, want_quantiles)
        elif name == "ttm":
            return run_ttm(history, horizon, want_quantiles)
        elif name == "moirai2_small":
            return run_moirai(history, horizon, quantiles, want_quantiles)
        elif name == "moment_small":
            return run_moment(history, horizon, want_quantiles)
        elif name == "cascade":
            return run_cascade(history, horizon, quantiles, want_quantiles)
        else:
            raise ValueError(f"Unknown TSFM: {name}")


    def run_cascade(history, horizon, quantiles, want_quantiles):
        # Local checkpoint: the parent resolved and pinned the directory, and
        # passes it through the environment the worker inherits.
        import importlib.util
        import os
        from pathlib import Path

        directory = os.environ.get("GNOMON_CASCADE_CHECKPOINT", "")
        if not directory:
            raise RuntimeError(
                "cascade needs a checkpoint directory in $GNOMON_CASCADE_CHECKPOINT")
        path = Path(directory).expanduser()
        spec = importlib.util.spec_from_file_location(
            "cascade_forecast_wrapper", path / "forecast_wrapper.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        wrapper = MODELS.get("cascade")
        if wrapper is None:
            wrapper = MODELS["cascade"] = module.Wrapper(str(path), "cpu")
        levels = [float(v) for v in wrapper.quantile_levels]
        out = wrapper.forecast_quantiles(list(history), horizon)
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        arr = out.reshape(horizon, len(levels))
        nearest = lambda level: min(
            range(len(levels)), key=lambda i: abs(levels[i] - level))
        point = [float(arr[step, nearest(0.5)]) for step in range(horizon)]
        if not want_quantiles:
            return {"point": point}
        return {"point": point, "quantiles": [
            {str(level): float(arr[step, nearest(level)]) for level in quantiles}
            for step in range(horizon)]}


    def run_chronos(name, history, horizon, quantiles, want_quantiles):
        import torch
        from chronos import BaseChronosPipeline

        model_id = {
            "chronos_bolt_mini": "amazon/chronos-bolt-mini",
            "chronos_bolt_small": "amazon/chronos-bolt-small",
        }[name]

        pipeline = BaseChronosPipeline.from_pretrained(
            model_id, revision=pinned(model_id), device_map="cpu",
            torch_dtype=torch.float32,
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


    def run_toto(name, history, horizon, quantiles, want_quantiles):
        import torch
        from toto2 import Toto2Model

        model_id = {
            "toto2_4m": "Datadog/Toto-2.0-4m",
            "toto2_22m": "Datadog/Toto-2.0-22m",
        }[name]
        model = MODELS.get(model_id)
        if model is None:
            model = Toto2Model.from_pretrained(
                model_id, revision=pinned(model_id),
            )
            model = model.to("cpu").eval()
            MODELS[model_id] = model

        patch_size = 32
        padding = (-len(history)) % patch_size
        target = torch.tensor([0.0] * padding + history, dtype=torch.float32)
        target = target.unsqueeze(0).unsqueeze(0)  # (batch, n_var, time)
        mask = torch.tensor(
            [False] * padding + [True] * len(history), dtype=torch.bool,
        ).unsqueeze(0).unsqueeze(0)
        ids = torch.zeros(1, 1, dtype=torch.long)

        q = model.forecast(
            {"target": target, "target_mask": mask, "series_ids": ids},
            horizon=horizon,
            has_missing_values=bool(padding),
        )
        # Preserve horizon as an axis for one-step forecasts.
        arr = q.detach().cpu().numpy().reshape(9, -1)
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

        # A branch name ("r1.1") can move under a fixed label; the parent's
        # pinned commit cannot.
        predictor = FlowStateForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-flowstate-r1",
            revision=pinned("ibm-granite/granite-timeseries-flowstate-r1"),
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
            revision=pinned("ibm-granite/granite-timeseries-ttm-r2"),
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

        module = Moirai2Module.from_pretrained(
            "Salesforce/moirai-2.0-R-small",
            revision=pinned("Salesforce/moirai-2.0-R-small"),
        )
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
            revision=pinned("AutonLab/MOMENT-1-small"),
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
            revision=pinned("AutonLab/MOMENT-1-small"),
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
            revision=pinned("AutonLab/MOMENT-1-small"),
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

    backend = "sandbox"
    revision: str | None = None

    def __init__(
        self,
        name: str,
        frequency: str = "h",
        timeout: int = 300,
    ):
        self.name = name
        pins = resolved_weights(name)
        self.revision = ",".join(
            f"{model_id}@{revision}"
            for model_id, revision in sorted(pins.items())) or None
        self.frequency = frequency
        self.timeout = timeout
        self._params_m = tsfm_parameter_count(name)
        self._supports_quantiles = tsfm_supports_quantiles(name)
        minimum = tsfm_capabilities(name).min_context_length
        self.min_history = minimum if minimum > 1 else None
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        # A timed-out model process is unhealthy for the remainder of this
        # run.  Opening a circuit avoids paying the full timeout again for
        # every fold/channel; a new command creates a fresh adapter and may
        # try the pinned model again.
        self._circuit_error: str | None = None

    @property
    def params_m(self) -> float:
        return self._params_m

    @property
    def supports_quantiles(self) -> bool:
        return self._supports_quantiles

    def _start_worker(self) -> subprocess.Popen[str]:
        """Start one long-lived worker for this adapter instance."""
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

        try:
            return subprocess.Popen(
                [str(venv_python), str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                # Model libraries can emit many warnings.  A persistent
                # worker must not retain an unread stderr pipe: once its OS
                # buffer fills, inference blocks forever.  Protocol errors
                # are returned as structured JSON on stdout, so discard
                # incidental library logging here.
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except FileNotFoundError as exc:
            raise TSFMError(f"Failed to start sandbox process: {exc}") from exc

    def close(self) -> None:
        """Release a persistent sandbox worker without affecting its venv."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=2)
        except Exception:
            process.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _run_subprocess(self, request: dict[str, Any]) -> dict[str, Any]:
        """Exchange one request with the persistent isolated worker."""
        if self._circuit_error is not None:
            raise TSFMError(self._circuit_error)
        request = dict(request)
        request["tsfm_name"] = self.name
        request["frequency"] = self.frequency
        # The worker refuses to load weights without these: the forecast id
        # records the pinned revisions (`resolved_weights`), so the load
        # must be at exactly those commits or the id would attest weights
        # the run never used.
        from .tsfm import resolved_weights

        request["revisions"] = resolved_weights(self.name)

        with self._process_lock:
            process = self._process
            if process is None or process.poll() is not None:
                process = self._start_worker()
                self._process = process
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise TSFMError(
                    f"Sandbox {self.name} worker stopped before request: {exc}") from exc

            ready, _, _ = select.select(
                [process.stdout.fileno()], [], [], self.timeout)
            if not ready:
                self.close()
                self._circuit_error = (
                    f"Sandbox for {self.name} timed out after {self.timeout}s; "
                    "adapter circuit is open for this run")
                raise TSFMError(
                    self._circuit_error)
            output = process.stdout.readline()
            if not output:
                self.close()
                raise TSFMError(
                    f"Sandbox {self.name} exited without a response")
            try:
                response = json.loads(output)
            except json.JSONDecodeError as exc:
                self.close()
                raise TSFMError(
                    f"Sandbox {self.name} produced invalid JSON output: {exc}") from exc

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

    def predict_many(
        self, histories: list[list[float]], horizon: int, season: int,
    ) -> list[list[float]]:
        """Forecast a fold batch in one isolated process/model load.

        This preserves sandbox dependency isolation while avoiding one weight
        load per rolling origin.  Evaluation still validates every returned
        trajectory independently before it may be scored.
        """
        if not histories:
            return []
        response = self._run_subprocess({
            "mode": "predict_batch", "histories": histories,
            "horizon": horizon, "season": season,
        })
        points = response.get("points")
        if not isinstance(points, list) or len(points) != len(histories):
            raise TSFMError(
                f"Sandbox {self.name} returned an invalid forecast batch")
        return [[float(value) for value in forecast] for forecast in points]

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
            # Multi-series forecasts evaluate channels concurrently.  Reuse
            # one worker per model/frequency in this process so those
            # evaluations share a single pinned weight load; the adapter's
            # request lock serialises access to its line protocol.
            key = (str(SANDBOX_ROOT.resolve()), name, frequency)
            with _ADAPTER_POOL_LOCK:
                adapter = _ADAPTER_POOL.get(key)
                if adapter is None:
                    adapter = SubprocessAdapter(name, frequency=frequency)
                    _ADAPTER_POOL[key] = adapter
            candidates.append(adapter)
            logger.info("Sandbox TSFM '%s' ready", name)
        except Exception:
            logger.debug("Failed to create SubprocessAdapter for %s", name, exc_info=True)
    return candidates


def select_tsfm_candidates(
    requested: list[str] | None = None,
    frequency: str = "h",
    in_process: "Callable[..., list[TSFMAdapter]] | None" = None,
) -> list[TSFMAdapter]:
    """Every requested adapter that can actually run, sandboxed where needed.

    A sandbox is dependency isolation, not a licence to compete. Choosing
    "sandboxes if any sandbox exists, else in-process" silently dropped every
    adapter that needs no isolation -- a locally trained checkpoint whose only
    dependency is torch would vanish from the candidate pool the moment an
    unrelated model was installed into a venv, and the run would look like a
    clean evaluation that the model simply lost.

    So the two sources are unioned: a sandbox wins for any name that has one,
    and the rest are loaded in-process. A name that can do neither is absent,
    which is what the capability notes already report.

    ``in_process`` is the loader for the non-sandboxed half. Callers pass
    their own module-level ``tsfm_candidates`` so that the seam they already
    substitute in tests stays the one this function uses.
    """
    ready = set(sandbox_available_tsfms())
    names = list(requested) if requested is not None else None
    sandboxed = sandbox_tsfm_candidates(
        requested=[n for n in names if n in ready] if names is not None else None,
        frequency=frequency,
    )
    covered = {adapter.name for adapter in sandboxed}
    if names is None:
        return sandboxed

    from .tsfm import dependency_missing
    from .tsfm import tsfm_candidates as _default_in_process

    # Filtered, not merely constructed: several adapters construct happily
    # without torch and only fail once a fold asks them to predict, which
    # turns a missing dependency into a hundred scored-as-failed folds and a
    # candidate that looks beaten rather than absent. The predicate is the
    # narrow one, so an adapter registered from outside the package is kept.
    remaining = [name for name in names
                 if name not in covered and not dependency_missing(name)]
    if not remaining:
        return sandboxed
    loader = in_process or _default_in_process
    return sandboxed + loader(requested=remaining, frequency=frequency)


def _close_adapter_pool() -> None:
    with _ADAPTER_POOL_LOCK:
        adapters = list(_ADAPTER_POOL.values())
        _ADAPTER_POOL.clear()
    for adapter in adapters:
        adapter.close()


atexit.register(_close_adapter_pool)
