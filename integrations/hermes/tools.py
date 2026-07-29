"""Tool handlers wrapping the Aion CLI as a subprocess.

The CLI's JSON contract is the integration API: success payloads on stdout
are passed through verbatim, and Aion's structured errors on stderr are
passed through verbatim. This layer adds nothing numerical — it only wraps
transport failures (CLI missing, timeout, protocol breakage) in the same
error envelope so the agent always receives one shape.

Handlers follow the Hermes plugin contract: ``(args, **kwargs) -> str``,
always a JSON string, never an exception.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any

SCHEMA_VERSION = "0.1"

# Override with e.g. AION_PLUGIN_CLI="/path/to/python -m aion" for a
# non-PATH install; AION_PLUGIN_TIMEOUT caps any single run in seconds.
_CLI_ENV = "AION_PLUGIN_CLI"
_TIMEOUT_ENV = "AION_PLUGIN_TIMEOUT"
_DEFAULT_TIMEOUT = 300.0

_INSTALL_HINT = (
    "Install Aion first: `uv tool install aion-forecast` or `bash install.sh` "
    "from a checkout, then verify with `aion capabilities`. If Aion is "
    "installed outside PATH, set AION_PLUGIN_CLI to the full command."
)


def _cli_argv() -> list[str]:
    return shlex.split(os.environ.get(_CLI_ENV, "") or "aion")


def _timeout() -> float:
    try:
        return float(os.environ[_TIMEOUT_ENV])
    except (KeyError, ValueError):
        return _DEFAULT_TIMEOUT


def _error(code: str, message: str, *, retryable: bool = False,
           details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": details or {},
        },
    }


def _run_aion(cli_args: list[str]) -> dict[str, Any]:
    """Run the CLI and return a payload dict, wrapping transport failures."""
    argv = _cli_argv() + cli_args
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=_timeout(),
        )
    except FileNotFoundError:
        return _error(
            "AION_NOT_INSTALLED",
            f"The Aion CLI was not found (tried {argv[0]!r}). {_INSTALL_HINT}",
        )
    except subprocess.TimeoutExpired:
        return _error(
            "AION_TIMEOUT",
            f"Aion did not finish within {_timeout():.0f}s. Retry with a "
            "smaller dataset or horizon, or raise AION_PLUGIN_TIMEOUT.",
            retryable=True,
        )
    except OSError as exc:
        return _error("AION_EXECUTION_FAILED", f"Could not run Aion: {exc}")

    # Success JSON arrives on stdout; Aion's structured errors on stderr.
    stream = completed.stdout if completed.returncode == 0 else completed.stderr
    try:
        return json.loads(stream)
    except (json.JSONDecodeError, TypeError):
        return _error(
            "AION_PROTOCOL_ERROR",
            "Aion returned output that is not valid JSON; the installed "
            "version may be incompatible with this plugin.",
            details={
                "exit_code": completed.returncode,
                "stdout_tail": (completed.stdout or "")[-500:],
                "stderr_tail": (completed.stderr or "")[-500:],
            },
        )


def _dataset_args(args: dict[str, Any]) -> list[str] | dict[str, Any]:
    """Translate shared tool arguments to CLI flags, or return an error payload."""
    missing = [key for key in ("input", "time_column", "target_column") if not args.get(key)]
    if missing:
        return _error(
            "INVALID_ARGUMENTS",
            f"Missing required argument(s): {', '.join(missing)}.",
        )
    cli = [str(args["input"]), "--time", str(args["time_column"]),
           "--target", str(args["target_column"])]
    if args.get("series_column"):
        cli += ["--series", str(args["series_column"])]
    if args.get("frequency"):
        cli += ["--frequency", str(args["frequency"])]
    return cli


def check_aion_available() -> bool:
    """Version handshake used as the Hermes ``check_fn``.

    True only when the CLI runs and speaks the schema version this plugin
    was written against — an incompatible install should hide the tools
    rather than fail mid-conversation.
    """
    payload = _run_aion(["capabilities"])
    return (
        payload.get("status") != "error"
        and payload.get("schema_version") == SCHEMA_VERSION
    )


def handle_aion_capabilities(args: dict[str, Any], **kwargs: Any) -> str:
    return json.dumps(_run_aion(["capabilities"]), allow_nan=False)


def handle_aion_inspect(args: dict[str, Any], **kwargs: Any) -> str:
    cli = _dataset_args(args)
    if isinstance(cli, dict):
        return json.dumps(cli)
    return json.dumps(_run_aion(["inspect", *cli]), allow_nan=False)


def handle_aion_forecast(args: dict[str, Any], **kwargs: Any) -> str:
    cli = _dataset_args(args)
    if isinstance(cli, dict):
        return json.dumps(cli)
    if not args.get("horizon"):
        return json.dumps(_error("INVALID_ARGUMENTS", "Missing required argument: horizon."))
    cli = ["forecast", *cli, "--horizon", str(int(args["horizon"]))]
    if args.get("output_dir"):
        cli += ["--output", str(args["output_dir"])]
    if args.get("minimum_baseline_improvement") is not None:
        cli += ["--minimum-baseline-improvement", str(args["minimum_baseline_improvement"])]
    if args.get("context_events_file"):
        cli += ["--context", str(args["context_events_file"])]
    if args.get("threshold") is not None:
        cli += ["--threshold", str(float(args["threshold"]))]
    if args.get("project"):
        cli += ["--project", str(args["project"])]
    return json.dumps(_run_aion(cli), allow_nan=False)


def handle_aion_submit_actuals(args: dict[str, Any], **kwargs: Any) -> str:
    return json.dumps(_run_aion([
        "track", "actuals", "--project", str(args.get("project", "")),
        "--file", str(args.get("actuals_file", "")),
    ]), allow_nan=False)


def handle_aion_list_open_forecasts(args: dict[str, Any], **kwargs: Any) -> str:
    cli = ["track", "due"]
    if args.get("project"):
        cli += ["--project", str(args["project"])]
    return json.dumps(_run_aion(cli), allow_nan=False)


def handle_aion_model_performance(args: dict[str, Any], **kwargs: Any) -> str:
    cli = ["track", "performance", "--project", str(args.get("project", ""))]
    if args.get("model"):
        cli += ["--model", str(args["model"])]
    return json.dumps(_run_aion(cli), allow_nan=False)


def handle_aion_record_decision(args: dict[str, Any], **kwargs: Any) -> str:
    return json.dumps(_run_aion([
        "track", "decision", "record",
        "--decision-id", str(args.get("decision_id", "")),
        "--project", str(args.get("project", "")),
        "--forecast-id", str(args.get("forecast_id", "")),
        "--action", str(args.get("action", "")),
        "--expected-outcome", str(args.get("expected_outcome", "")),
    ]), allow_nan=False)


def handle_aion_resolve_decision(args: dict[str, Any], **kwargs: Any) -> str:
    return json.dumps(_run_aion([
        "track", "decision", "resolve",
        "--decision-id", str(args.get("decision_id", "")),
        "--actual-outcome", str(args.get("actual_outcome", "")),
        "--correct", "true" if args.get("correct") else "false",
    ]), allow_nan=False)


def make_propose_context_handler(ctx: Any):
    """Build the context-investigation handler bound to the host's ``ctx.llm``.

    The prompt and the validation both come from the Aion CLI — the plugin
    only carries the model call: `aion context prompt` → host LLM →
    `aion context validate`. The model's output never reaches a forecast
    directly; only events surviving deterministic validation are written,
    and admission into numbers still requires the ablation gate at
    forecast time.
    """
    from .llm_adapter import HermesLLMAdapter

    def handle_aion_propose_context_events(args: dict[str, Any], **kwargs: Any) -> str:
        files = args.get("files") or []
        if not files:
            return json.dumps(_error("INVALID_ARGUMENTS", "Missing required argument: files."))
        file_args: list[str] = []
        for file in files:
            file_args += ["--file", str(file)]
        series_args: list[str] = []
        for series in args.get("series_names") or []:
            series_args += ["--series", str(series)]

        prompt_payload = _run_aion(["context", "prompt", *file_args, *series_args])
        if prompt_payload.get("status") == "error":
            return json.dumps(prompt_payload, allow_nan=False)

        adapter = HermesLLMAdapter(ctx, purpose="aion context investigation")
        try:
            raw_response = adapter.complete(
                prompt_payload["instructions"], prompt_payload["response_schema"]
            )
        except Exception as exc:
            return json.dumps(_error(
                "AION_LLM_FAILED",
                f"The host LLM call failed: {exc}. The forecast can still run "
                "without context events.",
                retryable=True,
            ))

        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(raw_response, handle)
            response_path = handle.name
        try:
            validated = _run_aion(
                ["context", "validate", "--response", response_path, *file_args]
            )
        finally:
            os.unlink(response_path)
        if validated.get("status") == "error":
            return json.dumps(validated, allow_nan=False)

        output_file = str(args.get("output_file") or "aion-context-events.json")
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(validated, handle, indent=2)
        validated["events_file"] = output_file
        validated["next_step"] = (
            "Pass events_file as context_events_file to aion_forecast. Events "
            "are proposals: Aion admits them into the forecast only if they "
            "demonstrate stable improvement on identical backtest folds."
        )
        return json.dumps(validated, allow_nan=False)

    return handle_aion_propose_context_events
