"""Pinned Workflow agent driver. No live model requests without explicit opt-in.

Backend factories are installed/operator Python code, never agent-selected imports.
The ordinary backend must expose real software; this driver does not synthesize a
weak baseline, score oracle or library-specific Gnomon adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib import parse, request

if __package__ in (None, ""):
    # The experiment invokes this exact checkout script, including from a
    # disposable working directory; never resolve a stale installed Gnomon wheel.
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root)]

from benchmarks.common.openrouter import OpenRouterClient
from benchmarks.workflow.bounded_agent import run_agent, WIRE_BYTES
from benchmarks.workflow.matched import ARMS, _keys
from benchmarks.workflow.accounting import AttemptJournal, reported_cost_limit
from benchmarks.workflow.agent_metrics import _decode_record


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def pinned_file(path, expected):
    with Path(path).open("rb") as stream:
        data = stream.read(WIRE_BYTES + 1)
    if len(data) > WIRE_BYTES or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("driver input differs from pinned content or exceeds byte limit")
    return data.decode("utf-8")


def endpoint(value, *, allow_model_requests):
    if not isinstance(value, str) or any(ord(char) < 33 for char in value):
        raise ValueError("invalid model endpoint")
    parsed = parse.urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("model endpoint must omit credentials, query and fragment")
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = False  # Numeric loopback only: do not trust an ambient DNS mapping.
    if not local and (parsed.scheme != "https" or not allow_model_requests):
        raise ValueError("remote model requests require HTTPS and --allow-model-requests")
    return value.rstrip("/")


def run(case, *, allow_model_requests=False):
    context = case["experiment"]
    _keys(context, {"experiment_id", "arm", "evidence_kind", "common", "surface"}, "driver context")
    common, arm = context["common"], context["arm"]
    if arm not in ARMS or context["evidence_kind"] not in {"scripted", "agent"}:
        raise ValueError("invalid matched arm/evidence kind")
    config = _decode_record(pinned_file(common["provider_config_file"], common["provider_config_sha256"]))
    prompt = pinned_file(common["prompt_file"], common["prompt_sha256"])
    _keys(config, {"schema_version", "llm", "backends"}, "driver provider configuration")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("unsupported driver configuration schema")
    _keys(config["llm"], {"base_url", "token_env"}, "LLM configuration")
    _keys(config["backends"], ARMS, "backend configurations")
    for item in config["backends"].values():
        _keys(item, {"factory", "options"}, "backend configuration")
        if (not isinstance(item["factory"], str)
                or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", item["factory"])
                or not isinstance(item["options"], dict)):
            raise ValueError("backend requires an installed module:factory and options object")
    url = endpoint(config["llm"]["base_url"], allow_model_requests=allow_model_requests)
    limit = reported_cost_limit(common["budget"])
    if ("_spending_allowance_usd" in case) != (limit is not None):
        raise ValueError("reported-cost policy requires a runner allowance binding")
    budget = dict(common["budget"])
    if limit is not None:
        allowance = case.get("_spending_allowance_usd")
        reported_cost_limit({"max_reported_cost_usd": allowance})
        if allowance > limit:
            raise ValueError("remaining reported-cost allowance exceeds pinned arm limit")
        budget["max_reported_cost_usd"] = allowance
    token_env = config["llm"]["token_env"]
    if not isinstance(token_env, str) or not re.fullmatch(r"[A-Za-z_]\w*", token_env):
        raise ValueError("token_env must name an environment variable")
    token = os.environ.get(token_env)
    if not token or any(ord(char) < 33 or ord(char) > 126 for char in token):
        raise ValueError("named model credential is absent or invalid")
    # Bind the request to the experiment's explicit credentials and endpoint.
    # No proxy environment or redirects: credentials go only to the explicit origin.
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    def client_factory():
        return OpenRouterClient(common["model"]["id"], api_key=token, base_url=url,
                                timeout=common["budget"]["timeout_seconds"],
                                request_opener=opener)
    private_episode = case.get("_private_episode")
    journal_spec = case.get("_episode_journal")
    if bool(private_episode) != bool(journal_spec):
        raise ValueError("private episode requires a durable journal binding")
    public = {key: value for key, value in case.items() if key not in {"experiment", "_private_episode", "_episode_journal", "_spending_allowance_usd"}}
    selected = config["backends"][arm]
    module, name = selected["factory"].split(":")
    # Only public case data reaches the model/backend. Do not expose provider file
    # paths, model credentials, oracle, or configurations for the other arms.
    with tempfile.TemporaryDirectory(prefix="gnomon-agent-") as workspace:
        def backend_factory():
            factory = getattr(importlib.import_module(module), name)
            return factory(case=json.loads(json.dumps(public)), options=selected["options"],
                           workspace=Path(workspace), timeout=common["budget"]["timeout_seconds"])
        journal = AttemptJournal(Path(journal_spec["path"])) if journal_spec else None
        try:
            result = run_agent(public, prompt=prompt + "\n" + context["surface"]["guidance"],
                               budget=budget, generation=common["generation"],
                               client_factory=client_factory, backend_factory=backend_factory,
                               episode=private_episode, checkpoint=(lambda record:
                                   journal.checkpoint(journal_spec["attempt_id"], public["id"], record)) if journal else None)
        finally:
            if journal:
                journal.close()
    return replace(result, metadata={**result.metadata, "experiment_id": context["experiment_id"],
                   "arm": arm, "evidence_kind": context["evidence_kind"],
                   "model": common["model"], "model_revision_basis": "operator_declared_not_attested",
                   "provider_config_sha256": common["provider_config_sha256"],
                   "backend_factory": selected["factory"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-model-requests", action="store_true",
                        help="Allow the explicitly configured remote model endpoint; not a dollar cap")
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(WIRE_BYTES + 1)
        if len(raw) > WIRE_BYTES:
            raise ValueError("driver input exceeds byte limit")
        result = run(_decode_record(raw.decode("utf-8")), allow_model_requests=args.allow_model_requests)
        print(json.dumps(asdict(result), allow_nan=False))
        return 0
    except Exception as error:
        # The runner journals this failed attempt with unknown usage. Do not echo
        # arbitrary exception text which might include a credential or case data.
        print(f"agent driver failed: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
