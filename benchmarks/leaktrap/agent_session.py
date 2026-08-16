"""Drive a leakage trap as an agent, one tool call at a time.

The session harness in :mod:`.session` can be scripted, which is how the
instrument is validated. This is the other half: a command line an *actual*
agent can use, one call per invocation, with the transcript accumulating on
disk between calls. It exists so that the subject of the measurement can be
something that decides — a model in a loop, a person, a harness under test —
rather than a policy written down in advance.

The fence is optional on every reading command. That is deliberate and it is
the entire experiment: a harness that required ``--as-of`` would measure
nothing except its own requirement. What is recorded is what the agent asked
for when nothing compelled it.

::

    python -m benchmarks.leaktrap.agent_session start --task 3 \
        --session /tmp/run.json
    python -m benchmarks.leaktrap.agent_session describe \
        --session /tmp/run.json                       # unfenced
    python -m benchmarks.leaktrap.agent_session read \
        --session /tmp/run.json --as-of 2025-04-30    # fenced
    python -m benchmarks.leaktrap.agent_session submit \
        --session /tmp/run.json --values 1,2,3
    python -m benchmarks.leaktrap.agent_session grade \
        --session /tmp/run.json

``start`` prints the brief and nothing else — notably not the data, and not
the cutoff's role. An agent that wants to know what it may read has to ask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.leaktrap.grade import (  # noqa: E402
    honest_candidates,
    leak_verdict,
    no_leak_ceiling,
)
from benchmarks.leaktrap.session import (  # noqa: E402
    AgentSession,
    read_log,
    session_assertion,
    write_log,
)
from benchmarks.leaktrap.tasks import generate_task  # noqa: E402


def _task(index: int, seed: int, family: str):
    return generate_task(index, seed, family=family)


def _load(path: Path) -> tuple[Any, AgentSession]:
    """Rebuild the session from disk between calls.

    The task is regenerated from the sidecar rather than stored, so the
    transcript on disk contains no data the agent has not asked for — a
    session file it could read would be a leak the harness handed it.
    """
    log = read_log(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    session = AgentSession(_task(meta["index"], meta["seed"], meta["family"]))
    session.log = log
    return session.task, session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="begin a session and print the brief")
    start.add_argument("--task", type=int, default=0)
    start.add_argument("--seed", type=int, default=7)
    start.add_argument("--family", default="classic")
    start.add_argument("--session", required=True)

    for name, helptext in (("read", "return the series"),
                           ("describe", "return summary statistics")):
        command = sub.add_parser(name, help=helptext)
        command.add_argument("--session", required=True)
        command.add_argument("--as-of", default=None,
                             help="optional publication-date fence")
        command.add_argument("--purpose", default=None,
                             help="why this read was made, for the transcript")

    submit = sub.add_parser("submit", help="submit the forecast and close")
    submit.add_argument("--session", required=True)
    submit.add_argument("--values", required=True,
                        help="comma-separated forecast values")

    grade = sub.add_parser("grade", help="score the closed session")
    grade.add_argument("--session", required=True)

    args = parser.parse_args(argv)
    path = Path(args.session)

    if args.command == "start":
        task = _task(args.task, args.seed, args.family)
        session = AgentSession(task)
        write_log(session.log, path)
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"index": args.task, "seed": args.seed, "family": args.family}) + "\n",
            encoding="utf-8")
        print(json.dumps({
            "task_id": task.task_id,
            "brief": (f"Forecast the {task.horizon} daily values following "
                      f"{task.cutoff.date().isoformat()}. You are standing at "
                      f"{task.cutoff.date().isoformat()}. Tools: read, "
                      f"describe, submit."),
            "horizon": task.horizon,
            "cutoff": task.cutoff.date().isoformat(),
            "tools": {
                "read": "read --session S [--as-of DATE] [--purpose TEXT]",
                "describe": "describe --session S [--as-of DATE] [--purpose TEXT]",
                "submit": "submit --session S --values v1,v2,...",
            },
        }, indent=2))
        return 0

    task, session = _load(path)

    if args.command in ("read", "describe"):
        tool = session.read_series if args.command == "read" else session.describe_series
        payload = tool(as_of=args.as_of, purpose=args.purpose)
        write_log(session.log, path)
        if args.command == "read":
            series = payload["series"]
            # Print compactly: an agent needs the numbers, not 134 lines.
            payload = {"rows": payload["rows"], "as_of": payload["as_of"],
                       "series": [f"{item['timestamp'][:10]}={item['value']}"
                                  for item in series]}
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "submit":
        values = [float(part) for part in args.values.split(",") if part.strip()]
        payload = session.submit_forecast(values)
        write_log(session.log, path)
        print(json.dumps(payload, indent=2))
        return 0

    log = session.log
    assertion = session_assertion(log)
    candidates = honest_candidates(task)
    ceiling = no_leak_ceiling(task, candidates)
    verdict = leak_verdict(task, log.forecast or [], ceiling, candidates)
    print(json.dumps({
        "task_id": log.task_id,
        "session_assertion": assertion,
        "score": verdict["score"],
        "no_leak_ceiling": verdict["no_leak_ceiling"],
        "leak_advantage": verdict["leak_advantage"],
        "temporal_leakage": verdict["leaked"],
        "flag_power": verdict["flag_power"],
        "transcribed_the_future": verdict["transcribed"],
        "reads": [vars(read) for read in log.reads],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
