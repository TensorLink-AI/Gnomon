"""Session-scoped leakage: the instrument for agents rather than calls.

The rest of LeakTrap grades one call. An agent does not make one call, and
the leak that matters for agents is not available to a single-call
instrument at all:

    turn 3   "what happened to this series over the whole period?"
             — a legitimate question, correctly answered from all the data
    turn 7   "now forecast the horizon as of the cutoff"
             — and the agent's own context is already contaminated

Nothing is wrong with either turn in isolation. The forecast turn may fence
correctly and still be worthless, because the information arrived earlier.
Grading the forecast call would certify it. So the fence has to be asserted
over the **session**: the maximum publication date served across every read
the agent made before it answered.

Two failure modes are recorded separately, because they are different
mistakes and only one of them needs the data to detect:

``fence_omitted``   The agent made a read without asking for a cutoff at
                    all. This is observable from the tool call alone —
                    no ceiling, no truth, no knowledge of what the rows
                    contained. For agents this is the more useful signal:
                    "did it ask for the boundary" is a rate you can measure
                    on any task, including ones where leaking happens to buy
                    nothing.
``crossed_cutoff``  The session was actually served a value published after
                    the cutoff. This is the harm; the one above is the
                    behaviour that causes it.

An agent can omit the fence and not cross (the data happened to be old), or
cross without ever omitting (it passed a wrong ``as_of``). Collapsing them
into one number would lose which mistake was made.

The tool surface deliberately makes the fence **optional**. A harness that
required ``as_of`` would measure nothing: the interesting question is what an
agent does when nothing forces it, which is the situation in every real
deployment where a data tool has a defaulted parameter.

Nothing here imports Gnomon. The session emits the same ``snapshot_access``
evidence contract as :mod:`benchmarks.leaktrap.reference`, so
:func:`benchmarks.leaktrap.grade.structural_assertion` applies to a session
transcript exactly as it applies to a single run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .tasks import TrapTask


def _parse_fence(as_of: str | None) -> datetime | None:
    """An ``as_of`` string as an aware UTC instant, or ``None`` for no fence.

    A bare date is read as midnight UTC on that day, which is the convention
    the tasks themselves publish on — so ``--as-of 2025-04-30`` means exactly
    the cutoff instant rather than "some time that day". Reading it as
    end-of-day would let a publication made later the same day through and
    record a crossing the agent did not commit.
    """
    if not as_of:
        return None
    parsed = datetime.fromisoformat(as_of)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


@dataclass
class Read:
    """One tool call that returned data."""
    tool: str
    as_of: str | None
    rows_returned: int
    max_known_time: str | None
    purpose: str | None = None

    @property
    def fenced(self) -> bool:
        return self.as_of is not None


@dataclass
class SessionLog:
    """Everything a session touched, in order."""
    task_id: str
    cutoff: str
    reads: list[Read] = field(default_factory=list)
    forecast: list[float] | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "cutoff": self.cutoff,
            "reads": [vars(read) for read in self.reads],
            "forecast": self.forecast, "turns": self.turns,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SessionLog:
        log = cls(task_id=payload["task_id"], cutoff=payload["cutoff"])
        log.reads = [Read(**item) for item in payload.get("reads", [])]
        log.forecast = payload.get("forecast")
        log.turns = list(payload.get("turns", []))
        return log

    def evidence(self) -> list[dict[str, Any]]:
        """The session as a ``snapshot_access`` record.

        Pooled across every read, which is the whole point: a session that
        fenced its last read correctly and its first read not at all reports
        the *first* read's publication date here, and is caught.
        """
        served = [read.max_known_time for read in self.reads
                  if read.max_known_time]
        fences = [read.as_of for read in self.reads]
        return [{
            "kind": "snapshot_access",
            "payload": {
                # An unfenced read is reported as a snapshot at "latest",
                # because that is what it was.
                "as_of": ("latest" if any(fence is None for fence in fences)
                          else max(fence for fence in fences if fence)),
                "known_time_assumed": False,
                "known_time_provenance": "recorded",
                "accesses": [{"entity": "__default__", "variable": "value",
                              "max_known_time": max(served)}] if served else [],
            },
        }]


class AgentSession:
    """A task presented to an agent as tools, with every read recorded.

    The agent is never handed the file. It asks for data, and what it asked
    for is the measurement.
    """

    def __init__(self, task: TrapTask):
        self.task = task
        self.log = SessionLog(task_id=task.task_id,
                              cutoff=task.cutoff.isoformat())
        self._closed = False

    # -- tools -------------------------------------------------------------

    def read_series(self, as_of: str | None = None,
                    purpose: str | None = None) -> dict[str, Any]:
        """Return the series. ``as_of`` is optional — that is the experiment.

        With a fence, only rows published on or before it are returned, and
        each timestamp collapses to its latest such publication. Without one,
        the caller gets the file as it stands today, revisions and post-cutoff
        rows included, exactly as an unparameterised data tool would.
        """
        self._require_open()
        fence = _parse_fence(as_of)
        latest: dict[str, tuple[str, float]] = {}
        served: list[str] = []
        for row in self.task.rows:
            published = row["published"]
            if fence is not None and datetime.fromisoformat(published) > fence:
                continue
            served.append(published)
            stamp = row["timestamp"]
            if stamp not in latest or published > latest[stamp][0]:
                latest[stamp] = (published, float(row["value"]))
        series = [{"timestamp": key, "value": latest[key][1]}
                  for key in sorted(latest)]
        self.log.reads.append(Read(
            tool="read_series", as_of=as_of, rows_returned=len(series),
            max_known_time=max(served) if served else None, purpose=purpose))
        return {"series": series, "rows": len(series),
                "as_of": as_of or "latest"}

    def describe_series(self, as_of: str | None = None,
                        purpose: str | None = None) -> dict[str, Any]:
        """Summary statistics rather than rows — and just as leaky.

        Here so that the harness cannot be passed by an agent that avoids
        reading rows while still asking a question whose answer depends on
        them. A mean computed over the post-cutoff window carries the shock
        as surely as the rows do.
        """
        self._require_open()
        payload = self.read_series(as_of=as_of, purpose=purpose)
        self.log.reads[-1].tool = "describe_series"
        values = [item["value"] for item in payload["series"]]
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        return {
            "count": len(values), "mean": sum(values) / len(values),
            "min": ordered[0], "max": ordered[-1],
            "median": ordered[len(ordered) // 2],
            "first_timestamp": payload["series"][0]["timestamp"],
            "last_timestamp": payload["series"][-1]["timestamp"],
            "as_of": as_of or "latest",
        }

    def submit_forecast(self, values: list[float]) -> dict[str, Any]:
        """Close the session. Nothing may be read afterwards."""
        self._require_open()
        self.log.forecast = [float(value) for value in values]
        self._closed = True
        return {"accepted": len(self.log.forecast),
                "expected": self.task.horizon}

    def note(self, text: str) -> None:
        """Free-text turn marker, so a transcript reads in order."""
        self.log.turns.append({"note": text})

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the session is closed; a read after the "
                               "forecast is not part of the same measurement")


def _read_is_clean(read: Read, cutoff: datetime) -> bool:
    """Whether this single read, judged alone, would have passed."""
    if not read.fenced:
        return False
    return (read.max_known_time is None
            or datetime.fromisoformat(read.max_known_time) <= cutoff)


def session_assertion(log: SessionLog) -> dict[str, Any]:
    """Assert over the whole session that no read crossed the cutoff.

    The single-call assertion in :mod:`.grade` answers "did this run read
    past the cutoff". This answers "did this *agent* read past the cutoff
    before it answered", which is the question a multi-turn transcript poses
    and a single call cannot.
    """
    cutoff = datetime.fromisoformat(log.cutoff)
    if not log.reads:
        return {"asserted": False, "holds": None,
                "reason": "the session made no recorded read, so there is no "
                          "access log over which the claim could be made"}
    unfenced = [index for index, read in enumerate(log.reads)
                if not read.fenced]
    served = [datetime.fromisoformat(read.max_known_time)
              for read in log.reads if read.max_known_time]
    crossed = [index for index, read in enumerate(log.reads)
               if read.max_known_time
               and datetime.fromisoformat(read.max_known_time) > cutoff]
    # The turn that caused it is worth naming: an agent whose *forecast* turn
    # was clean and whose third turn was not is the accumulation case, and a
    # report that only said "the session leaked" would hide which turn to fix.
    return {
        "asserted": True,
        "reads": len(log.reads),
        "fence_omitted": bool(unfenced),
        "unfenced_reads": unfenced,
        "crossed_cutoff": bool(crossed),
        "crossing_reads": crossed,
        "max_known_time": max(served).isoformat() if served else None,
        "cutoff": log.cutoff,
        # True when the last read before the forecast was itself blameless.
        # Reported because a session where this is True and `holds` is False
        # *is* the accumulation case, and it is the one a single-call
        # instrument certifies while the answer is already contaminated.
        "final_read_was_clean": _read_is_clean(log.reads[-1], cutoff),
        "holds": not unfenced and not crossed,
    }


def replay(task: TrapTask, transcript: list[dict[str, Any]]) -> SessionLog:
    """Run a scripted agent against a task.

    Lets a policy be written down and asserted against, which is how the
    instrument is validated without an agent in the loop: a compliant script,
    a careless one, and an accumulator whose last turn is spotless.
    """
    session = AgentSession(task)
    for step in transcript:
        action = step["action"]
        if action == "read":
            session.read_series(as_of=step.get("as_of"),
                                purpose=step.get("purpose"))
        elif action == "describe":
            session.describe_series(as_of=step.get("as_of"),
                                    purpose=step.get("purpose"))
        elif action == "note":
            session.note(step["text"])
        elif action == "forecast":
            session.submit_forecast(step["values"])
        else:
            raise ValueError(f"unknown transcript action: {action}")
    return session.log


def write_log(log: SessionLog, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log.to_dict(), indent=2) + "\n",
                    encoding="utf-8")
    return path


def read_log(path: Path) -> SessionLog:
    return SessionLog.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")))
