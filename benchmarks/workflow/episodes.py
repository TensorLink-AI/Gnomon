"""Agent-owned ordered commitments, separate from historical compiled stages."""

import json
from dataclasses import asdict, replace

from .matched import fingerprint
from .schema import Observation, Oracle


class Episode:
    def __init__(self, phases, checkpoint):
        if not isinstance(phases, (list, tuple)) or not 2 <= len(phases) <= 8 or not callable(checkpoint):
            raise ValueError("episode requires2..8 phases and durable checkpoint callback")
        self.phases = json.loads(json.dumps(phases, allow_nan=False))
        names = []
        for phase in self.phases:
            if (set(phase) != {"name", "revealed", "answer_schema"} or not isinstance(phase["name"], str)
                    or not 0 < len(phase["name"]) <= 64 or phase["name"] in names
                    or not isinstance(phase["revealed"], dict) or not isinstance(phase["answer_schema"], dict)):
                raise ValueError("invalid or oracle-bearing private episode phase")
            names.append(phase["name"])
        if self.phases[0]["revealed"]:
            raise ValueError("initial episode data belongs in public input")
        self.checkpoint = checkpoint
        self.records = []

    def commit(self, answer, usage):
        index = len(self.records)
        if index >= len(self.phases):
            raise ValueError("episode is already complete")
        record = {"schema_version": 1, "sequence": index, "phase": self.phases[index]["name"],
                  "answer": json.loads(json.dumps(answer, allow_nan=False)), "usage": usage,
                  "prior_sha256": self.records[-1]["record_sha256"] if index else None}
        record["record_sha256"] = fingerprint(record)
        self.checkpoint(record)  # Must commit before the next reveal is returned.
        self.records.append(record)
        return self.phases[index + 1] if index + 1 < len(self.phases) else None


def grade_episode(case, observation):
    """Grade only original ordered submissions; never fill numbers from tools."""
    from .scoring import _case_score
    records = observation.metadata.get("episode_checkpoints", [])
    phases, prior, valid = [], None, isinstance(records, list) and len(records) <= len(case.episode)
    if not valid:
        records = []
    for index, phase in enumerate(case.episode):
        item = records[index] if index < len(records) else None
        try:
            core = {key: value for key, value in item.items() if key != "record_sha256"}
            ordered = (set(core) == {"schema_version", "sequence", "phase", "answer", "usage", "prior_sha256"}
                       and item["schema_version"] == 1 and type(item["schema_version"]) is int
                       and item["sequence"] == index and type(item["sequence"]) is int
                       and item["phase"] == phase["name"] and item["prior_sha256"] == prior
                       and item["record_sha256"] == fingerprint(core))
            answer = Observation.from_dict({"case_id": case.id, **item["answer"]})
            prior = item["record_sha256"]
        except (AttributeError, KeyError, TypeError, ValueError):
            ordered = False
        valid = valid and ordered
        if valid:
            child = _case_score(replace(case, episode=(), stages=(), oracle=Oracle.from_dict(phase["oracle"])), answer)
            accuracy = child["correctness"] if (child["disclosures_pass"] and child["support_pass"]
                                                and child["trust_components"]["forbidden_claims"]) else 0.0
        else:
            accuracy = 0.0
        phases.append({"name": phase["name"], "committed": bool(valid), "correctness": accuracy,
                       "disclosures_pass": child["disclosures_pass"] if valid else None,
                       "forbidden_claims_pass": child["trust_components"]["forbidden_claims"] if valid else None})
    complete = bool(valid and len(records) == len(case.episode) and observation.status != "error")
    if complete:
        # Final output must still be that final committed agent answer.
        last = asdict(Observation.from_dict({"case_id": case.id, **records[-1]["answer"]}))
        delivered = asdict(observation)
        complete = all(last[key] == delivered[key] for key in ("status", "support", "numbers", "choices", "facts", "disclosures", "claims"))
    return {"complete": complete, "correctness": min([phase["correctness"] for phase in phases]) if complete else 0.0,
            "phases": phases, "basis": "ordered_agent_submissions_not_host_recovered_answers"}
