"""Validate and report an evidence-backed delivery-progress document.

Delivery records are not user documentation and are not shipped in docs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path




def summarize(document: dict) -> dict:
    total = earned = 0
    streams = []
    identifiers = set()
    for stream in document["workstreams"]:
        name = stream["id"]
        if name in identifiers:
            raise ValueError(f"duplicate workstream: {name}")
        identifiers.add(name)
        weight = stream["weight"]
        checks = stream["checks"]
        if sum(check["weight"] for check in checks) != weight:
            raise ValueError(f"check weights do not equal workstream weight: {name}")
        subtotal = 0
        check_ids = set()
        for check in checks:
            if check["id"] in check_ids:
                raise ValueError(f"duplicate check: {name}.{check['id']}")
            check_ids.add(check["id"])
            if type(check["weight"]) is not int or check["weight"] <= 0:
                raise ValueError("check weights must be positive integers")
            if check["status"] not in {"pending", "in_progress", "verified", "blocked"}:
                raise ValueError(f"invalid check status: {name}.{check['id']}")
            if check["status"] == "verified":
                evidence = check.get("evidence")
                if not isinstance(evidence, list) or not evidence or any(
                    not isinstance(item, str) or not item.strip() for item in evidence
                ):
                    raise ValueError(f"verified check requires evidence: {name}.{check['id']}")
                subtotal += check["weight"]
        streams.append({"id": name, "earned": subtotal, "possible": weight})
        total += weight
        earned += subtotal
    if total != 100:
        raise ValueError(f"framework must total 100, got {total}")
    return {"earned": earned, "possible": total, "iteration": document["iteration"],
            "complete": earned == 100, "workstreams": streams}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Delivery-progress JSON document (kept outside the public docs tree)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = summarize(json.loads(args.path.read_text(encoding="utf-8")))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Verified production progress: {result['earned']}/100 (iteration {result['iteration']})")
        for stream in result["workstreams"]:
            print(f"  {stream['id']}: {stream['earned']}/{stream['possible']}")
    return 1 if args.require_complete and not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
