"""Task-specific argument examples shared by CLI and MCP errors."""

from copy import deepcopy
from datetime import datetime


def _instant(value):
    try:
        return isinstance(value, str) and datetime.fromisoformat(value).utcoffset() is not None
    except (ValueError, TypeError):
        return False


def argument_recovery(name, arguments):
    arguments = arguments if isinstance(arguments, dict) else {}
    if name == "gnomon_ledger":
        templates = {
            "search": {"limit": 10}, "pending": {},
            "execution": {"execution_id": "EXECUTION_ID"},
            "study": {"study_id": "STUDY_ID"},
            "actuals_as_of": {"series_id": "SERIES_ID"},
            "evaluations": {"execution_id": "EXECUTION_ID"},
            "evaluate": {"execution_id": "EXECUTION_ID", "allow_partial": True},
            "compare": {"execution_ids": ["EXECUTION_ID"]},
            "compare_history": {"series_id": "SERIES_ID", "horizon": 2,
                                "providers": {"last_value": "REVISION", "historical_mean": "REVISION"},
                                "start": "2026-01-01T00:00:00Z", "end": "2026-01-31T00:00:00Z",
                                "source_as_of": "2026-02-01T00:00:00Z", "recorded_as_of": "2026-02-01T00:00:00Z"},
            "decision": {"decision_id": "DECISION_ID"},
            "append_actual": {"series_id": "SERIES_ID", "valid_time": "2026-01-21T00:00:00Z",
                              "value": 21, "source_available_at": "2026-02-01T00:00:00Z"},
            "record_decision": {"execution_ids": ["EXECUTION_ID"], "policy": {}, "inputs": {}, "action": {}},
            "append_decision_outcome": {"decision_id": "DECISION_ID", "outcome": {},
                                        "source_available_at": "2026-02-01T00:00:00Z"},
        }
        operation = arguments.get("operation")
        operation = operation if isinstance(operation, str) and operation in templates else "search"
        example = {"operation": operation, **deepcopy(templates[operation])}
        for key, value in list(example.items()):
            supplied = arguments.get(key)
            if type(supplied) is type(value) and supplied and (
                    key not in {"start", "end", "valid_time", "source_available_at", "source_as_of", "recorded_as_of"}
                    or _instant(supplied)):
                example[key] = deepcopy(supplied)
        for key in ("source_as_of", "recorded_as_of", "unit"):
            if isinstance(arguments.get(key), str) and (key == "unit" or _instant(arguments[key])):
                example[key] = arguments[key]
        if operation == "evaluate":
            example["allow_partial"] = True
            if isinstance(arguments.get("execution_ids"), list):
                example.pop("execution_id")
                example["execution_ids"] = arguments["execution_ids"]
        return {"example_arguments": example, "schema_command": "gnomon ledger --schema",
                "guidance": "Keep the intended operation. Replace example IDs and dates with your task values. "
                            "For evaluate, allow_partial=true returns available coverage; strict scoring requires every matching-unit actual at the chosen cutoffs."}
    if name == "gnomon_forecast":
        provider = arguments.get("provider")
        example = {"provider": provider if isinstance(provider, str) else "last_value",
                   "request": {"history": [1, 2, 3], "horizon": 2}}
        details = {"example_arguments": example, "schema_command": "gnomon infer --schema",
                   "guidance": "Python uses session.forecast(provider, request). MCP uses provider plus request, "
                               "or provider, data_ref and horizon. CLI --input maps to gnomon_inspect followed by gnomon_forecast."}
        if isinstance(arguments.get("input"), str):
            details["inspect_arguments"] = {"input": arguments["input"]}
            details["next_tool"] = "gnomon_inspect"
        return details
    return {}
