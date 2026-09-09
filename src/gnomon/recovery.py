"""Task-specific argument examples shared by CLI and MCP errors."""

from copy import deepcopy
import math


def _example_copy(value):
    """Keep error examples JSON-safe without inventing a finite observation."""
    if isinstance(value, float) and not math.isfinite(value):
        return "REPLACE_NONFINITE_VALUE"
    if isinstance(value, dict):
        return {key: _example_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_example_copy(item) for item in value]
    return deepcopy(value)


def _matches(value, schema):
    """Check the shared schema vocabulary before copying example fields."""
    if "const" in schema and value != schema['const']:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    kind = schema.get("type")
    if isinstance(kind, list):
        return any(_matches(value, {**schema, "type": item}) for item in kind)
    if kind == "null":
        return value is None
    if kind == "boolean":
        return type(value) is bool
    if kind == "number":
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if kind == "string":
        return isinstance(value, str) and schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", float("inf"))
    if kind == "integer":
        return type(value) is int and schema.get("minimum", float("-inf")) <= value <= schema.get("maximum", float("inf"))
    if kind == "array":
        return isinstance(value, list) and schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")) and all(
            _matches(item, schema["items"]) for item in value) and (
                not schema.get("uniqueItems") or all(item not in value[:i] for i, item in enumerate(value)))
    if kind == "object":
        if not isinstance(value, dict) or not schema.get("minProperties", 0) <= len(value) <= schema.get("maxProperties", float("inf")):
            return False
        fields, extra = schema.get("properties", {}), schema.get("additionalProperties", True)
        return set(schema.get("required", ())) <= value.keys() and all(
            _matches(item, fields[key]) if key in fields else
            _matches(item, extra) if isinstance(extra, dict) else extra
            for key, item in value.items())
    return True


def example_changes(arguments, example, prefix=""):
    """Describe every replacement, including removed fields and nested facts."""
    changed = []
    for key in sorted(set(arguments) | set(example)):
        path = prefix + key
        if key not in arguments or key not in example:
            changed.append(path)
        elif isinstance(arguments[key], dict) and isinstance(example[key], dict):
            changed.extend(example_changes(arguments[key], example[key], path + "."))
        elif type(arguments[key]) is not type(example[key]) or arguments[key] != example[key]:
            changed.append(path)
    return changed


def temporal_recovery(arguments):
    from .temporal_ops import TEMPORAL_SCHEMA, TEMPORAL_EXAMPLES, TemporalChoiceError, temporal_operation

    original = _example_copy(arguments)
    arguments = deepcopy(original)
    mechanical = []
    if 'timestamp' in arguments and 'value' not in arguments and arguments.get('operation') in ('normalize', 'shift'):
        arguments['value'] = arguments.pop('timestamp')
        mechanical.append({'from': 'timestamp', 'to': 'value', 'action': 'rename_field'})
    units = {'second': 'seconds', 'minute': 'minutes', 'hour': 'hours', 'day': 'days',
             'week': 'weeks', 'month': 'months', 'year': 'years'}
    if arguments.get('operation') == 'shift' and isinstance(arguments.get('unit'), str) and arguments['unit'] in units:
        arguments['unit'] = units[arguments['unit']]
        mechanical.append({'field': 'unit', 'action': 'pluralize_unit'})
    operation = arguments.get("operation")
    known = isinstance(operation, str) and operation in TEMPORAL_EXAMPLES
    operation = operation if known else "interval"
    schema = next(s for s in TEMPORAL_SCHEMA["oneOf"] if s["properties"]["operation"]["const"] == operation)
    example = deepcopy(TEMPORAL_EXAMPLES[operation])
    for key, field in schema["properties"].items():
        if key == "operation" or key not in arguments:
            continue
        supplied = arguments[key]
        if _matches(supplied, field):
            example[key] = deepcopy(supplied)
        elif field.get("type") == "integer" and isinstance(supplied, str) and supplied.lstrip("+-").isascii() and supplied.lstrip("+-").isdigit() and len(supplied) <= 16:
            # Propose an explicit type correction without changing the number.
            # The operation itself continues to require an integer.
            converted = int(supplied)
            example[key] = converted if _matches(converted, field) else supplied
            if _matches(converted, field):
                mechanical.append({'field': key, 'action': 'integer_string_to_integer'})
        elif field.get("type") == "object" and isinstance(supplied, dict):
            for child, value in supplied.items():
                example[key][child] = deepcopy(value)
        else:
            # An invalid supplied fact is still the caller's fact. Do not
            # replace it with a convenient, valid value from the schema.
            example[key] = deepcopy(supplied)
    kind = "parameter_preserving_example" if known else "schema_illustration"
    choices = {}
    if operation == "shift" and arguments.get("mode") not in ("calendar", "elapsed"):
        choices["mode"] = ["calendar", "elapsed"]
    runnable, unresolved = False, []
    for _ in range(5):
        try:
            temporal_operation(**example)
            runnable = True
            break
        except TemporalChoiceError as exc:
            choices[exc.field] = exc.choices
            example[exc.field] = exc.illustrative_value
        except (ValueError, TypeError) as exc:
            # Invalid facts such as clock gaps cannot be repaired by inventing
            # another date. Keep the task and provide a separate illustration.
            kind = "task_template" if known else "schema_illustration"
            unresolved.append({"message": str(exc), "action": "resolve_invalid_temporal_facts",
                               "guidance": "Resolve this constraint using the intended dates, local times and arithmetic semantics; the template retains those facts."})
            break
    changed = example_changes(original, example)
    details = {"example_arguments": example, "example_kind": kind,
               "example_runnable": runnable,
               "preserved_fields": [key for key in arguments if key in example and not example_changes({key: arguments[key]}, {key: example[key]})],
               "resolution_required": unresolved,
               "changed_fields": changed, "supported_operations": list(TEMPORAL_EXAMPLES),
               "schema_command": "gnomon temporal --schema",
               "guidance": "Example values for changed_fields are illustrative, not inferred task facts. "
                           "Confirm missing choices and correct invalid values before retrying."}
    details["supplied_arguments"] = original
    details['defaulted_input_options'] = {key: field['default'] for key, field in schema['properties'].items()
        if 'default' in field and key not in original}
    details['mechanical_corrections'] = mechanical
    proposal = deepcopy(example)
    for key, values in choices.items():
        proposal[key] = '<' + '|'.join(str(v) for v in values) + '>'
    missing = [k for k in schema.get('required', []) if k not in arguments and k not in choices]
    for key in missing:
        proposal[key] = '<supply_' + key + '>'
    # A supplied interval object may still omit its endpoints. Never promote
    # endpoints borrowed from the schema illustration into an exact retry.
    for key, field in schema['properties'].items():
        supplied = arguments.get(key)
        if field.get('type') == 'object' and isinstance(supplied, dict):
            for child in field.get('required', []):
                if child not in supplied:
                    path = key + '.' + child
                    missing.append(path)
                    proposal[key][child] = '<supply_' + path + '>'
    if missing:
        choices['missing_facts'] = missing
    unknown = sorted(set(arguments) - set(schema['properties']))
    if unknown:
        choices['unknown_fields'] = unknown
        for key in unknown:
            proposal[key] = arguments[key]
    if 'timestamp' in original and 'value' in original:
        choices['timestamp_or_value'] = ['supply one intended value; timestamp is not accepted']
        proposal['timestamp'] = original['timestamp']
    exact = known and runnable and not choices and not missing and not unresolved
    details['admissible'] = True if exact else False
    details['next_call'] = {'tool': 'gnomon_temporal', 'arguments': proposal,
                          'runnable': exact, 'admissible': exact}
    if kind == "task_template":
        details["schema_example_arguments"] = deepcopy(TEMPORAL_EXAMPLES[operation])
        details["guidance"] += " This task template still requires correction; schema_example_arguments is a separate runnable illustration."
    if choices:
        details["choices_required"] = choices
        details["guidance"] += " Example choices are illustrative: choose each listed policy explicitly before retrying."
        if "invalid_date" in choices:
            details["guidance"] += " For this invalid month-end target, clamp selects the last valid day; reject leaves the request rejected."
    if operation == "shift" and arguments.get("mode") not in ("calendar", "elapsed"):
        details["guidance"] += (" Choose mode explicitly: calendar applies local calendar arithmetic; elapsed applies "
                                "a duration to an offset-aware instant. The example uses calendar as an illustration.")
    return details


def frozen_recovery(arguments, rejected, *, tool='gnomon_inspect'):
    """Remove preparation flags only; never alter a frozen snapshot's facts."""
    corrected = {k: _example_copy(v) for k, v in arguments.items() if k not in rejected}
    return {'supplied_arguments': _example_copy(arguments), 'rejected_fields': sorted(rejected),
            'preserved_fields': list(corrected), 'changed_fields': sorted(rejected),
            'example_arguments': corrected, 'example_kind': 'task_correction',
            'example_runnable': True, 'admissible': True,
            'guidance': 'Remove the rejected preparation options. The saved snapshot already fixes these values; all forecasting parameters are preserved.',
            'next_call': {'tool': tool, 'arguments': corrected, 'runnable': True, 'admissible': True}}


def column_recovery(details, arguments):
    """Propose only an unambiguous known alias, preserving every other argument."""
    columns, missing = details.get('available_columns', []), details.get('missing_columns', [])
    if missing != ['timestamp'] or 'ts' not in columns or arguments.get('time_column', 'timestamp') != 'timestamp':
        return {'choices_required': {'column_mapping': {'available_columns': columns, 'missing_columns': missing}},
                'example_runnable': False}
    corrected = {**_example_copy(arguments), 'time_column': 'ts'}
    return {'example_arguments': corrected, 'supplied_arguments': _example_copy(arguments),
        'example_kind': 'task_correction', 'changed_fields': example_changes(arguments, corrected),
        'preserved_fields': [k for k in arguments if k != 'time_column'], 'example_runnable': True,
        'admissible': None, 'correction_scope': 'column_mapping_only_other_input_and_provider_checks_not_run',
        'guidance': 'The available ts column is the timestamp alias. Only time_column changed; all other task parameters are preserved.'}


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
            "compare": {"execution_ids": ["EXECUTION_ID_1", "EXECUTION_ID_2"]},
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
        from .session import ledger_schema
        variants = [v for v in ledger_schema(allow_outcome_writes=True)["oneOf"]
                    if v["properties"]["operation"]["const"] == operation]
        schema = next((v for v in variants if any(key in arguments and key in v["required"]
                      for key in ("actuals", "execution_ids"))), variants[0])
        # Field schema, rather than the example value's Python type/truthiness,
        # determines whether observations like zero and 2.5 survive recovery.
        for key, supplied in arguments.items():
            field = schema["properties"].get(key)
            if key != "operation" and field is not None and _matches(supplied, field):
                example[key] = _example_copy(supplied)
        if operation == "append_actual" and isinstance(arguments.get("actuals"), list):
            # An invalid batch remains a task template: never replace its rows
            # with one unrelated synthetic scalar observation.
            example = {"operation": operation, "actuals": _example_copy(arguments["actuals"])}
        if operation == "evaluate":
            example["allow_partial"] = True
            if isinstance(arguments.get("execution_ids"), list):
                example.pop("execution_id")
                example.setdefault("execution_ids", _example_copy(arguments["execution_ids"]))
        if operation == 'search' and type(arguments.get('limit')) is int:
            example['limit'] = max(1, min(100, arguments['limit']))
        if operation == 'compare_history' and arguments.get('series_id') == '__default__':
            example.pop('series_id', None)
        placeholders = []
        if operation == "compare":
            supplied = arguments.get("execution_ids", [])
            ids = list(dict.fromkeys(item for item in supplied if isinstance(item, str) and item)) if isinstance(supplied, list) else []
            while len(ids) < 2:
                placeholder = f"EXECUTION_ID_{len(ids) + 1}"
                while placeholder in ids:
                    placeholder += "_OTHER"
                placeholders.append(placeholder)
                ids.append(placeholder)
            example["execution_ids"] = ids
        details = {"example_arguments": example, "schema_command": "gnomon ledger --schema",
                "example_kind": "task_template", "changed_fields": example_changes(arguments, example),
                "guidance": "Keep the intended operation. Replace example IDs and dates with your task values. "
                            "Changed fields are illustrative, not inferred observations or cutoffs. "
                            "REPLACE_NONFINITE_VALUE marks an invalid numeric input; supply a real finite observation. "
                            "Correct the reported issue before executing this template; validation and stored identities still apply. "
                            "For evaluate, allow_partial=true returns available coverage; strict scoring requires every matching-unit actual at the chosen cutoffs."}
        if operation == "compare":
            details.update(required_execution_count=2, placeholder_execution_ids=placeholders,
                           discovery_arguments={"operation": "search", "limit": 10})
            details["guidance"] = ("Choose between two and 100 distinct recorded execution IDs with matched inputs, snapshot and forecast origin. "
                                   "Use ledger search to find IDs; replace placeholder_execution_ids before retrying. "
                                   "The template does not assert that selected executions are compatible.")
        if operation == 'compare_history' and arguments.get('series_id') == '__default__':
            details.update(rejected_fields=['series_id'], choices_required={'series_id': 'Select an explicit stable recorded series; __default__ is ineligible.'})
        return details
    if name == "gnomon_forecast":
        from .session import REQUEST_SCHEMA, FORECAST_SCHEMA
        from .forecast_adapter import ForecastRequest, ForecastAdapterError
        provider = arguments.get("provider")
        example = {"provider": provider if isinstance(provider, str) else "last_value",
                   "request": {"history": [1, 2, 3], "horizon": 2}}
        request = arguments.get("request")
        kind = "schema_illustration"
        if isinstance(request, dict):
            kind = "parameter_preserving_example"
            for key, value in request.items():
                field = REQUEST_SCHEMA["properties"].get(key)
                if field is not None and _matches(value, field):
                    example["request"][key] = deepcopy(value)
            if "history" not in request or not _matches(request["history"], REQUEST_SCHEMA["properties"]["history"]):
                # Do not substitute made-up observations for supplied invalid data.
                example["request"].pop("history")
                kind = "task_template"
            try:
                ForecastRequest.from_dict(example["request"])
            except (ForecastAdapterError, TypeError, ValueError, OverflowError):
                kind = "task_template"
        elif isinstance(arguments.get("data_ref"), str):
            example = {"provider": example["provider"], "data_ref": arguments["data_ref"], "horizon": 2}
            for key, field in FORECAST_SCHEMA["oneOf"][1]["properties"].items():
                if key in arguments and _matches(arguments[key], field):
                    example[key] = deepcopy(arguments[key])
            kind = "task_template"
        if type(arguments.get("use_cache")) is bool:
            example["use_cache"] = arguments["use_cache"]
        details = {"example_arguments": example, "schema_command": "gnomon infer --schema",
                   "example_kind": kind, "changed_fields": example_changes(arguments, example),
                   "guidance": "Python uses session.forecast(provider, request). MCP uses provider plus request, "
                               "or provider, data_ref and horizon. CLI --input maps to gnomon_inspect followed by gnomon_forecast."}
        details["guidance"] += (" Changed fields are illustrative; correct only the reported issue while retaining valid task parameters. "
                                "Supply real history observations if history is absent. Task templates may still require correction; "
                                "no provider or snapshot availability has been verified by this example.")
        if isinstance(arguments.get("input"), str):
            details["inspect_arguments"] = {"input": arguments["input"]}
            details["next_tool"] = "gnomon_inspect"
        return details
    return {}
