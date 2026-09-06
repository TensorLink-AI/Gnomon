"""Legacy schema and horizon argument resolution, independent of the tool registry."""

from __future__ import annotations


from typing import Any



def _resolve_schema_arguments(
    arguments: dict[str, Any], tool_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Fill ``time_column``/``target_column`` from the file, or say why not.

    The CLI learned this in v0.4 because a bare missing-argument failure on
    an obvious two-column file was the most common first-run failure there
    is; the tool surface had kept the old behaviour. Same rules as the CLI:
    strict inference (exactly one column qualifies), every inference
    disclosed as an assumption, ambiguity refused with the candidates named
    — in tool-parameter vocabulary, not CLI flags."""
    if arguments.get("time_column") and arguments.get("target_column"):
        return arguments, []
    from .contracts import GnomonError

    source = str(arguments.get("input") or "")
    missing = [name for name in ("time_column", "target_column")
               if not arguments.get(name)]
    if source.startswith("store:"):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "time_column and target_column are required for store:<dataset> "
            "inputs; a stored dataset has no file header to infer from.",
            {"input": source, "missing_parameters": missing},
            repair_options=[{
                "action": "supply_arguments",
                "description": "Pass time_column and target_column "
                               "explicitly; gnomon_list_datasets shows each "
                               "dataset's variables.",
                "arguments": missing,
            }],
        )
    from .data import infer_schema_columns

    inferred = infer_schema_columns(source)
    resolved = dict(arguments)
    assumptions: list[str] = []
    for parameter, key, candidates_key in (
        ("time_column", "time", "time_candidates"),
        ("target_column", "target", "target_candidates"),
    ):
        if resolved.get(parameter):
            continue
        chosen = inferred[key]
        if chosen is None:
            candidates = list(inferred[candidates_key])
            if parameter == "target_column" and candidates \
                    and tool_name in {"gnomon_inspect", "gnomon_describe"}:
                # Inspection is read-only and cheap, and a caller who
                # names no target on a wide file is asking about the
                # file: inspect every qualifying column rather than
                # refuse. Not a guess — nothing is chosen over anything
                # else, and the expansion is disclosed. Forecasting
                # still refuses loudly: acting on one guessed column
                # (or paying for all of them) is a real choice.
                resolved[parameter] = ",".join(candidates)
                assumptions.append(
                    f"target_column was not supplied and "
                    f"{len(candidates)} columns qualify "
                    f"({', '.join(candidates)}); all of them were "
                    f"inspected. Pass target_column to narrow."
                )
                continue
            repairs: list[dict[str, Any]] = [{
                "action": "supply_arguments",
                "description": f"Pass {parameter} explicitly."
                               + (f" Candidates: {', '.join(candidates)}."
                                  if candidates else ""),
                "arguments": [parameter],
            }]
            if candidates:
                repairs = [
                    {
                        "action": "supply_arguments",
                        "description": f"Retry with {parameter}={candidate!r}.",
                        "tool_call": {
                            "name": tool_name,
                            "arguments": {**resolved, parameter: candidate},
                        },
                    }
                    for candidate in candidates
                ]
            if parameter == "target_column" and tool_name == "gnomon_forecast":
                repairs.append({
                    "action": "forecast_all_candidates",
                    "description": "Or batch every numeric column in one "
                                   "run: pass target_column \"auto\".",
                    "tool_call": {
                        "name": tool_name,
                        "arguments": {**resolved, parameter: "auto"},
                    },
                })
            raise GnomonError(
                "AMBIGUOUS_SCHEMA",
                f"{parameter} was not supplied and cannot be inferred: "
                + (f"{len(candidates)} columns qualify "
                   f"({', '.join(candidates)})."
                   if candidates else "no column qualifies.")
                + f" Pass {parameter} explicitly; every other parameter is "
                  f"inferred from the file.",
                {"parameter": parameter, "candidates": candidates,
                 "columns_examined": list(inferred["columns"])},
                repair_options=repairs,
            )
        resolved[parameter] = chosen
        assumptions.append(
            f"{parameter} was not supplied; inferred as {chosen!r}, the only "
            f"column in the input that qualifies."
        )
    return resolved, assumptions


def _default_forecast_horizon(arguments: dict[str, Any]) -> int:
    """One seasonal period of the input's own grid — the CLI's default."""
    from .pipeline import load_stage
    from .temporal import detect_season

    target_spec = str(arguments["target_column"])
    if "," in target_spec or target_spec.strip().lower() == "auto":
        # Borrow one concrete column so the loader sees a real name; the
        # grid and season are shared across a wide file's channels.
        from .data import resolve_target_spec

        target_spec = resolve_target_spec(
            str(arguments["input"]), target_spec,
            time_column=arguments.get("time_column"),
            series_column=arguments.get("series_column"),
        )[0]
    loaded = load_stage(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=target_spec,
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
        repair=str(arguments.get("repair", "safe")),
        regrid=arguments.get("regrid"),
    )
    longest = max(loaded.groups.values(), key=len, default=[])
    season, _, _ = detect_season([item.value for item in longest], loaded.frequency)
    return max(1, int(season))


def _parse_as_of(raw: Any):
    if not raw:
        return None
    from .data import _parse_timestamp
    return _parse_timestamp(str(raw), 0)
