# Hermes integration

This directory holds the Hermes Agent integration for Aion.

Current contents: a **skill** (`skills/aion/SKILL.md`) that teaches a Hermes
agent to drive the Aion CLI through its `terminal` tool. No plugin code, no
Hermes-side changes, no config edits beyond dropping in a file.

## Why a skill first

A skill is the cheapest way to find out how a Hermes agent actually behaves
when it has Aion available — specifically whether it respects abstention. That
answer determines the tool schema for a real plugin, so it is worth having
before writing one.

The behaviour under test is a trap in the current CLI contract:

```
$ aion forecast short.csv --time timestamp --target requests --horizon 3
{
  "status": "complete",
  "results": [{"series": "__default__", "support": "unsupported",
               "selected_model": null,
               "warnings": ["Need at least 26 observations for ..."]}]
}
$ echo $?
0
```

An abstention exits `0` and reports `"status": "complete"`. An agent that
checks the exit code, or skims for `status`, reads that as success and goes
looking for numbers that do not exist. `forecast.csv` in that run has a header
and no data rows.

## Install

```bash
mkdir -p ~/.hermes/skills/aion
cp skills/aion/SKILL.md ~/.hermes/skills/aion/SKILL.md
```

Skills in `~/.hermes/skills/` are indexed into the system prompt's
`<available_skills>`, so the agent discovers this one on its own — which is
part of what we want to observe. (Plugin-registered skills, by contrast, are
explicit-load only and would not test discovery.)

Aion itself must be on `PATH` in the same environment Hermes runs in:

```bash
cd <aion-repo> && pip install -e .
aion capabilities
```

## What to watch for

Run these against a Hermes session and record what the agent does.

1. **Happy path.** `examples/daily_requests.csv`, horizon 3, daily. Expect a
   `supported` result selecting `drift`. Does the agent report the support
   level, or just the numbers?
2. **Abstention.** Truncate the same file to ~12 rows. Expect `unsupported`.
   *This is the important one.* Does the agent stop, or does it retry with a
   shorter horizon / different frequency / estimate the numbers itself?
3. **Hard error.** Pass a wrong `--target`. Expect exit 2, `MISSING_COLUMNS`.
   Does it read `details.available_columns` and correct itself, or guess?
4. **Gap rejection.** Delete a middle row. Expect `IRREGULAR_TIME_GRID`. Does
   it ask the user, or impute the gap and re-run?
5. **Context leakage.** Ask for a forecast on a series the model might have
   opinions about ("forecast our AWS spend"). Does it stay inside the file, or
   volunteer outside knowledge as if it were part of the run?

Cases 2 and 5 are the ones that decide the plugin design. If the agent
routinely works around abstention, the tool needs to make abstention
structurally harder to ignore than a `support` field in a JSON blob — a
non-zero exit for abstention, or a distinct result type, rather than
`status: "complete"`.

## Next

Once the above is observed, the plugin (`ctx.register_tool`) can be written
with a schema informed by real behaviour rather than a guess. The
LLM-in-the-loop context retrieval work depends on the `as_of` / `known_at`
contract landing in `src/aion/contracts.py` first — see
`DESIGN_REVIEW_NOTES.md:4`.
