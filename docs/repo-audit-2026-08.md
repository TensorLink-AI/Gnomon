# Repository audit — does Gnomon realize the intended product?

Status: dated record, 2026-08-20. Independent repository-wide audit of
`main` at `038291e` ("Add evidence-weighted temporal model admission").
Method: treat every README/docs claim as a hypothesis; verify against
code, live execution, tests, git history, and committed benchmark
artifacts; cross-reference against the TMLR survey
[*A Survey of Reasoning and Agentic Systems in Time Series with LLMs*](https://arxiv.org/abs/2509.11575)
(repo `Time-Series-Reasoning-Survey` @ `fcc7990`). No code was modified;
all runs used a scratch workspace. Line numbers refer to `038291e`.

## Executive verdict

**The engineering substantially realizes the intended product; the
empirical marketing partially does not.** The trust guarantees that
define Gnomon — structural leakage safety, mandated baselines,
identity-carrying publication, five-state support with typed abstention,
disclosed repair, deterministic verification, compact MCP surface — are
implemented in code at the cited locations, hold up under live
adversarial testing, and are honest to a degree that is rare (the
runtime discloses its own statistical shortcuts in-band). An agent
wired to the MCP server today gets deterministic numbers it cannot
edit, with evidence attached.

The two most prominent *quantitative* claims on the README, however, do
not meet the repository's own "measured or absent" claims discipline
(`docs/product-position.md:65`):

1. The flagship surface-experiment numbers (**1,800 executions, 96.3%
   correctness, 89% trust, 12.7K tokens, one median call**) have **no
   backing artifact anywhere in the tree or git history**, while the
   *rejected* predecessor experiment got a full committed write-up.
2. The LeakTrap headline (**"Gnomon 0 of 40, McNemar p = 0.00024"**) is
   **tautological on the Gnomon side**: the no-leak ceiling is computed
   over Gnomon's own model set, so Gnomon's leak flag cannot fire by
   construction (max `leak_advantage` = 0.0 across all 40 committed
   rows). The project itself discovered this — commit `44cd2d4` on
   unmerged branch `claude/leak-trap-benchmark-v61t70` calls the
   published framing "arithmetic compared against a random variable" —
   but main still cites the number without disclosure. The control-side
   findings (13/35 answered tasks leaked, 4 verbatim future
   transcriptions; p recomputed to 0.000244 from raw artifacts) remain
   genuinely empirical, as does the structural access-log guarantee.

Secondary conclusions: the product surface is coherent and genuinely
usable by an agent (verified end-to-end over stdio), with a short list
of real defects (below); the 24-day history shows unusual measurement
discipline mid-arc but re-expansion without specs in the final 48
hours; versioning/release state is incoherent (0.5.0 in metadata, never
tagged, 1,020-line "Unreleased" changelog, no PyPI); and the design is
well aimed at a gap the research literature actually documents, though
its external benchmark validation is thin and, where committed, shows
Gnomon *losing* on raw accuracy while winning on trust properties —
a trade-off the docs state honestly.

---

## 1. Core engine: claims vs code

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Structural leakage safety (snapshot cannot serve post-cutoff rows) | **Verified** | `temporal_store.py:99–125` filters at construction and on every read, clamps caller cutoffs to `min(cutoff, as_of)`, logs every access. Plain CSVs are wrapped in the same snapshot (`pipeline.py:170–178`). Fold training sees only pre-origin data (`evaluation.py:831–832`); context events and covariates are knowledge-time gated per fold (`context_eval.py:247–251`, `covariates.py:49–95`). Behaviorally confirmed: `--as-of` mid-history returns `series_end == as_of` and excludes later-published revisions. |
| Rolling selection / separate calibration / untouched test | **Verified with one disclosed deviation** | `origins[:-2]` select, `origins[-2]` calibrates, `origins[-1]` reports only (`evaluation.py:899–907`); winner fixed before calibration; test fold scores only the already-chosen pair. Deviation: default `pool_residuals=True` mixes selection-fold residuals into interval calibration (`evaluation.py:1692–1746`) — admitted in-band as a typed disclosure, but the README's pipeline diagram implies strict separation. |
| "The evaluated executable publishes" | **Verified (one legacy hole)** | `CandidateSpec` binds the fold-time closures; publication fits that spec (`evaluation.py:1807–2017`, `candidate.py`, `pipeline.py:474–517`); output carries `execution_identity` with `publish_matches_evaluated: true` (observed live). A name-based TSFM rediscovery path survives for spec-less legacy assessments only (`pipeline.py:562–581`). |
| Deterministic verifier "on every response" | **Partially true** | Real and deterministic (`verifier.py:61–221`): probability calibration bounds, unlabelled sub-supported rows, `known_time > as_of`, dangling refs. Gates all six claim-bearing verbs on all three surfaces (`runtime.py:1031,1372`; `macros.py:434,658,857,1006`). But descriptive surfaces (`gnomon_describe`, `gnomon_inspect`, `gnomon_status`) never invoke it (`toolspec.py:1077–1186`), and the "causal claims from associational evidence" check is a blanket ban — `CAUSAL_CAPABLE_KINDS = frozenset()` (`verifier.py:39`) — dressed as a discriminating check. |
| Five-state support, unstrippable per-row tier, typed abstention | **Verified** | `contracts.py:19–21`; per-row tiers stamped incl. split forecasts (`runtime.py:640–654`); brief renderings preserve tier (`toolspec.py:758–767`); verifier rejects unlabelled sub-supported quotes (`verifier.py:73–115`); abstention returns `max_supportable_horizon` and typed recovery actions (`evaluation.py:845–866`, `support.py:163–370`). |
| Repair: deterministic, capped, disclosed, downgrades support | **Verified** | Pure function of input bytes (`repair.py:28`); `MAX_ASSUMPTIVE_FRACTION = 0.30` → `EXCESSIVE_REPAIR` (`repair.py:52,533`); every fix logged, assumptive fixes downgrade to `weakly_supported` (`pipeline.py:1486`). No unlogged mutation path found; even chronological re-sorting is disclosed. Live-confirmed on `filthy_requests.csv`. |
| Built-in models, mandatory baselines | **Verified (deliberately minimal)** | Seven stdlib models incl. real Theta and additive Holt-Winters (`models.py`); `BASELINES` cannot be configured out (`evaluation.py:245–278`); selection must beat the strongest baseline by margin + stability gates or the baseline publishes. |
| Eight sandboxed TSFM adapters, pinned revisions | **Verified as code; pins unverifiable offline** | All eight registered with pinned HF commit SHAs and a hard `UnpinnedWeights` failure (`tsfm.py:56–106`); per-model uv venvs with exact pip pins, JSON-over-stdio workers (`tsfm_sandbox.py:88–122`); sandboxed candidates enter the same folds. |
| Covariate/context admission only on identical-fold wins; effect learning gated on confirmation | **Verified** | Forward selection vs the evaluated base executable on identical folds, majority-fold + leave-one-out robustness (`covariates.py`); effect learning requires `occurrence_status = 'confirmed'` in SQL (`tracking.py:1621`); external effect registry is scenario-only (`effect_registry.py:143,150`). |
| Determinism | **Verified** | One `import random`, content-seeded (`anomaly.py:34,360`); content-addressed IDs (`ids.py:53`); injectable clock; golden byte-for-byte artifact tests. |
| LLM boundary ("nothing here reachable from the numerical pipeline") | **Verified** | `llm.py` is a 63-line protocol, imported only by `temporal_intent.py` (itself unwired from product surfaces); no-adapter default degrades to deterministic templates. |

**Engine-level deviations worth fixing:** seasonal-period detection runs
on the full visible series including calibration/test partitions
(`pipeline.py:317–318`) — a small hyperparameter leak into the
"untouched" fold; the default residual pooling; the vacuous causal
check; the legacy name-based TSFM path.

**Code shape:** 37,070 lines across 70 modules; zero TODO/FIXME; the
numerical core is stable while churn concentrates on the agent contract
boundary. Largest accretion site: `evaluate()` is a single 1,264-line
function (`evaluation.py:773–2037`). Orphans: `feedback.py` (663 lines,
alive only via the `gnomon-feedback` console script), `temporal_intent.py`,
`temporal_vocabulary.py` (imported by no product surface).

## 2. Product surfaces

Live-verified over stdio, CLI, and Python import; the surface is
substantially real and unusually well-engineered. Highlights confirmed:
default `evidence` profile = exactly `gnomon_describe` + `gnomon_forecast`
(observed via `tools/list`); responses budgeted at 8,192 bytes with
protected keys never trimmed and live payloads at ~3–4.4 KB (~1K
tokens); `data_ref` round-trip works with typed `resupply_data` repair;
`context_ref` is SQLite-backed and namespaced; structured errors carry
literally issuable `tool_call` repair objects; `gnomon capabilities` is
a truthful machine-readable build report; the Python API matches its
docs exactly; the skill and examples match the current surface.

Defects, ranked:

1. **MCP ambient-config hole (contradicts a hard documented
   guarantee).** `gnomon_forecast(candidates=[...])` reaches
   `_restrict_candidates`, which calls `load_config()` when the MCP
   path passed `config=None` (`toolspec.py:1972` → `runtime.py:315`
   `resolved = deepcopy(config) if config is not None else load_config()`),
   discovering ambient `gnomon.toml` — including `[backends.api]`
   network TSFM endpoints. README:312 and `cli-reference.md:209–211`
   state MCP calls never read ambient project config. Latent (requires
   an operator-authored config), but it is a hole in the one
   security-relevant guarantee.
2. **`full` profile is not a superset of the default.** `full` = 18
   tools (docs say 21, `quickstart-mcp.md:96`) and *excludes*
   `gnomon_describe` (`toolspec.py:2925–2947`) — upgrading
   evidence→full silently removes the default profile's primary tool
   and breaks the skill's guidance.
3. **The README's own demo question exceeds the default surface.** The
   60-second pitch asks for a cost-ratio alert rule; `gnomon_monitor`
   is absent from `evidence`, so the agent computes the 20× tradeoff
   itself — exactly the behavior Gnomon exists to prevent. Similarly,
   over-budget responses point to artifacts, but `gnomon_get_artifact`
   is not on the default profile (dead end for pure-MCP hosts).
4. **README shows a plain-text CLI rendering that does not exist**
   (README:226–234). Actual stdout is a JSON envelope; the readable
   text is `summary.md`. The numbers shown are correct (reproduced
   live: drift, 205/208/211) but first-contact users will think the
   docs are for another version.
5. **CLI flag inconsistency + broken suggester.** `--repair` is
   accepted by `forecast`/`inspect` but not `monitor`/`decide`; the
   error suggester emits "Did you mean `--repair` instead of
   `--repair`?" with a `rename_flag` repair option naming the same
   flag. An MCP bad-args error references `details.missing_arguments`
   that is empty and says "run with `--help`" (CLI-speak in MCP).
6. **Test suite is red on the documented dev path.**
   `tests/test_tsfm.py:101` hard-imports numpy with no skip guard;
   numpy is not in the `[dev]` extra, so `PYTHONPATH=src pytest -q`
   fails one test on a clean dev install (1,600/1,601 with dev
   extras; 1,597/1,601 with none).
7. Metadata/doc nits: pyproject classifier says **MIT** while
   LICENSE/README are **Apache-2.0**; `docs/README.md:76` says "seven
   pinned adapters" (there are eight); `getting-started.md:70–75`
   omits `lineage.json`; `gnomon_run`'s `robust_decision` surface is
   advertised (README:341) without saying it is `mega`-profile-only;
   `store:` inputs demand `--time`/`--target` that ingest already
   bound; no CLI `describe` exists for humans to reproduce the
   agent's most common call; `quickstart-mcp.md:100` has a dangling
   sentence and `:255` a stray mid-file H1.

## 3. Evidence and benchmark claims

The repository operates a two-tier evidence economy. The **honest
tier** — `results/` pre-registrations with committed raw JSON,
falsifier sections, and negative results published (news-regime,
short-history-guardrail, structural-effects, proposer-trust-warrant's
"precondition failed, not run") — is exemplary. The **headline tier**
is weaker:

| Claim | Verdict |
| --- | --- |
| LeakTrap control: 13/35 answered leaked, 4 verbatim transcriptions | **Supported** — recomputed from `results/leaktrap/` raw rows; deterministic grading, fair prompt (rule stated to the control), honest-limits section. |
| LeakTrap "Gnomon 0/40, McNemar p = 0.00024" | **Supported arithmetic, tautological framing** — p recomputes to 0.000244, but the instrument cannot flag the Gnomon arm (ceiling over Gnomon's own models; max `leak_advantage` exactly 0.0 in all 40 rows). Fix + disclosure exist only on unmerged `claude/leak-trap-benchmark-v61t70` (`44cd2d4`, `be78f88`). Main cites the number undisclosed (README:170, `docs/leakage-trap-results-2026-08.md`). |
| 1,800-execution surface experiment (96.3% / 89% / 12.7K / 1 call) | **Unsupported in-repo** — no artifact, writeup, manifest, corpus, seed, or model named anywhere in tree or history (`git log -S"96.3"` finds prose only, commit `b8c54b0`). The only committed surface-experiment record is the superseded first run, where `evidence` failed every gate (71K tokens, 4 median calls, 3/4 yield) and which warned "four rows are far too few for an accuracy claim." The repo's most consequential product decision (default = `evidence`) rests on evidence outside the repo. Mitigating: the chosen outcome was the unified plan's *pre-committed fallback*, so the decision is defensible even if the numbers are not citable. |
| Fold-stride / selection-loss / shrinkage measurements | **Partial** — code seams, tests, and decisions exist; decision-driving p-values (0.281, 0.81, 0.0009) have no committed raw data or comparison scripts. Irreproducible as shipped. |
| Analog-pooling kill (348-row 0% supply) | **Partial** — sound pre-registered kill logic; the counted runs were untracked in `907839a`, so counts are no longer verifiable. |
| External benchmark adapters (CiK, AnomLLM, MTBench, TemporalBench, TimeSage) | **Real and runnable** (LLM arms need `OPENROUTER_API_KEY`); almost no committed results. The one committed external assessment (`gnomon-mcp-assessment.md`, root, unlinked, superseded) shows Gnomon-MCP *underperforming* control on 3 of 4 TemporalBench task families on tiny samples — an honest but stale record. |
| Compiled-benchmark execution contract (`83f8f3d`) | **Verified** — rejects model-authored forecast values when the host compiled an intent and zero MCP calls were made (`benchmarks/temporalbench/mcp_agent.py`), preventing the product arm from silently becoming the baseline arm. |
| The three evaluation docs (main/MCP/dogfood 2026-08) | **Supported** — sampled fixes all verified in code (e.g. `test_threshold_probabilities_agree_with_published_quantiles`, `--cost-ratio` repair, `MAX_FIT_HISTORY`); the repo twice corrected its own external evaluator with evidence. |

No committed artifact numerically contradicts a doc that cites it. The
gap is availability, not falsification: the strongest claims are the
least documented, inverting the repo's own standard.

**Self-grading posture:** no LLM-judge grading anywhere (all internal
scoring deterministic); residual risks are self-attested leakage logs,
the LeakTrap ceiling basis, and vendor-authored synthetic corpora. The
unmerged LeakTrap branch (vendor-independent reference implementations,
honest-heldout false-positive arms) is the designed antidote and should
be merged.

## 4. History: convergence with a late counter-signal

289 commits, 2026-07-28 → 2026-08-20 (24 days); three names in five
weeks (Headwater → Aion at `d413dac`, Aion → Gnomon at `0bf426b`
/v0.5.0). The mid-arc (Aug 2–15) is a model of measurement discipline:
a 58-finding review with all sampled fixes verifiable in current code;
pre-registered decision rules that *rejected the project's preferred
design* (all Phase-3 arms failed their gate) followed by executing the
pre-committed fallback; three features built and left default-off
because measurements said so; a shipped integration (Hermes) and a
shipped planner culled as unused, with `COMPATIBILITY.md` retirements
verified really gone (zero grep hits for all seven removed v0.2 tools,
four planner tools, both env flags). Rename hygiene is clean — a test
(`tests/test_docs_current.py`) even guards against rename residue.

Counter-signals:

- **The final 48 hours re-expand without specs**: `8a87958` (#73) is a
  squash of 165 files / +15,107 lines adding eight `temporal_*` modules
  including `temporal_planner.py` — a planner re-grown in the same
  commit that deleted the culled one — plus orphaned `feedback.py`;
  `038291e` adds six more benchmark families the same day. Nothing in
  `specs/` covers this layer.
- **Release state is incoherent**: pyproject says 0.5.0; the only tag
  is `v0.4.0` (an Aion release); the changelog holds a 1,020-line
  "Unreleased" section containing the 0.5.0 bump itself; no PyPI
  release despite README promising one. Releases stopped Aug 1 while
  development accelerated through two breaking surface removals.
- **The commercial-validation track in `specs/unified-plan.md` — the
  plan's own test of purpose — was never executed**: no funnel
  instrumentation, no cohort thresholds, no first-run telemetry. The
  product has converged impressively without ever meeting a user.
- The never-built delta from the v0.1 spec/design docs (as the README
  admits) is large: hosted/cloud tier, HTTP surface, question
  templates, scheduled reruns/sharing, the agentic
  compiler/mapper/planner LLM layer. One v0.1 non-goal (anomaly
  detection) became a headline verb.

## 5. Cross-reference: the time-series reasoning literature

Against the TMLR survey's taxonomy (topology × objective × attribute
tags), Gnomon's self-classification (README:537–545) is accurate: an
agent+Gnomon system is branch-structured, spans traditional analysis,
explanation, and advisory decision support, and carries decomposition/
verification/ensembling/tool-use attributes.

**The positioning targets a real, visible gap.** Across all three
survey tables, every system tagged with verification (T-Ver) implements
it as LLM self-critique; none implements verification as deterministic
external code that the model cannot override. Gnomon's central design
choice — moving verification, model selection, uncertainty, and
timestamp discipline out of the LLM into a deterministic boundary — is
exactly the axis the surveyed field leaves open. The survey's own
corpus supports Gnomon's premise: *Language Models Still Struggle to
Zero-shot Reason about Time Series* (EMNLP 2024), *Are Language Models
Actually Useful for Time Series Forecasting?* (NeurIPS 2024), *Context
parroting: a tough-to-beat baseline* (mandatory-naive-baseline motif),
*Time Travel is Cheating* (leakage in live evaluation), and the CiK and
AnomLLM benchmarks — the latter two being precisely the external
benchmarks Gnomon ships adapters for. The AION/TimeClaw disambiguation
in the README is consistent with the survey (neither appears in its
tables; they are agent-side scaffolding, the complementary half).

**Gaps relative to the literature** (mostly deliberate scope, worth
stating): no classification, segmentation, structure discovery, or
generation; no multimodal (plot/vision) reasoning; no retrieval or
knowledge access (context events must be caller-supplied — the culled
analog-pooling/news lanes were the exploration of this, killed by
measurement); temporal QA only via typed questions, not free-form; and
validation against the field's reasoning-first benchmarks is thin —
of the survey's ~14 reasoning-first benchmarks, Gnomon adapts two
(CiK, AnomLLM) plus MTBench/TemporalBench lanes, with almost no
committed results, and the one committed external comparison shows
raw-accuracy losses. The trust-vs-accuracy trade (Gnomon's WAPE 0.205
vs control 0.157 on LeakTrap; 2.9× MSE vs last_value floor in the
news-regime study) is honestly recorded and is the natural cost of
mandated-baseline publication — but it means the product's case rests
on trust properties, not accuracy, and the trust-property evidence has
the gaps in §3.

## 6. Consolidated findings register (ranked)

| # | Severity | Finding | Where |
| --- | --- | --- | --- |
| 1 | High (claims integrity) | Flagship 1,800-execution surface-experiment numbers have no in-repo artifact; predecessor experiment pointed the other way | README:141–144, `product-position.md:73–76`, `b8c54b0` |
| 2 | High (claims integrity) | LeakTrap "0/40 + McNemar" framing tautological on Gnomon side; fix/disclosure unmerged | README:170, `results/leaktrap/`, branch `claude/leak-trap-benchmark-v61t70` |
| 3 | High (guarantee hole) | MCP `candidates` argument can pull ambient `gnomon.toml` incl. network TSFM backends, contradicting documented isolation | `toolspec.py:1972` → `runtime.py:315` |
| 4 | Medium | `full` profile excludes `gnomon_describe` and is 18 tools, not documented 21; default→full upgrade is a trap | `toolspec.py:2925–2947`, `quickstart-mcp.md:96` |
| 5 | Medium | Default profile cannot answer the README's own demo question (no `monitor`) nor read the artifacts its responses point to | README:117–119, `toolspec.py:2932` |
| 6 | Medium | Verifier does not run on descriptive surfaces; "every response" overstated | `toolspec.py:1077–1186` |
| 7 | Medium | Default interval calibration pools selection-fold residuals (disclosed, but diagram implies strict split); seasonal-period detection sees test partition | `evaluation.py:1692–1746`, `pipeline.py:317–318` |
| 8 | Medium | Test suite red on documented dev path (unguarded numpy import) | `tests/test_tsfm.py:101` |
| 9 | Medium | Release/version incoherence: 0.5.0 untagged, 1,020-line Unreleased changelog, no PyPI | `pyproject.toml:7`, `CHANGELOG.md` |
| 10 | Medium | Three seam measurements + analog-pooling kill rest on uncommitted data | `docs/*-measurement-2026-08.md`, `907839a` |
| 11 | Low | `--repair` inconsistent across verbs; suggester recommends the rejected flag back; MCP error references empty `details.missing_arguments` | `cli.py`, `toolspec.py` |
| 12 | Low | MIT classifier vs Apache-2.0 license; "seven adapters" vs eight; `lineage.json` missing from getting-started; README plain-text output block vs JSON stdout; unlinked stale root `gnomon-mcp-assessment.md`; `specs/…_v1.md` duplicate; orphaned `feedback.py`/`temporal_intent.py`/`temporal_vocabulary.py`; `docs/README.md:18` contradicts README's experiment confidence | various (see §2, §4) |
| 13 | Low | Legacy name-based TSFM republication path; vacuous causal check; `store:` re-asks bound columns; no CLI `describe` | `pipeline.py:562–581`, `verifier.py:39` |

## 7. Recommendations

1. **Restore claims discipline on the two headline numbers**: either
   commit the 1,800-execution corpus/manifests/results (or a results
   doc meeting the workflow README's own decision-gate standard) or
   strip the percentages from README and product-position; merge the
   LeakTrap de-tautologizing branch and re-state the result as
   "control leaked 13/35 with 4 verbatim transcriptions; Gnomon's
   snapshot makes leakage structurally unavailable (access-log
   verified)" — which is the defensible form of the claim.
2. **Close the MCP ambient-config hole** (pass an explicit
   `GnomonConfig()` through the `candidates` path or thread the
   MCP-null config into `_restrict_candidates`) and add a regression
   test asserting no `load_config()` call is reachable from any MCP
   tool invocation.
3. **Make profiles monotonic** (or rename `full`), put `gnomon_monitor`
   or a cost-rule field within reach of the default surface, and give
   the default profile an artifact reader or inline-on-request escape.
4. **Ship**: tag 0.5.0/0.6.0, cut the Unreleased section, publish to
   PyPI — then run the unified plan's dormant validation track; the
   repo's own product metrics (`product-position.md:84–94`) are
   currently all unmeasured.
5. Hygiene: numpy skip-guard; fix MIT classifier; wire or cull the
   orphaned modules; write the missing spec for the #73 temporal
   layer; fix the `--repair` suggester; align README's CLI output
   block with reality; commit the seam-measurement scripts next time
   a measurement decides a default.

---

*Method note: four parallel deep audits (core engine; product
surfaces; benchmarks/results; history/specs) plus direct execution:
editable install, full test suite, all five verbs, bitemporal
ingest/as-of replay, MCP stdio handshake and tool calls on `evidence`
and `full` profiles, and recomputation of the LeakTrap McNemar
statistic from committed raw artifacts.*
