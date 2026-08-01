# Aion architecture

Mermaid maps of the design described in
[`Aion_System_Design.md`](../Aion_System_Design.md), drawn against what the
code in `src/aion` actually does. Every diagram renders on GitHub.

The single invariant all of these encode: **the LLM proposes; Aion validates,
computes, and owns every number.** Nothing on an agent's side of a boundary
in these diagrams can produce or edit a forecast value, a metric, an
interval, a selection decision, or a support status.

| Diagram | Question it answers |
| --- | --- |
| [1. Layers](#1-layers) | Who calls what, and where does business logic live? |
| [2. Five verbs](#2-five-verbs) | What can Aion be asked to do? |
| [3. Forecast pipeline](#3-forecast-pipeline) | How does a file become an artifact? |
| [4. Evaluation partitions](#4-evaluation-partitions) | Why is the reported score honest? |
| [5. Enrichment admission](#5-enrichment-admission) | How does outside knowledge earn its place? |
| [6. Support states](#6-support-states) | When does Aion refuse? |
| [7. Bitemporal store](#7-bitemporal-store) | How is leakage made structural? |
| [8. Agent sequence](#8-agent-sequence) | What does a real agent turn look like? |
| [9. Module map](#9-module-map) | Which module owns which responsibility? |

---

## 1. Layers

The runtime library is canonical. CLI, MCP, Python, and the Hermes plugin are
adapters over the same contracts — no business logic exists only in a command
handler or a prompt.

```mermaid
flowchart TB
    subgraph consumers ["Consumers"]
        human["Human or application"]
        agent["Agent: Hermes, Claude Code,<br/>any MCP client"]
    end

    subgraph adapters ["Interface layer — adapters only, no business logic"]
        cli["CLI<br/>aion forecast / inspect / investigate<br/>detect / decide / monitor"]
        mcp["MCP stdio server<br/>20 typed tools"]
        pyapi["Python API<br/>aion.runtime"]
        hermes["Hermes plugin + skill<br/>integrations/hermes"]
    end

    subgraph core ["Canonical deterministic runtime"]
        gate["Task compile + policy gate<br/>contracts, config, registry"]
        repair["Disclosed repair<br/>repair.py"]
        temporal["Temporal normalisation + snapshot<br/>temporal, temporal_store"]
        pipe["Forecast pipeline<br/>pipeline.py"]
        macro["Macros: investigate, detect,<br/>decide, monitor"]
        evaluate["Rolling evaluation + selection<br/>evaluation, ensemble, meta_model"]
        supportm["Support assessment<br/>support.py"]
    end

    subgraph guard ["Trust boundary — every response passes here"]
        lineage["Typed lineage<br/>lineage.py"]
        verifier["Claim verifier<br/>verifier.py"]
    end

    subgraph out ["Outputs"]
        artifact["artifact.json + evidence.jsonl<br/>forecast.csv + summary.md"]
        tracking["Tracking store<br/>realised scoring, leaderboards"]
    end

    llm["Host LLM<br/>bring your own brain"]

    human --> cli
    agent --> mcp
    agent --> hermes
    human --> pyapi
    hermes --> mcp

    cli --> gate
    mcp --> gate
    pyapi --> gate
    hermes --> gate

    gate --> repair --> temporal --> pipe
    temporal --> macro
    pipe --> evaluate --> supportm
    macro --> supportm
    supportm --> lineage --> verifier
    verifier -->|"pass"| artifact
    verifier -->|"violation"| err["AionError<br/>CLAIM_VERIFICATION_FAILED"]
    artifact --> tracking
    tracking -.->|"realised priors"| gate

    llm -.->|"prompt + parse owned by aion.workflows"| gate
    agent -.->|"proposes questions, mappings,<br/>context, explanations"| gate

    classDef never fill:#fde,stroke:#c33,color:#000
    class llm,agent never
```

Pink nodes are LLM-side — nothing there can produce a number. Dashed edges are
**proposal** paths. Solid edges inside the runtime are the only paths that
produce numbers, and they all terminate at the verifier.

---

## 2. Five verbs

One router, five questions, one shared substrate. Routing is advisory — the
evaluated run's own backtest is the final selector, and an explicit `model=`
override beats the router entirely.

```mermaid
flowchart LR
    q["Agent or user question"] --> router["router.route<br/>capability filter → tracking prior<br/>→ backtest tiebreak"]

    router --> f["aion forecast<br/><i>What happens next?</i>"]
    router --> i["aion investigate<br/><i>What changed?</i>"]
    router --> d["aion detect<br/><i>What is abnormal?</i>"]
    router --> dec["aion decide<br/><i>What should we do?</i>"]
    router --> m["aion monitor<br/><i>When do we intervene?</i>"]

    f --> fout["Backtested selection,<br/>residual-quantile intervals,<br/>threshold crossings"]
    i --> iout["Changepoints, regime vs transient,<br/>ranked <b>associational</b> explanations"]
    d --> dout["Detectors graded on injected<br/>anomalies; every candidate F1 disclosed"]
    dec --> decout["Exceedance scenarios, feasibility,<br/>expected utility"]
    m --> mout["Sequential exceedance risk,<br/>cost-optimal alert rule"]

    fout & iout & dout & decout & mout --> sub["Shared substrate:<br/>operators • snapshot • lineage • verifier"]
    sub --> art["Evidence-linked artifact<br/>or typed abstention"]
```

`investigate` stops at ranked associational explanations **by design** — the
verifier's `CAUSAL_CAPABLE_KINDS` set is deliberately empty, so a causal claim
cannot be emitted no matter what an LLM writes.

---

## 3. Forecast pipeline

The stages in `pipeline.py`, in execution order, as driven by
`runtime.forecast`.

```mermaid
flowchart TD
    start(["aion forecast INPUT"]) --> load["<b>load_stage</b><br/>read CSV/TSV/JSON/Parquet/Excel<br/>or store:dataset at --as-of"]

    load --> rep{"repair level"}
    rep -->|"off"| strict["strict validation<br/>reject on any defect"]
    rep -->|"safe (default)"| safe["cell-text normalisation<br/>formats, currency, sentinels"]
    rep -->|"aggressive"| aggr["structural fixes<br/>gaps, snapping, conflicts<br/>capped + downgrades support"]
    strict & safe & aggr --> fp["fingerprint + schema resolution<br/>frequency, timezone, series"]

    fp --> hz["<b>horizon_stage</b><br/>future timestamp grid<br/>matching the input geometry"]
    hz --> elig["model eligibility<br/>MODELS + eligible_tsfms"]

    elig --> ev["<b>evaluate_stage</b><br/>rolling-origin folds<br/>vs mandatory baselines"]
    ev --> enrich

    subgraph enrich ["Enrichment gates — identical folds"]
        direction LR
        ctx["<b>context_stage</b><br/>known-at gated events"]
        cov["<b>covariate_stage</b><br/>point-in-time covariates"]
        adj["<b>adjudicate_enrichments_stage</b><br/>championship ladder"]
        ctx --> adj
        cov --> adj
    end

    enrich --> mv["<b>multivariate_stage</b><br/>guarded VAR(1), opt-in"]
    mv --> pred["<b>predict_stage</b><br/>refit winner on all observations"]
    pred --> intv["<b>interval_stage</b><br/>residual quantiles from the<br/>calibration fold, horizon-widened"]
    intv --> thr["<b>threshold_analysis_stage</b><br/>crossing events, optional"]

    thr --> sup["support.assess_forecast_support"]
    sup --> lin["build lineage:<br/>artifacts • evidence • claims"]
    lin --> ver{"verifier.verify_or_raise"}
    ver -->|"violations"| fail["AionError with typed violations"]
    ver -->|"clean"| write["atomic write:<br/>artifact.json, evidence.jsonl,<br/>forecast.csv, summary.md"]
    write --> done(["immutable forecast_&lt;id&gt;/ directory"])
```

---

## 4. Evaluation partitions

Design review decision 1: the partitions are **disjoint**. This is what makes
the reported number something other than the number the selection procedure
was optimised against.

```mermaid
flowchart LR
    subgraph history ["Observed history, oldest → newest"]
        direction LR
        s1["fold 1"] --> s2["fold 2"] --> s3["fold …"] --> cal["penultimate fold"] --> test["final fold"]
    end

    s1 & s2 & s3 --> sel["<b>Model selection</b><br/>candidates vs mandatory baselines<br/>last_value • seasonal_naive"]
    cal --> calib["<b>Interval calibration</b><br/>residual quantiles"]
    test --> report["<b>Report only</b><br/>error + measured interval coverage"]

    sel -->|"winner must beat the strongest<br/>baseline by minimum_baseline_improvement"| pick["selected model"]
    pick --> refit["refit on <i>all</i> observations"]
    calib --> refit
    refit --> future["forecast the future"]
    report -.->|"never changes the choice"| pick
```

If there is not enough history to separate all three windows, the run does not
silently collapse them — it reports `degraded` (see next diagram).

---

## 5. Enrichment admission

Outside knowledge — an LLM-proposed launch date, an externally fetched
covariate — never improves a forecast by assertion. Each enrichment passes its
own ablation gate; when both are supplied, the adjudication ladder picks a
winner on identical folds and records the whole comparison as evidence.

```mermaid
flowchart TD
    prop["Agent proposes<br/>context events and/or covariates"] --> known{"known_at ≤ every<br/>historical fold cutoff?"}
    known -->|"no"| rejL["reject: TEMPORAL_LEAKAGE<br/>recorded with reason"]
    known -->|"yes"| scope{"correctly scoped to<br/>series and timestamps?"}
    scope -->|"no"| rejS["reject: out of scope"]
    scope -->|"yes"| abl["independent ablation<br/>on the identical selection folds"]

    abl --> gate{"beats the univariate control<br/>by the configured margin,<br/>on a majority of valid folds,<br/>not one anomalous fold,<br/>without degrading calibration?"}
    gate -->|"no"| rejE["reject: no demonstrated lift<br/>base model retained"]
    gate -->|"yes"| admitted["independently admitted"]

    admitted --> both{"both kinds admitted?"}
    both -->|"no"| single["use the admitted enrichment"]
    both -->|"yes"| ladder

    subgraph ladder ["adjudication.py — championship ladder, identical folds"]
        direction TB
        c0["base"]
        c1["base + context"]
        c2["base + covariates"]
        c3["base + both"]
        c0 & c1 & c2 & c3 --> cmp["best mean fold score<br/>ties → fewest enrichments<br/>→ fixed candidate order"]
    end

    ladder --> winner["winner"]
    single --> winner
    rejL & rejS & rejE --> base["base model"]
    winner & base --> evid["enrichment_adjudication evidence:<br/>the artifact <b>proves</b> the choice"]
```

---

## 6. Support states

Abstention is an answer. The v0.2 wire enum has five values;
`support.assess_forecast_support` maps them onto the harness vocabulary with
typed reasons and recovery actions, so a refusal is never a dead end.

```mermaid
flowchart TD
    run["evaluated run"] --> ok{"did rolling evaluation<br/>complete at all?"}
    ok -->|"no"| unsup["<b>unsupported</b> (wire)<br/>→ inconclusive"]
    ok -->|"yes"| sep{"separated selection,<br/>calibration and test windows?"}
    sep -->|"no"| deg["<b>degraded</b> (wire)<br/>→ conditionally_supported"]
    sep -->|"yes"| warn{"material warnings?<br/>few folds • high fold variance<br/>recent shift • missingness<br/>poor coverage • assumptive repairs"}
    warn -->|"yes"| weak["<b>weakly_supported</b> (wire)<br/>→ conditionally_supported"]
    warn -->|"no"| ens{"an ensemble beat the<br/>strongest baseline?"}
    ens -->|"yes"| se["<b>supported_ensemble</b> (wire)<br/>→ supported"]
    ens -->|"no"| sp["<b>supported</b> (wire)<br/>→ supported"]

    unsup --> rec1["recovery: reduce_horizon to the<br/>max supportable horizon,<br/>or provide_more_history"]
    deg --> rec2["recovery: provide_more_history"]
    weak --> rec3["recovery: review_warnings —<br/>each names the condition<br/>under which this holds"]

    sp & se --> deliver["forecast values delivered"]
    weak & deg --> deliver
    unsup --> nofc["<b>no forecast values</b><br/>the host cannot manufacture them"]

    classDef good fill:#dfd,stroke:#3a3,color:#000
    classDef mid fill:#ffd,stroke:#c93,color:#000
    classDef bad fill:#fdd,stroke:#c33,color:#000
    class sp,se good
    class weak,deg mid
    class unsup,nofc bad
```

A `supported` result means the current deterministic checks passed. It is not
a guarantee that the future will resemble history.

---

## 7. Bitemporal store

Leakage is structural, not behavioural: every read goes through a `Snapshot`
that *cannot* serve data published after its cutoff.

```mermaid
flowchart TB
    src["Source rows<br/>valid_time • value • published"] --> ing["aion ingest"]
    ing --> store[("Bitemporal store<br/>every value carries<br/>valid_time + known_time")]

    asof["--as-of INSTANT"] --> snap["Snapshot<br/>read handle bounded by known_time ≤ as_of"]
    store --> snap

    snap --> ops["Every operator reads<br/><b>only</b> through the snapshot"]
    ops --> art["Artifact records<br/>max_known_time of everything touched"]

    art --> vcheck{"verifier: any cited artifact with<br/>max_known_time &gt; as_of?"}
    vcheck -->|"yes"| viol["TEMPORAL_LEAKAGE violation<br/>response blocked"]
    vcheck -->|"no"| out["response leaves the process,<br/>provably clean"]

    note["Replay any past instant as it was<br/>honestly knowable — backtests cannot<br/>see revisions published later"]
    snap -.- note
```

---

## 8. Agent sequence

A realistic turn: the agent frames and explains, Aion computes and refuses.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant A as Agent (LLM)
    participant M as aion mcp serve
    participant R as Runtime
    participant V as Verifier

    U->>A: "Forecast requests 14 days out, and alert if we cross 340"
    A->>M: aion_inspect(file, time, target)
    M->>R: inspect_dataset
    R-->>M: schema, frequency, data_quality: repaired_safe, suggested_next
    M-->>A: typed result

    Note over A: LLM proposes mappings and a<br/>threshold from the user's words —<br/>never invents a business threshold

    A->>M: aion_forecast(horizon=14, threshold=340, repair=safe)
    M->>R: pipeline: load → evaluate → adjudicate → predict → interval
    R->>R: support.assess_forecast_support
    R->>V: verify_or_raise(lineage, as_of)

    alt claim verification fails
        V-->>R: violations
        R-->>M: AionError CLAIM_VERIFICATION_FAILED
        M-->>A: typed error + repair options
    else clean
        V-->>R: ok
        R-->>M: artifact + evidence IDs + support status
        M-->>A: typed result
    end

    alt support == unsupported
        A-->>U: reports the abstention and its recovery actions
        Note over A,U: cannot manufacture values,<br/>cannot rename it "low confidence"
    else supported / conditionally_supported
        A->>M: aion_explain_run(forecast_id)
        M-->>A: evidence-linked claims
        A-->>U: explanation citing Aion's evidence IDs,<br/>warnings and support status preserved
    end

    Note over U,V: Later: aion_submit_actuals → aion_score →<br/>realised regret feeds the tracking prior
```

---

## 9. Module map

`src/aion`, grouped by responsibility. Arrows show the dominant dependency
direction; the temporal core never depends on an adapter.

```mermaid
flowchart TB
    subgraph adapt ["Adapters"]
        cli["cli"]
        mcps["mcp_server"]
        tool["toolspec"]
        initm["__init__ / runtime"]
    end

    subgraph contract ["Contracts and identity"]
        con["contracts"]
        ver["versioning"]
        ids["ids"]
        reg["registry"]
        cfg["config"]
    end

    subgraph temporal ["Temporal core"]
        tmp["temporal"]
        tstore["temporal_store"]
        datam["data"]
        repairm["repair"]
        fpm["fingerprint"]
    end

    subgraph compute ["Computation"]
        modelsm["models"]
        tsfmm["tsfm"]
        sandbox["tsfm_sandbox"]
        apii["api_inference"]
        evalm["evaluation"]
        ensm["ensemble"]
        meta["meta_model"]
        mvm["multivariate"]
        anom["anomaly"]
        opsm["operators"]
    end

    subgraph enrichm ["Enrichment"]
        ctxm["context"]
        ctxe["context_eval"]
        ctxmod["context_model"]
        covm["covariates"]
        adjm["adjudication"]
    end

    subgraph orch ["Orchestration"]
        pipem["pipeline"]
        macrom["macros"]
        planm["plan"]
        execm["execution"]
        routem["router"]
    end

    subgraph trust ["Trust and output"]
        linm["lineage"]
        verm["verifier"]
        supm["support"]
        artm["artifacts"]
        trackm["tracking"]
        decm["decision_model"]
    end

    subgraph llmb ["LLM boundary"]
        llmm["llm"]
        wfm["workflows"]
        epim["episodes"]
        aem["agent_eval"]
    end

    adapt --> orch
    adapt --> contract
    orch --> compute
    orch --> enrichm
    orch --> temporal
    compute --> temporal
    enrichm --> compute
    orch --> trust
    trust --> contract
    compute --> contract
    temporal --> contract
    llmb -.->|"proposals only"| orch
    trackm -.->|"realised priors"| routem
    tsfmm --> sandbox
    tsfmm --> apii
```

---

## The one-line version

```mermaid
flowchart LR
    a["agent frames<br/>the question"] --> b["Aion validates<br/>the temporal claim"]
    b --> c["Aion computes<br/>every number"]
    c --> d["verifier checks<br/>every claim"]
    d --> e["evidence-linked answer<br/><b>or</b> typed abstention"]
    e --> f["agent explains<br/>— without editing a value"]
```
