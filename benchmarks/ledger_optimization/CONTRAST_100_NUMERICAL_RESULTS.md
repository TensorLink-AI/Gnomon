# Candidate 100: complete archived numerical results

The independently recalculated complete development result is negative. Across
104 matched cases, ledger mean RMSLE is 2.3994% worse than matched no-ledger
Gnomon. All 312 session grades are valid and workflow-complete, with no fallback
forecasts. The original boundary audit failed on the known cached-dictionary
ordering check; that nonzero exit remains preserved. Its full independent
reconciliation was still running when this numerical report was recorded.

| Arm | Cases | Mean per-case RMSLE | Numerical attempts |
| --- | ---: | ---: | ---: |
| Hermes alone | 104 | 0.4891305539018427 | 1116 |
| Hermes + Gnomon, no ledger | 104 | 0.477838621193496 | 1228 |
| Hermes + Gnomon + ledger | 104 | 0.4893039730220508 | 1136 |

Ledger versus no-ledger has 19 wins, 39 losses and 46 ties at the existing
1e-12 tolerance. The existing 2,000-draw series bootstrap (seed 142) gives a
relative-improvement interval of [-4.2506%, -0.8418%]. It excludes zero in the
unfavorable direction on this small, reused development sample. It is not a
confirmatory population claim. Against Hermes alone, ledger is 0.0355% worse,
with exploratory interval [-1.6252%, +0.9834%].

Every archived forecast was independently rescored against its original
14-step actual vector; maximum discrepancy from its saved grade was
2.220446049250313e-16. The check retained all four series, 26 origins, three
arms, the original 36-session pilot and all 276 continuation sessions. The
full original archive had already passed local verification of 51,908 files
plus its inventory. Neither forecasts nor agent sessions were rerun.

Reported execution usage is 2,627 forwarded/returned requests and 44,975,020
tokens, with one API error and one response lacking usage. Separately, 314
readiness requests report 4,368 tokens and two unknown usage records. Dollar
billing remains unknown, not zero. Full original costs remain unchanged.

Evidence: `results/contrast-100-arithmetic-001/{verify.py,command.json,report.json,rows.json}`
and `results/contrast-100-terminal-export-001`. The committed receipt records
the script, bootstrap source, original archive and numerical report hashes.
This calculation verifies arithmetic and coverage; it does not replace the
still-required boundary/history/source audit. The original failed status is
not relabeled as a clean completed audit.

This result does not meet the 20% target and does not open held-out data.
Candidate 107 was prospectively frozen before paid execution and remains a
separate proposed improvement in turning requested historical evidence into
optional experiments. It must be measured with fresh matched agent sessions;
this result cannot be removed or replaced by a later favorable run.

## Completed boundary reconciliation

The separate reconciliation subsequently passed all 397,019 checks over all
312 sessions, with zero integrity failures and shutdown gaps. It reproduced
the exact original cached-order failure before applying only the corrected
offline annotation auditor. Original execution files, original failed status
and costs stayed unchanged. Its process exited zero; no agents, forecasts or
provider fits were rerun.

The downloaded corrected report matches every independently checked grade and
both ledger contrasts exactly. Its hash is
`120c603885bc8e74beb4618de7b8c2fb29b496735267e3e014cfe11c23cec2ae`.
The receipt is `evidence/contrast-100-reconciled-final-001.json`; the separate
43-file reconciliation archive has SHA-256
`f5ad72a97960348306ef85a5ece68d8f1f0aef3680b61e6899a8101be68257ec`.
The earlier numerical report's pending-audit label is preserved as its state
when created, rather than overwritten.

The original SSH observation wrapper returned nonzero before the remote audit
finished. Same-boot PID/start-time checks proved its controller and worker were
still live, so neither was restarted. A later original-host check verified both
terminal identities and the successful reconciliation exit. That observation
failure is retained separately from the actual audit outcome. The first local
verification invocation also failed to import the repository from its script
directory; running the unchanged script through `runpy` from the repository
completed successfully. Neither incident changed an execution or score.

The completed report records 103 successful ledger reviews across 104 ledger
sessions. Mean distinct tested configurations were 2.73 for ledger versus 2.95
for no-ledger Gnomon. This is descriptive evidence that access to historical
comparisons did not itself produce a better search or better forecasts here;
it does not establish why agents made their choices.
