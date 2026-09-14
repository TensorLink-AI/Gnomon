# Validation identities locked before outcome access

The strongest numerical mechanism, 050 intraday mixtures, is locked for a
disjoint-series development validation. This stage selected identities only;
it did not produce forecasts or a new accuracy result. The original 20%
matched-agent goal remains unmet.

| Domain | Eight validation identities |
| --- | --- |
| Electricity | T107, T197, T298, T110, T20, T13, T34, T66 |
| Pedestrian | sensor_20, sensor_30, sensor_21, sensor_55, sensor_3, sensor_4, sensor_18, sensor_22 |

These are positions 24–31 in the original fixed hash order, after verifying the
first eight development and next sixteen final-reserved identities per domain.
No identity overlaps either previous panel. All 32 previously reserved final
series remain unchanged. Initial 730-hour eligibility prefixes were examined
in 037; new scored outcomes have not been numerically parsed in this stage or
used in preceding mechanism tests.

The [protocol](BROAD_VALIDATION_052.md) fixes 416 tasks, dates, numerical recipes,
the intraday action, historical retrieval, cold-start handling, both controls
and paired cluster/circular-block intervals before outcome access. It retains
the 20% promotion bar. A pass would still not establish a matched-agent final
result; the method must not be tuned on this validation and rerun until it wins.

Selection implementation/protocol frozen at `7e2abe1`; chosen algorithm checked
byte-for-byte against `11af992`. Three tests and 41 independent metadata checks
passed. Selection SHA-256:
`7940935335b7a924965f5974e861701d488fa961af893d4085c290b7a78de92d`.
[The receipt](evidence/broad-validation-identity-052.json) includes selected
metadata, source/implementation hashes, original memberships and audit results.

Next work is source preparation and a frozen execution adapter. No new API calls,
provider fits, new-series forecast scores, or final-reserved outcomes have been
accessed. Main and PyPI remain unchanged.
