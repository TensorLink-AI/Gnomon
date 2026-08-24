# DossierBench real-series corpus

Eight real observational time series (4,178 points), extracted verbatim
from the datasets bundled with `statsmodels` (BSD-licensed), which
redistributes them from their public originators. Values only — no
timestamps, no names reach a prompt.

| File | What it really is | Original source |
| --- | --- | --- |
| `co2_weekly_mauna_loa.csv` | Weekly atmospheric CO₂ at Mauna Loa, 1958–2001 | Scripps/NOAA |
| `sunspots_yearly.csv` | Yearly sunspot activity, 1700–2008 | SIDC/SILSO |
| `nile_annual_flow.csv` | Annual Nile volume at Aswan, 1871–1970 (contains the real 1898 regime break) | Cobb (1978) |
| `us_realgdp_quarterly.csv` | US real GDP, quarterly 1959–2009 | BEA via statsmodels macrodata |
| `us_unemployment_quarterly.csv` | US unemployment rate, quarterly 1959–2009 | BLS via statsmodels macrodata |
| `us_cpi_quarterly.csv` | US CPI, quarterly 1959–2009 | BLS via statsmodels macrodata |
| `us_m1_quarterly.csv` | US M1 money stock, quarterly 1959–2009 | FRB via statsmodels macrodata |
| `elnino_sst_monthly.csv` | Monthly Pacific sea-surface temperature, 1950–2010 | NOAA |

These are famous series and must be assumed to sit verbatim in every
LLM's training data. The benchmark therefore never shows a series whole
or raw: each case is a windowed slice at a sampled cutoff, passed through
a per-case seeded affine transform (positive scale, fresh offset) that
preserves every dynamic the questions ask about — direction, trend,
volatility ratios, seasonality, breaks — while breaking verbatim
sequence lookup. The transform and the memorization rationale are
disclosed in the run summary.

To extend the corpus, drop additional single-column CSVs (header
`value`) in this directory — real measured series only, and prefer data
past the models' training cutoffs where available.
