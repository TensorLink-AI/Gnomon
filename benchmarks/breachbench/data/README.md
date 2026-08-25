# BreachBench real-telemetry corpus

Four real operational-flavoured series (23,397 points), fetched
2026-08-24 from the MIT-licensed `facebook/prophet` repository's bundled
example datasets (`examples/*.csv` at `main`). Values only — no
timestamps, no names reach a prompt.

| File | What it really is | Texture it contributes |
| --- | --- | --- |
| `wiki_traffic_daily_log.csv` | Daily log page views of a real Wikipedia article, 2007–2016 (`example_wp_log_peyton_manning.csv`) | Real web traffic: weekly seasonality, event-driven spikes, slow drift |
| `sensor_temps_5min.csv` | Real 5-minute temperature sensor readings, Yosemite, 2017 (`example_yosemite_temps.csv`) | High-frequency sensor telemetry: diurnal cycles, sensor noise, gaps |
| `pedestrian_counts_daily.csv` | Real daily pedestrian sensor counts, Melbourne, 2017–2021 (`example_pedestrians_covid.csv`) | City telemetry with a genuine regime collapse (COVID lockdowns) |
| `retail_sales_monthly.csv` | Real US monthly retail sales, 1992–2016 (`example_retail_sales.csv`) | Strong seasonality on a growth trend |

These public series must be assumed present in LLM training data, so
every case is a windowed slice under a seeded positive affine transform
(threshold transformed identically), which preserves breach structure and
every dynamic while defeating verbatim recall; prompts carry values only.

To extend the corpus — the highest-value change is the client's own
operational exports — drop additional single-column CSVs (header
`value`, temporal order) in this directory. Prefer data past the models'
training cutoffs where available.
