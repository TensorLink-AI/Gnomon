"""Adapter for TemporalBench (arXiv:2602.13272), the four-tier benchmark
for contextual and event-informed time series reasoning.

Task rows, official prompts, labels, and the reference metric module all
come from the official dataset (hf.co/datasets/Melady/TemporalBench,
Apache-2.0). Forecast metrics are computed by the officially shipped
``forecast_metrics_utils.py``, imported as-is. See
benchmarks/temporalbench/README.md.
"""
