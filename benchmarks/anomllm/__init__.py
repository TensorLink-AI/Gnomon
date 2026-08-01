"""Adapter for the AnomLLM benchmark (ICLR 2025, "Can LLMs Understand
Time Series Anomalies?").

Datasets, prompt variants, and metrics belong to the official repo
(rose-stl-lab/AnomLLM); this package adds an Aion treatment that writes
predictions in the official results format so the official aggregator
scores every condition identically. See benchmarks/anomllm/README.md.
"""
