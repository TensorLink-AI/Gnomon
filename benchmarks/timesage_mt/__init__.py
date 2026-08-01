"""Adapter for TimeSage-MT, the multi-turn benchmark for agentic time
series reasoning (arXiv:2606.01498).

Task files, dialogues, visibility contracts, and per-turn verification
specs come from the official dataset (hf.co/datasets/Timesage/TimeSage-MT,
Apache-2.0). This package replays the dialogues against an agent and
scores responses with the dataset-embedded ``finding_verify`` specs. See
benchmarks/timesage_mt/README.md for what is official and what is not.
"""
