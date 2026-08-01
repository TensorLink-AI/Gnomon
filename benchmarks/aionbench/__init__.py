"""AionBench — temporal-reasoning benchmarks for Aion and for LLMs alone.

The package answers one question: *does routing a temporal question through
Aion make a model better at it?* Every suite therefore runs the same tasks,
same model, same decoding parameters under at least two conditions — the
model on its own, and the model with Aion's tool surface — and reports the
paired difference.

Models are reached through OpenRouter so open- and closed-weight frontier
models can be compared on one code path with one key.

Nothing here is part of Aion's numerical contract: this package consumes the
runtime through its public tool surface exactly as an agent would.
"""

from __future__ import annotations

from .contracts import (
    Attempt,
    BenchError,
    BenchTask,
    SeriesRef,
    TaskOutcome,
    ToolCall,
)

__all__ = [
    "Attempt",
    "BenchError",
    "BenchTask",
    "SeriesRef",
    "TaskOutcome",
    "ToolCall",
]
__version__ = "0.1.0"
