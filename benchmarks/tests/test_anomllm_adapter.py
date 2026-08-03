"""Unit tests for the AnomLLM adapter's pure logic.

These run without the AnomLLM package, without numpy, and without any
dataset: only the adapter's own conversion and layout decisions are
covered. The faithfulness-critical scoring belongs to the official
``result_agg.py`` and is deliberately not reimplemented here — what is
tested is that the adapter matches the official conventions where it
previews them, and stays out of the official aggregator's way where it
does not.
"""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.anomllm.gnomon_detector import (
    binary_f1,
    default_records_path,
    intervals_to_vector,
    shield_from_official_scoring,
)
from benchmarks.anomllm.run_anomllm import gnomon_variant_name


def test_default_sidecar_lives_outside_official_results_tree(tmp_path):
    # Upstream collect_results scores every non-'requests' *.jsonl under
    # results/; a sidecar there rendered a phantom all-zero variant.
    path = default_records_path(tmp_path, "point", "gnomon", "detect")
    results_tree = (tmp_path / "results").resolve()
    assert results_tree not in path.resolve().parents
    assert path.name == "detect.gnomonbench.jsonl"


def test_sidecar_forced_into_results_tree_gets_requests_shield(tmp_path):
    inside = (tmp_path / "results" / "synthetic" / "point" / "gnomon"
              / "detect.gnomonbench.jsonl")
    shielded = shield_from_official_scoring(inside, tmp_path)
    # Upstream's filter is: 'requests' not in file and .endswith('.jsonl')
    assert "requests" in shielded.name
    assert shielded.suffix == ".jsonl"
    assert shielded.parent == inside.parent


def test_sidecar_shield_leaves_safe_paths_alone(tmp_path):
    outside = tmp_path / "gnomonbench" / "detect.gnomonbench.jsonl"
    assert shield_from_official_scoring(outside, tmp_path) == outside
    already = tmp_path / "results" / "detect.requests.jsonl"
    assert shield_from_official_scoring(already, tmp_path) == already


def test_binary_f1_matches_official_convention_on_clean_series():
    # Official compute_metrics: empty GT + empty prediction => all 1.0,
    # included in the mean; either side empty alone => 0.0.
    assert binary_f1([0, 0, 0], [0, 0, 0]) == 1.0
    assert binary_f1([0, 0, 0], [0, 1, 0]) == 0.0
    assert binary_f1([0, 1, 0], [0, 0, 0]) == 0.0


def test_binary_f1_ordinary_cases():
    truth = intervals_to_vector([{"start": 2, "end": 4}], 6)
    assert binary_f1(truth, truth) == 1.0
    partial = intervals_to_vector([{"start": 3, "end": 5}], 6)
    assert 0.0 < binary_f1(truth, partial) < 1.0
    disjoint = intervals_to_vector([{"start": 0, "end": 1}], 6)
    assert binary_f1(truth, disjoint) == 0.0


def test_variant_names_upstream_would_rescale_are_rejected():
    # postprocess_configs (upstream src/config.py) keys exactly these
    # names to an integer-rescaling postprocessor.
    for name in ("0shot-text-s0.3", "1shot-text-s0.3",
                 "0shot-text-s0.3-cot", "1shot-text-s0.3-cot"):
        with pytest.raises(argparse.ArgumentTypeError):
            gnomon_variant_name(name)


def test_ordinary_variant_names_pass_through():
    for name in ("detect", "detect-t0.5", "0shot-text", "gnomon-default"):
        assert gnomon_variant_name(name) == name
