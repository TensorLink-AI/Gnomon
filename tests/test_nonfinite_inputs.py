"""No non-finite numeric value reaches evaluation.

``float("nan")`` and ``float("inf")`` parse, so the strict loader used
to admit them as observations — after which ``error_score`` returned
NaN (not ``None``), and every aggregate that touched the score silently
became NaN. The strict loader now refuses with a structured error;
repair modes treat the documented textual sentinels as missing and
handle the rest through the unparseable path (refuse under ``safe``,
drop-with-disclosure under ``aggressive``); ``error_score`` guards its
own arithmetic because TSFM adapter outputs never pass the loader.
"""

import math
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.data import load_observations
from gnomon.repair import RepairLog, parse_number


def _csv(tmp_path: Path, cells: list[str]) -> Path:
    rows = [f"2026-01-{index + 1:02d},{cell}" for index, cell in enumerate(cells)]
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


CLEAN = ["100.0", "101.0", "102.0", "103.0"]


@pytest.mark.parametrize("cell", ["inf", "-inf", "Infinity", "1e999"])
def test_strict_loader_refuses_infinities(tmp_path, cell):
    path = _csv(tmp_path, CLEAN + [cell])
    with pytest.raises(GnomonError) as caught:
        load_observations(str(path), "timestamp", "value", None)
    assert caught.value.code == "NON_FINITE_TARGET"
    assert caught.value.details["row"] == 6
    actions = [option["action"] for option in caught.value.to_dict()
               ["error"]["repair_options"]]
    assert "supply_arguments" in actions


def test_strict_loader_refuses_nan(tmp_path):
    # float("nan") parses, so the old loader admitted it; unlike "inf"
    # it then poisoned aggregates invisibly (NaN compares false to
    # everything, so no threshold ever caught it).
    path = _csv(tmp_path, CLEAN + ["nan"])
    with pytest.raises(GnomonError) as caught:
        load_observations(str(path), "timestamp", "value", None)
    assert caught.value.code == "NON_FINITE_TARGET"


def test_json_nan_literal_is_refused(tmp_path):
    # Python's json module accepts the nonstandard NaN literal, so a
    # .json input could deliver a float NaN without any text form.
    path = tmp_path / "series.json"
    path.write_text(
        '[{"timestamp": "2026-01-01", "value": 100.0},'
        ' {"timestamp": "2026-01-02", "value": NaN}]',
        encoding="utf-8",
    )
    with pytest.raises(GnomonError) as caught:
        load_observations(str(path), "timestamp", "value", None)
    assert caught.value.code == "NON_FINITE_TARGET"


def test_safe_repair_still_refuses_infinity(tmp_path):
    # "inf" is corrupt, not missing: safe repair refuses rather than
    # silently narrowing the series.
    path = _csv(tmp_path, CLEAN + ["inf"])
    with pytest.raises(GnomonError) as caught:
        load_observations(str(path), "timestamp", "value", None, repair="safe")
    assert caught.value.code == "INVALID_TARGET"


def test_aggressive_repair_drops_infinity_with_disclosure(tmp_path):
    # Enough clean rows that one dropped row stays under the
    # EXCESSIVE_REPAIR ceiling (5% of rows).
    clean = [f"{100 + index}.0" for index in range(30)]
    path = _csv(tmp_path, clean + ["inf"])
    log = RepairLog()
    observations, _, _ = load_observations(
        str(path), "timestamp", "value", None,
        repair="aggressive", repair_log=log,
    )
    assert len(observations) == len(clean)
    assert all(math.isfinite(item.value) for item in observations)
    codes = [action.code for action in log.actions()]
    assert "unparseable_row_dropped" in codes


def test_textual_nan_is_a_missing_sentinel_under_repair(tmp_path):
    # Documented sentinel behaviour is unchanged: textual NaN means a
    # missing measurement, disclosed as such, not a corrupt one.
    path = _csv(tmp_path, CLEAN + ["NaN"])
    log = RepairLog()
    observations, _, _ = load_observations(
        str(path), "timestamp", "value", None,
        repair="safe", repair_log=log,
    )
    assert len(observations) == len(CLEAN)
    codes = [action.code for action in log.actions()]
    assert "missing_value_dropped" in codes


def test_parse_number_rejects_nonfinite_normalised_forms():
    with pytest.raises(ValueError):
        parse_number("inf", None)
    with pytest.raises(ValueError):
        parse_number("-Infinity", None)


def test_error_score_returns_none_for_nonfinite_predictions():
    # TSFM adapter outputs never pass the loader: a NaN prediction must
    # make the fold unscoreable (None), not a NaN score.
    from gnomon.evaluation import error_score

    assert error_score([100.0, 101.0], [float("nan"), 101.0]) is None
    assert error_score([100.0, 101.0], [float("inf"), 101.0]) is None
    assert error_score([float("nan"), 101.0], [100.0, 101.0]) is None
    assert error_score([100.0, 101.0], [99.0, 102.0]) is not None
