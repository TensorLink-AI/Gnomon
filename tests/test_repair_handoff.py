"""C5: safe repair by default, a spelled-out next call on failure, deterministic freezing."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from gnomon import GnomonSession
from gnomon.cli import main
from gnomon.contracts import GnomonError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FILES = ["daily_requests.csv", "messy_requests.csv", "filthy_requests.csv"]


def _cli(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("name", FILES)
def test_mcp_agent_reaches_a_data_ref_in_at_most_two_calls(name):
    with GnomonSession.from_config() as session:
        arguments = {"input": str(EXAMPLES / name), "target_column": "requests"}
        calls = 0
        while True:
            calls += 1
            try:
                result = session.call("gnomon_inspect", arguments, compact=False)
                break
            except GnomonError as exc:
                handoff = exc.details["next_call"]
                assert handoff["tool"] == "gnomon_inspect" and handoff["runnable"] is True
                # Within budget is not permission: aggressive changes evidence, so the user chooses it.
                assert handoff["admissible"] is None and handoff["requires_user_choice"] is True
                assert handoff["would_change"]["actions"] and exc.details["choices_required"] == {"repair": ["aggressive"]}
                assert exc.details["data_quality"]["status"] == "needs_aggressive"
                assert calls == 1, "one call to fail, one to recover, never a third"
                arguments = handoff["arguments"]  # the user's choice, made explicit
        assert calls <= 2 and result["data_ref"].startswith("data_")
        assert list(result)[:3] == ["schema_version", "status", "data_quality"]
        quality = result["data_quality"]
        assert quality["next_call"] is None
        assert quality["fixes"] + quality["dropped"] == sum(r["count"] for r in result["repairs"]
                                                          if r["code"] not in {"timezone_declared", "window_selected"})
        assert (quality["status"] == "clean") == (not result["repairs"])
        if quality["status"] != "clean":
            assert arguments["repair"] == "aggressive" and quality["status"] == "repaired_aggressive"


@pytest.mark.parametrize("name", FILES)
def test_cli_agent_reaches_a_data_ref_in_at_most_two_calls(name, capsys):
    argv = ["inspect", str(EXAMPLES / name), "--target", "requests"]
    code, payload = _cli(capsys, argv)
    if code != 0:
        handoff = payload["error"]["details"]["next_call"]
        assert handoff["runnable"] and handoff["argv"][0] == "gnomon" and handoff["argv"][-2:] == ["--repair", "aggressive"]
        assert handoff["requires_user_choice"] is True and handoff["admissible"] is None
        assert payload["error"]["details"]["data_quality"]["next_call"] == handoff
        assert payload["data_quality"] == payload["error"]["details"]["data_quality"], "agents read one top-level key"
        code, payload = _cli(capsys, handoff["argv"][1:])
    assert code == 0 and payload["data_ref"].startswith("data_")


def test_safe_never_invents_values_and_aggressive_is_never_applied_unasked(capsys):
    code, payload = _cli(capsys, ["inspect", str(EXAMPLES / "filthy_requests.csv"), "--target", "requests"])
    assert code == 2, "filthy needs aggressive; the default must not escalate on its own"
    assert payload["error"]["details"]["data_quality"]["status"] == "needs_aggressive"
    code, result = _cli(capsys, ["inspect", str(EXAMPLES / "filthy_requests.csv"), "--target", "requests", "--repair", "aggressive"])
    assert code == 0 and any(r["assumptive"] for r in result["repairs"]), "interpolation is disclosed, itemised"
    with GnomonSession.from_config() as session:
        safe = session.call("gnomon_inspect", {"input": str(EXAMPLES / "messy_requests.csv"), "target_column": "requests"}, compact=False)
        assert not any(r["assumptive"] for r in safe["repairs"])


def test_uncorrectable_input_hands_off_to_the_source(tmp_path):
    path = tmp_path / "broken.csv"
    rows = ["2026-01-%02d,%d" % (day, day) for day in range(1, 21)]
    rows[4] = "2026-01-05,not-a-number"
    rows[9] = "2026-01-10,also-bad"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n")
    with GnomonSession.from_config() as session:
        with pytest.raises(GnomonError) as caught:
            session.call("gnomon_inspect", {"input": str(path), "repair": "aggressive"}, compact=False)
        quality = caught.value.details["data_quality"]
        assert quality["status"] == "rejected"
        assert quality["next_call"] == {"action": "correct_target", "column": "value", "row": 6,  # 1-based, header is row 1
                                        "value": "not-a-number", "guidance": quality["next_call"]["guidance"]}
        # The safe default reaches the same verdict in one failing call.
        with pytest.raises(GnomonError) as default:
            session.call("gnomon_inspect", {"input": str(path)}, compact=False)
        assert default.value.details["next_call"]["action"] == "correct_target"
        assert caught.value.details["admissible"] is False


def test_same_file_always_freezes_to_the_same_snapshot():
    def freeze(name, **options):
        with GnomonSession.from_config() as session:
            result = session.call("gnomon_inspect", {"input": str(EXAMPLES / name), "target_column": "requests", **options}, compact=False)
            return result["data_ref"], result["snapshot"]["snapshot_id"], result["snapshot"]["source_ref"], len(result["repairs"])
    for name, options in (("daily_requests.csv", {}), ("messy_requests.csv", {}), ("filthy_requests.csv", {"repair": "aggressive"})):
        assert freeze(name, **options) == freeze(name, **options)
    # A clean file is untouched by safe mode: same identity as the strict path.
    assert freeze("messy_requests.csv") == freeze("messy_requests.csv", repair="off")
    # Across processes too: identity is content-addressed, not session-addressed.
    script = ("import json,sys; from gnomon import GnomonSession\n"
              "with GnomonSession.from_config() as s:\n"
              "    r = s.call('gnomon_inspect', {'input': sys.argv[1], 'target_column': 'requests'}, compact=False)\n"
              "    print(json.dumps([r['data_ref'], r['snapshot']['snapshot_id'], r['snapshot']['source_ref']]))\n")
    runs = [subprocess.run([sys.executable, "-c", script, str(EXAMPLES / "messy_requests.csv")], capture_output=True, text=True, check=True).stdout
            for _ in range(2)]
    assert runs[0] == runs[1] and json.loads(runs[0])[:3] == list(freeze("messy_requests.csv")[:3])


def test_invalid_timestamps_hand_off_to_the_timestamp_column_not_the_target(tmp_path):
    path = tmp_path / "badtime.csv"
    rows = ["2026-01-%02d,%d" % (day, day) for day in range(1, 21)]
    rows[2] = "not-a-date,3"
    rows[7] = "2026-13-45,8"
    path.write_text("when,hits\n" + "\n".join(rows) + "\n")
    with GnomonSession.from_config() as session:
        with pytest.raises(GnomonError) as caught:
            session.call("gnomon_inspect", {"input": str(path), "time_column": "when", "target_column": "hits"}, compact=False)
        correction = caught.value.details["data_quality"]["next_call"]
        assert correction["action"] == "correct_timestamp" and correction["column"] == "when"
        assert correction["row"] == 4 and correction["value"] == "not-a-date"
        assert "Preserve other supplied values" in correction["guidance"]
        assert "further validation may identify additional errors" in correction["guidance"]
