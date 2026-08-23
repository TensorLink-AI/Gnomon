from gnomon.project_report import build_project_report, write_project_report
from gnomon.tracking import TrackingStore
from gnomon.cli import main


def test_empty_project_report_is_honest_and_offline(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    report = build_project_report(store, "capacity", month="2026-08")
    assert report["forecasts"]["registered"] == 0
    assert report["forecasts"]["mean_mase"] is None
    markdown, html = write_project_report(
        store, "capacity", tmp_path / "report", month="2026-08")
    assert "not measured" in markdown.read_text(encoding="utf-8")
    rendered = html.read_text(encoding="utf-8")
    assert "https://" not in rendered and "http://" not in rendered


def test_report_rejects_ambiguous_month(tmp_path):
    store = TrackingStore(tmp_path / "registry.db")
    try:
        build_project_report(store, "capacity", month="August")
    except ValueError as error:
        assert "YYYY-MM" in str(error)
    else:
        raise AssertionError("invalid month was accepted")


def test_top_level_report_is_the_friendly_track_report_alias(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    destination = tmp_path / "monthly"
    assert main(["report", "--project", "capacity", "--month", "2026-08",
                 "--output", str(destination)]) == 0
    assert (destination / "report.md").is_file()
    assert (destination / "report.html").is_file()
    assert '"status": "complete"' in capsys.readouterr().out
