"""Process isolation, contention, cutoff and recovery regression for milestone A."""
import json
from benchmarks.hosted_ledger.validate import run


def test_shared_ledger_handoff_and_restore(tmp_path):
    report = run(tmp_path / 'probe', workers=2, writes=4)
    assert report['concurrency']['distinct_writes'] == 8
    assert report['snapshot']['writer_committed_during_review']
    assert report['restore']['saved_lesson_unchanged']
    assert report['restore']['old_mae'] == 1.5
    assert report['restore']['revised_mae'] == 4.5
    manifest = json.loads((tmp_path / 'probe' / 'manifest.json').read_text())
    assert len(manifest['backup_sha256']) == 64
