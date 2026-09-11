from copy import deepcopy

import pytest

from benchmarks.ledger_optimization.m5_prepare import CUTOFF, FIRST, calendar_mapping, fetch, prefix_metadata, prepare, select, verify_archive


def fixture():
    return [{'store_id': f'S{s}', 'item_id': f'I{s}_{i}',
             **{f'd_{day}': '1' for day in range(FIRST, CUTOFF + 1)},
             f'd_{CUTOFF + 1}': 'not read'} for s in range(10) for i in range(6)]


def test_future_targets_cannot_change_eligibility_or_partition():
    rows = fixture()
    before = deepcopy(rows)
    eligible, counts = prefix_metadata(rows)
    split = select(eligible, counts['stores'])
    assert rows == before
    for row in rows:
        row[f'd_{CUTOFF + 1}'] = object()
        row['future_oracle_score'] = -999
    assert prefix_metadata(rows) == (eligible, counts)
    assert select(eligible, counts['stores']) == split


def test_order_invariance_and_disjoint_stores_and_items():
    rows = fixture()
    e, c = prefix_metadata(rows)
    a = select(e, c['stores'])
    e, c = prefix_metadata(rows[::-1])
    assert select(e, c['stores']) == a
    assert len(a['development']) == 8 and len(a['reserved']) == 24
    for field in ('store_id', 'item_id'):
        assert not ({r[field] for r in a['development']} & {r[field] for r in a['reserved']})
    assert len({r['item_id'] for split in a.values() for r in split}) == 32


def test_invalid_prefix_excluded_but_future_invalid_values_ignored():
    rows = fixture()
    rows[0][f'd_{FIRST}'] = 'nan'
    rows[1].update({f'd_{day}': '0' for day in range(FIRST, CUTOFF + 1)})
    e, c = prefix_metadata(rows)
    assert len(e) == 58
    assert c['rejected_prefix_reasons'] == {'invalid_initial_history': 1, 'fewer_than_28_nonzero_initial_days': 1}


def test_duplicates_insufficient_stores_and_items_stop_without_relaxing():
    rows = fixture()
    with pytest.raises(ValueError, match='duplicate'):
        prefix_metadata(rows + [rows[0]])
    e, c = prefix_metadata(rows)
    with pytest.raises(ValueError, match='ten'):
        select(e, c['stores'][:9])
    with pytest.raises(ValueError, match='Insufficient'):
        select([r for r in e if r['item_id'].endswith('_0')], c['stores'])


def test_shared_items_are_not_reused_across_stores():
    rows = fixture()
    for row in rows:
        row['item_id'] = row['item_id'].split('_')[-1]
    e, c = prefix_metadata(rows)
    with pytest.raises(ValueError, match='Insufficient'):
        select(e, c['stores'])


def test_existing_outputs_and_wrong_archive_rejected_before_work(tmp_path):
    archive = tmp_path / 'm5.zip'
    archive.write_bytes(b'not the pinned data')
    with pytest.raises(ValueError, match='size'):
        verify_archive(archive)
    with pytest.raises(ValueError, match='overwrite'):
        fetch(archive)
    with pytest.raises(ValueError, match='reuse'):
        prepare(archive, tmp_path)


def test_calendar_without_d_uses_verified_one_based_daily_order():
    rows = [{'date': '2011-01-29'}, {'date': '2011-01-30'}]
    result = calendar_mapping(rows)
    assert str(result['d_1']) == '2011-01-29' and str(result['d_2']) == '2011-01-30'
    assert calendar_mapping([dict(row, d=f'd_{i}') for i, row in enumerate(rows, 1)]) == result
    with pytest.raises(ValueError, match='consecutive'):
        calendar_mapping(rows[::-1])
    with pytest.raises(ValueError, match='identifiers'):
        calendar_mapping([{'date': '2011-01-29', 'd': 'd_2'}])
