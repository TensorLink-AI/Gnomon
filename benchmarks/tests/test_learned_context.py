import copy
from datetime import datetime, timedelta, timezone
import unittest
import numpy as np
from benchmarks.ledger_optimization.learned_context import MODELS, contrast, train_at, retrieve


class LearnedContextTest(unittest.TestCase):
    def fixture(self):
        rows = []; raw = {}; start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        for i in range(40):
            at = start+timedelta(days=i); close = at+timedelta(days=1)
            r = {'series_id': 'electricity:a', 'domain': 'electricity', 'origin': at.isoformat(),
                'last_target': close.isoformat(), 'outcome_recorded_at': close.isoformat(), 'features': [float(i)]*12}
            rows.append(r);raw[r['series_id'],r['origin']] = {'actual': [1.]*24,
                'point': {m: [float(j+i+1)]*24 for j,m in enumerate(MODELS)}, 'cv': {m: .1*j for j,m in enumerate(MODELS)}}
        return rows, raw, (start+timedelta(days=50)).isoformat()

    def test_centered_label_with_independent_arithmetic(self):
        pair = {'actual': [0.]*24, 'point': {m: [float(j)]*24 for j,m in enumerate(MODELS)}, 'cv': {m: .05*j for j,m in enumerate(MODELS)}}
        expected = np.log1p(np.arange(6))-.05*np.arange(6);expected -= expected.mean()
        np.testing.assert_allclose(contrast(pair), expected, atol=1e-12)

    def test_recording_and_origin_exclusions_do_not_access_unavailable_labels(self):
        rows, raw, now = self.fixture();model, baseline = train_at(rows,raw,'electricity',now)
        excluded = copy.deepcopy(rows[0]);excluded.update(series_id='electricity:late',outcome_recorded_at='2030-01-01T00:00:00+00:00')
        simultaneous = copy.deepcopy(excluded);simultaneous.update(series_id='electricity:same',origin=now)
        # Deliberately supply no raw entries for unavailable rows: lookup would fail.
        _, observed = train_at(rows+[excluded,simultaneous],raw,'electricity',now)
        self.assertEqual(baseline,observed)
        self.assertEqual(retrieve(model,baseline,[10.]*12)['current_features'],[10.]*12)

    def test_leaf_weights_match_independent_tree_traversal(self):
        rows,raw,now=self.fixture();model,record=train_at(rows,raw,'electricity',now)
        query=[10.]*12;result=retrieve(model,record,query);expected=np.zeros(40)
        def leaf(tree,x):
            node=0;x=np.asarray(x,dtype=np.float32)
            while tree['children_left'][node]!=-1:
                node=tree['children_left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['children_right'][node]
            return node
        for tree in record['trees']:
            indices=[i for i,r in enumerate(rows) if leaf(tree,r['features'])==leaf(tree,query)]
            expected[indices]+=1/(64*len(indices))
        np.testing.assert_allclose(result['record_weights'],expected,atol=1e-12)
        self.assertAlmostEqual(sum(expected),1.);self.assertTrue(np.all(expected>=0))

    def test_insufficient_history_never_trains_and_bad_query_rejected(self):
        rows,raw,now=self.fixture();model,record=train_at(rows[:20],{},'electricity',now)
        self.assertIsNone(model);self.assertFalse(retrieve(model,record,[0.]*12)['ready'])
        with self.assertRaises(ValueError):retrieve(model,record,[float('nan')]*12)


if __name__=='__main__':unittest.main()
