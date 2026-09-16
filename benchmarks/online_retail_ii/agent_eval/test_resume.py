import json
import pytest
from benchmarks.online_retail_ii.test_benchmark import case_fixture
from benchmarks.online_retail_ii.data import dump,sha
from benchmarks.online_retail_ii.models import metrics
from .resume import recover_rows


def test_recover_unaggregated_success_and_preserve_failed_score(tmp_path):
    case=tmp_path/'case';task=case_fixture(case)
    root=tmp_path/'run';root.mkdir();actuals={task['case_id']:[2]*14}
    rows=[]
    for seed in (7,19):
        row={'case_id':task['case_id'],'origin':task['origin'],'series_id':task['series_id'],
             'arm':'hermes','seed':seed,'point':[1]*14,'stub':False,'resolved':seed==19}
        out=root/f'seed-{seed}'/'hermes'/task['series_id']/task['origin'];out.mkdir(parents=True)
        dump(out/'result.json',row)
        dump(out/'inputs.json',{'hashes':{p.name:sha(p) for p in case.iterdir()}})
        rows.append({**row,'metrics':metrics(row['point'],[2]*14,[1]*200)})
    (root/'scores.jsonl').write_text(json.dumps(rows[0])+'\n')
    old,new=recover_rows(root,[case],actuals)
    assert old==rows[:1] and new==rows[1:]
    assert not old[0]['resolved']
    (root/'scores.jsonl').write_text(json.dumps(rows[0])+'\n'+json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='Duplicate'):recover_rows(root,[case],actuals)
