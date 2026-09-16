from datetime import timedelta
import json
import pytest
from .data import START,FIRST_ORIGIN,END,cohort,origin
from .queue import predecessor_status


def test_43_complete_periods_and_84_day_first_cv_training():
    assert (FIRST_ORIGIN-START).days+1==126
    assert 126-42==84
    assert origin(42)+timedelta(days=14)==END
    assert origin(43)+timedelta(days=14)>END


def test_cohort_uses_only_warmup_not_later_success():
    data={}
    offsets=[[0,14,28,42,56,70,84,98,112,125],list(range(0,126,5)),list(range(126))]
    for group,days in enumerate(offsets):
        for i in range(16):data[str(10000+group*100+i)]={START+timedelta(days=d):3 for d in days}
    before=cohort(data)
    assert len(before[0])==48
    for values in data.values():values[END]=1e9
    data['99999']={END:1e9}
    assert cohort(data)==before
    with pytest.raises(ValueError,match='Insufficient'):cohort({'99999':{END:1e9}})


def test_queue_requires_exact_terminal_success(tmp_path):
    (tmp_path/'launch-001').mkdir();(tmp_path/'paid-001').mkdir()
    assert predecessor_status(tmp_path)=='waiting'
    exit=tmp_path/'launch-001/exit.json';finish=tmp_path/'paid-001/FINISHED.json'
    exit.write_text(json.dumps({'exit_code':0}))
    assert predecessor_status(tmp_path)=='blocked'
    finish.write_text(json.dumps({'sessions':3744,'stub':False}))
    assert predecessor_status(tmp_path)=='ready'
    exit.write_text(json.dumps({'exit_code':2}))
    assert predecessor_status(tmp_path)=='blocked'
