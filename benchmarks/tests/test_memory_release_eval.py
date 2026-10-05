"""The release comparison must preserve cohorts and domain metrics."""
import pytest
np = pytest.importorskip('numpy')
from benchmarks.memory_release_eval.run import policy_for
from benchmarks.memory_release_eval.summarize import paired, score


def test_favorita_metric_weights_series_equally_not_by_units():
    base=np.array([[1.,100.],[1.,100.]])
    values=np.array([[.5,100.],[.5,100.]])
    assert score(values,base,relative=True)==.75
    assert score(values,base)==50.25


def test_paired_comparison_rejects_changed_targets_or_cohorts():
    old=[{'origin':str(i),'series_id':'s','served':'a','served_loss':2.,'losses':{'base':2.,'a':1.}} for i in range(8)]
    new=[{**d,'served':'base','served_loss':1.} for d in old]
    report=paired(old,new,block=2,draws=20)
    assert report['percent_change']==-50
    assert report['ci95_time_block']==[-1.,-1.]
    with pytest.raises(ValueError,match='same scored'):
        paired(old,new[:-1],draws=2)
    with pytest.raises(ValueError,match='forecasts or targets'):
        paired(old,[{**new[0],'losses':{'base':3.,'a':1.}},*new[1:]],draws=2)


def test_crypto_profile_is_a_redundant_control_and_no_other_options_are_enabled():
    raw={'baseline':'base','candidates':['a'],'memory':{'long_window':56,'short_window':7}}
    assert policy_for(raw,'rv_1d','current')==policy_for(raw,'rv_1d','v2_profile')
    fase=policy_for(raw,'rv_1d','fase')
    assert fase['memory']['long_window']==56
    assert fase['memory']['k']==10  # profile default when caller has not specified k
    assert not fase['memory']['shrinkage']
    assert not fase['memory']['novelty_threshold']
    assert 'switch_penalty' not in fase
