"""Independent075 calculation and complete source-identity audit."""
import hashlib,json,math
from pathlib import Path
import sys,time
import numpy as np

KEYS=('0','0.25','0.5','0.75','1')


def verify(root,source):
    root,source=Path(root),Path(source);here=Path(__file__).parent;start=time.monotonic();checks=0
    def load(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def check(v,m):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(m)
    def near(a,b):check(math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-10),'Numeric disagreement')
    def score(p,y):return float(np.linalg.norm(np.log1p(p)-np.log1p(y))/math.sqrt(24))
    manifest=load(root/'manifest.json');receipt_path=here/'evidence/memory-strength-074.json';receipt=load(receipt_path);check(sha(receipt_path)==manifest['source_receipt_sha256'],'Source receipt identity')
    for n,h in manifest['code_sha256'].items():check(sha(here/n)==h,'Frozen diagnostic identity')
    accessed=load(root/'source_access.json')
    for n,h in accessed.items():check(h==receipt['files'][n]==sha(source/n),'All accessed source hashes')
    rows=[]
    for path in sorted((source/'cases').glob('*.json')):
        old=load(path)
        if old['round']<0:continue
        tid=old['task_id'];r=load(root/(tid+'.json'));d=load(source/'decisions'/(tid+'.json'));record=d['records']['ledger'];historical=record['selection']['candidate_score_means'];realized={k:score(record['candidates'][k]['point'],old['actual']) for k in KEYS};key=record['selected_key']
        for k in ('task_id','series_id','domain','origin','round'):check(r[k]==old[k]==d[k],'Case identity')
        check(r['source_files']=={n:receipt['files'][n] for n in ('cases/'+tid+'.json','decisions/'+tid+'.json')},'Case source hashes');check(r['historical']==historical and r['selected_key']==key,'Unchanged historical selection');check(r['candidate_execution_ids']=={k:record['candidates'][k]['execution_id'] for k in KEYS},'Executed candidates')
        for k in KEYS:near(r['realized'][k],realized[k])
        near(r['selected'],realized[key])
        for k in ('control','strong_block_cv'):near(r[k],score(old['point'][k],old['actual']))
        counts=dict.fromkeys(('concordant','discordant','historical_only_tie','realized_only_tie','both_tied'),0)
        for i,a in enumerate(KEYS):
            for b in KEYS[i+1:]:
                hd=historical[a]-historical[b];rd=realized[a]-realized[b];ht=abs(hd)<=1e-12;rt=abs(rd)<=1e-12
                label='both_tied' if ht and rt else 'historical_only_tie' if ht else 'realized_only_tie' if rt else 'concordant' if hd*rd>0 else 'discordant';counts[label]+=1
        check(r['metrics']['pair_counts']==counts and sum(counts.values())==10,'Independent full pair classifications')
        minimum=min(realized.values());hd=historical['0.5']-historical[key];rd=realized['0.5']-realized[key]
        for k,v in {'hindsight_minimum':minimum,'historical_advantage_vs_half':hd,'realized_advantage_vs_half':rd,'selected_regret':realized[key]-minimum,'half_regret':realized['0.5']-minimum}.items():near(r['metrics'][k],v)
        check(r['metrics']['realized_comparison']==('tied' if abs(rd)<=1e-12 else 'improved' if rd>0 else 'worse'),'Gain direction');check(r['metrics']['selected_regret']>=0 and r['metrics']['half_regret']>=0,'Finite-set lower bound');rows.append(r)
    check(len(rows)==416,'Full scored cohort')
    report=load(root/'report.json')
    def summarize(group,saved):
        check(saved['cases']==len(group),'Group denominator');means={k:sum(r['realized'][k] for r in group)/len(group) for k in KEYS};means.update({k:sum(r[k] for r in group)/len(group) for k in ('selected','control','strong_block_cv')});means['hindsight_minimum']=sum(min(r['realized'].values()) for r in group)/len(group)
        for k,v in means.items():near(saved['mean_rmsle'][k],v)
        for k in ('control','strong_block_cv','0.5'):near(saved['hindsight_reduction'][k],1-means['hindsight_minimum']/means[k])
        check(saved['finite_family_can_reach_twenty_percent']==all(means['hindsight_minimum']<=.8*means[k] for k in ('control','strong_block_cv')),'Finite-set target feasibility')
        counts={k:sum(r['metrics']['pair_counts'][k] for r in group) for k in group[0]['metrics']['pair_counts']};check(saved['pair_counts']==counts and sum(counts.values())==10*len(group),'Pair aggregate');untied=counts['concordant']+counts['discordant']
        if untied:near(saved['untied_pair_concordance'],counts['concordant']/untied)
        else:check(saved['untied_pair_concordance'] is None,'No untied denominator')
        for k,v in saved['mean_differences'].items():near(v,sum(r['metrics'][k] for r in group)/len(group))
        check(saved['selected_vs_half_counts']=={k:sum(r['metrics']['realized_comparison']==k for r in group) for k in ('improved','worse','tied')},'Win/loss/tie aggregate')
    summarize(rows,report['overall'])
    for d in ('electricity','pedestrian'):summarize([r for r in rows if r['domain']==d],report['domains'][d])
    summarize([r for r in rows if r['round']<8],report['phases']['early_0_7']);summarize([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    for k in KEYS:
        group=[r for r in rows if r['selected_key']==k]
        if group:summarize(group,report['selection_groups'][k])
        else:check(k not in report['selection_groups'],'Empty groups omitted')
    oldreport=load(source/'report.json')
    for k,v in report['inherited074costs'].items():check(oldreport[k]==v,'Inherited costs retained')
    check(report['new_forecasts']==report['weight_fits']==report['api_calls']==0 and manifest['validation_or_final_access'] is False,'Diagnostic scope')
    result={'checks':checks,'failures':0,'cases':416,'candidate_risks':2080,'pair_comparisons':4160,'seconds':time.monotonic()-start,'new_forecasts':0,'api_calls':0,'verifier_sha256':sha(__file__),'scope':'Independent candidate risks, finite-set minimum, all rank-pair classifications, selected-versus-half changes, group summaries, source identities and costs.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
