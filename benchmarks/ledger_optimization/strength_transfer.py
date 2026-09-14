"""Descriptive rank transfer and finite-candidate hindsight bound; no policy."""
from itertools import combinations
import math

KEYS=('0','0.25','0.5','0.75','1')
TOLERANCE=1e-12


def score(point,actual):
    if len(point)!=24 or len(actual)!=24 or any(not math.isfinite(v) or v<0 for v in list(point)+list(actual)):raise ValueError('Complete finite nonnegative24-step pair required')
    return math.sqrt(math.fsum((math.log1p(p)-math.log1p(y))**2 for p,y in zip(point,actual,strict=True))/24)


def pair_counts(historical,realized):
    if set(historical)!=set(KEYS) or set(realized)!=set(KEYS) or any(not math.isfinite(v) for v in list(historical.values())+list(realized.values())):raise ValueError('Complete finite five-candidate risks required')
    counts=dict.fromkeys(('concordant','discordant','historical_only_tie','realized_only_tie','both_tied'),0)
    def sign(v):return 0 if abs(v)<=TOLERANCE else 1 if v>0 else -1
    for a,b in combinations(KEYS,2):
        h=sign(historical[a]-historical[b]);r=sign(realized[a]-realized[b])
        label='both_tied' if h==r==0 else 'historical_only_tie' if h==0 else 'realized_only_tie' if r==0 else 'concordant' if h==r else 'discordant'
        counts[label]+=1
    return counts


def case_metrics(historical,realized,selected):
    counts=pair_counts(historical,realized)
    if selected not in KEYS:raise ValueError('Grid selection required')
    best=min(realized.values());delta=realized['0.5']-realized[selected]
    return {'pair_counts':counts,'hindsight_minimum':best,'historical_advantage_vs_half':historical['0.5']-historical[selected],
        'realized_advantage_vs_half':delta,'realized_comparison':'tied' if abs(delta)<=TOLERANCE else 'improved' if delta>0 else 'worse',
        'selected_regret':realized[selected]-best,'half_regret':realized['0.5']-best}


def summarize(rows):
    if not rows:raise ValueError('Nonempty cohort required')
    def avg(values):return math.fsum(values)/len(rows)
    means={k:avg(r['realized'][k] for r in rows) for k in KEYS};means.update({k:avg(r[k] for r in rows) for k in ('selected','control','strong_block_cv')});means['hindsight_minimum']=avg(r['metrics']['hindsight_minimum'] for r in rows)
    counts={k:sum(r['metrics']['pair_counts'][k] for r in rows) for k in rows[0]['metrics']['pair_counts']};untied=counts['concordant']+counts['discordant']
    return {'cases':len(rows),'mean_rmsle':means,'hindsight_reduction':{k:1-means['hindsight_minimum']/means[k] for k in ('control','strong_block_cv','0.5')},
        'finite_family_can_reach_twenty_percent':all(means['hindsight_minimum']<=.8*means[k] for k in ('control','strong_block_cv')),
        'pair_counts':counts,'untied_pair_concordance':counts['concordant']/untied if untied else None,
        'mean_differences':{k:avg(r['metrics'][k] for r in rows) for k in ('historical_advantage_vs_half','realized_advantage_vs_half','selected_regret','half_regret')},
        'selected_vs_half_counts':{k:sum(r['metrics']['realized_comparison']==k for r in rows) for k in ('improved','worse','tied')}}
