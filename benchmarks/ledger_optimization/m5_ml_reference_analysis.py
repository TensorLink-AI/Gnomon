"""Historical four-arm experiment; not the active corrected three-arm protocol.

No I/O, dispatch, outcome lookup or declaration that the goal has been achieved.
The three-arm helper defines the active numerical comparison.
"""
import hashlib
import json
import re

from .m5_ml_analysis import _analyze

ARMS=('plain','gnomon','ledger','ledger_reference')


def analyze(panel, seeds, rows, *, reference_identity):
    """Require all 4,992 decisions and an explicit frozen prior-ledger identity.

    The identity is a caller declaration, not proof that those sources executed.
    Independent runtime, source, information/budget parity, scoring and temporal
    audits remain mandatory before any final efficacy claim.
    """
    required={'label','policy_sha256','runtime_version'}
    if type(reference_identity) is not dict or set(reference_identity)!=required:
        raise ValueError('Frozen reference label, policy_sha256 and runtime_version required')
    if (type(reference_identity['label']) is not str or not reference_identity['label'].strip()
            or type(reference_identity['policy_sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}',reference_identity['policy_sha256']) is None
            or reference_identity['runtime_version']!='1.2.0'):
        raise ValueError('Explicit reference policy hash and user-requested 1.2.0 runtime required')
    result=_analyze(panel,seeds,rows,arms=ARMS)
    ref=result['contrasts']['ledger_vs_ledger_reference']
    # The requested 20% threshold applies to no-ledger. For prior ledger the
    # objective requires improvement, rather than an additional 20% threshold.
    ref.pop('point_target_met')
    means=result['mean_per_case_rmsle']
    ref['point_improvement_met']=means['ledger'] < means['ledger_reference']
    ref['required_relative_reduction']='strictly positive'
    result['numerical_all_objective_criteria_met']=(result['numerical_primary_criteria_met'] and ref['point_improvement_met'])
    result['reference_identity']=dict(reference_identity)
    result['reference_uncertainty_requirement']='Interval reported; prior-ledger point improvement required. Primary no-ledger interval must exclude zero.'
    result['numerical_input_sha256']=result['analysis_input_sha256']
    result['analysis_input_sha256']=hashlib.sha256(json.dumps(
        {'rows_sha256':result['numerical_input_sha256'],'reference_identity':reference_identity},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    result['scope']='Four-arm numerical analysis only; identity declarations, untouched status, provenance, fairness and costs require independent audits.'
    result['active_protocol_eligible'] = False
    assert result['target_established'] is False
    return result
