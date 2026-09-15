"""Freeze candidate 100 on verified synthetic evidence; never dispatch a run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

PARENT_PLAN_SHA = '1398c649da651441d77a34ea3d5ea86ac5838170bb55c0d4690101b26a0b2665'
CAPSULE_SHA = 'ea0920fd92a5e888816a93ef3ef95c88dcea8482bd5a541bc987c7f78c082519'
WORKER_PROOF_SHA = '1cfa707163c4686bc23665908d4aed80041d77afdf63d8be76addf88ac6c651c'

WORKER_EVIDENCE_SHA = {'report.json': '2858f69311aa067ed1fd9c9e03684b448188a74dc8ad73f2dd8cf566a74ffe01', 'manifest.json': '47a6f6b87c0c2b466f85e1131f688f2fd4013d1a8ddc71c91e6ddab8e2b890f0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(parent_plan, capsule, worker, task_source):
    parent_plan, capsule, worker, task_source = map(Path, (parent_plan, capsule, worker, task_source))
    if sha(parent_plan) != PARENT_PLAN_SHA: raise ValueError('Exact frozen 097 plan required')
    parent = read(parent_plan)
    if sha(capsule/'capsule.json') != CAPSULE_SHA: raise ValueError('Exact independently audited candidate 100 capsule required')
    manifest = read(capsule/'capsule.json'); package = capsule/'benchmarks/hermes_ml_checkpoint_v6'
    if any(p.is_symlink() or (p.is_dir() and p.name != '__pycache__') for p in package.iterdir()):
        raise ValueError('Unexpected capsule entry')
    sources = {p.name: sha(p) for p in package.iterdir() if p.is_file()}
    if sources != manifest['sources']: raise ValueError('Capsule sources changed')
    if manifest['parent_sources'] != parent['capsule']['sources']: raise ValueError('Capsule parent mismatch')
    if sha(task_source) != parent['task_source_sha256']: raise ValueError('Development tasks changed')
    if sha(worker/'passed.json') != WORKER_PROOF_SHA: raise ValueError('Exact synthetic worker proof required')
    if any(sha(worker/name) != expected for name, expected in WORKER_EVIDENCE_SHA.items()):
        raise ValueError('Synthetic worker report or runtime changed')
    proof, report, runtime = read(worker/'passed.json'), read(worker/'report.json'), read(worker/'manifest.json')
    expected = {(a, 'synthetic-collection-worker', n) for a in parent['arms'] for n in (0, 1)}
    rows = report['rows']
    if (proof['sources'] != sources or runtime['sources'] != sources
            or proof['passed'] is not True or proof['engy_calls'] != 0
            or not proof['checks'] or not all(c['passed'] is True for c in proof['checks'])
            or report['audit_failures'] or report['shutdown_record_gaps']
            or len(rows) != 6 or {(r['arm'], r['series_id'], r['round']) for r in rows} != expected
            or not all(r['valid'] and r['workflow_complete'] and r['contrast_annotation_audit']['passed'] for r in rows)
            or sum(r['contrast_annotation_audit']['annotations'] for r in rows) != 36
            or sum(r['contrast_annotation_audit']['checks'] for r in rows) != 3174
            or report['audit_checks'] != 3859):
        raise ValueError('Complete integrated synthetic audit required')
    if runtime['build']['package_version'] != '1.2.0': raise ValueError('Published 1.2.0 runtime required')
    return parent, manifest, runtime


def freeze(parent_plan, capsule, worker, task_source, output):
    output = Path(output)
    if output.exists() or output.is_symlink(): raise ValueError('Fresh plan path required')
    parent, manifest, runtime = verify(parent_plan, capsule, worker, task_source)
    plan = deepcopy(parent)
    plan.update(candidate='100_current_history_contrast', capsule=manifest, capsule_sha256=CAPSULE_SHA,
        parent_plan_sha256=PARENT_PLAN_SHA,
        status='frozen_development_plan_not_dispatch_authorization',
        preflight={'worker_proof_sha256': WORKER_PROOF_SHA, 'worker_report_sha256': sha(Path(worker)/'report.json'),
                   'runtime_manifest_sha256': sha(Path(worker)/'manifest.json'),
                   'runtime_inventory': runtime['inventory'], 'engy_calls': 0},
        amendment={'reason': 'Development decisions sometimes confuse current CV and sparse older production evidence.',
                   'common_current_cv': True, 'historical_overlay_arm': 'ledger',
                   'all_current_complete_configurations_visible': True,
                   'only_latest_requested_task_matching_review': True,
                   'additional_fits': 0, 'additional_history_queries': 0,
                   'forecast_selection_by_agent': True,
                   'no_accuracy_dependent_gate': True,
                   'old_run_predictions_and_memories_reused': False,
                   'prior_097_excluded_from_100_scores': True},
        launch_prerequisites=[
            'Original 097 controller and worker verified terminal by PID/start ticks/boot identity.',
            '097 complete 312-session archive, costs, source hashes and independent audit preserved; no audit failures or shutdown gaps.',
            'Exact candidate source, runtime, worker proof and new launch/continuation checks pass.',
            'Fresh arm homes and output directory; no prior experimental memory or execution state.'],
        final_baseline_requirement={
            'improvement_over_frozen_pre_optimization_ledger_required': True,
            'historical_run_scores_sufficient': False,
            'runtime_for_new_experiments': '1.2.0',
            'note': 'The original goal names the 1.1.9 ledger baseline; later user direction requires runtime 1.2.0. Final baseline behavior and its fair integration must be frozen separately, not inferred from old scores.'},
        efficacy_scope='Reused development cohort. Only fresh matched candidate-100 arms establish within-run contrasts. No held-out claim or outcome-dependent continuation.',
        development_decision_rules={
            'continuation': 'Use inherited completion/audit pilot gate only; retain its 36 sessions exactly once.',
            'accuracy': 'Report all-case RMSLE and paired exploratory uncertainty, including failures. No optional stopping on accuracy.',
            'costs': 'Report all attempts, unknown usage, readiness calls, tokens, fits and elapsed time; no zero imputation for missing usage.',
            'comparison_with_097': 'Descriptive prior-run reference, not a randomized estimate of the new overlay effect.',
            'next_candidate': 'Freeze a new plan before any change; do not alter this running protocol.',
            'final_admission': 'Separate frozen final protocol and baseline comparison required; this plan never opens the final gate.'})
    # These requirements intentionally survive the amendment byte-for-byte.
    for field in ('arms', 'cases', 'seed', 'agent', 'gnomon', 'planned', 'budgets', 'pilot_gate', 'comparison', 'final_target', 'final_gate_opened', 'task_source_sha256'):
        if plan[field] != parent[field]: raise AssertionError('Changed fixed comparison: '+field)
    plan['comparison']['primary'] = 'ledger versus gnomon within the fresh candidate-100 run'
    plan['comparison']['secondary'] = 'ledger versus plain within the fresh candidate-100 run'
    plan['comparison']['old_097_controls_allowed'] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(plan, stream, indent=2); stream.write('\n')
    return plan
