import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmarks.ledger_optimization.contrast_plan_100 import freeze, verify, sha

PARENT = Path('results/workflow-097-prospective-plan-001/plan.json')
CAPSULE = Path('results/contrast-capsule-100-offline-003/capsule')
WORKER = Path('results/contrast-capsule-100-worker-003')
TASKS = Path('results/ledger-ml-continuous-022/export-002/host-jobs.json')


@unittest.skipUnless(all(p.exists() for p in (PARENT,CAPSULE,WORKER,TASKS)), 'requires retained independent synthetic evidence')
class ContrastPlanTests(unittest.TestCase):
    def test_freeze_preserves_cohort_budgets_gates_and_never_dispatches(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'plan.json'
            plan = freeze(PARENT,CAPSULE,WORKER,TASKS,out)
            parent = json.loads(PARENT.read_text())
            for field in ('arms','cases','seed','agent','gnomon','planned','budgets','pilot_gate','final_target','final_gate_opened','task_source_sha256'):
                self.assertEqual(plan[field],parent[field])
            self.assertEqual(plan['engy_calls'],0)
            self.assertTrue(plan['final_baseline_requirement']['improvement_over_frozen_pre_optimization_ledger_required'])
            self.assertFalse(plan['final_baseline_requirement']['historical_run_scores_sufficient'])
            self.assertFalse(plan['comparison']['old_097_controls_allowed'])
            self.assertFalse(plan['amendment']['old_run_predictions_and_memories_reused'])
            self.assertEqual(plan['comparison']['primary'],'ledger versus gnomon within the fresh candidate-100 run')
            with self.assertRaisesRegex(ValueError,'Fresh plan'): freeze(PARENT,CAPSULE,WORKER,TASKS,out)

    def test_changed_parent_or_tasks_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); parent=root/'parent.json'; shutil.copyfile(PARENT,parent)
            d=json.loads(parent.read_text());d['budgets']['model_requests']=32;parent.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'frozen 097'): freeze(parent,CAPSULE,WORKER,TASKS,root/'out')
            tasks=root/'tasks.json';tasks.write_bytes(TASKS.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'tasks changed'): freeze(PARENT,CAPSULE,WORKER,tasks,root/'out')
            self.assertFalse((root/'out').exists())

    def test_changed_capsule_module_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); capsule=root/'capsule';shutil.copytree(CAPSULE,capsule)
            module=capsule/'benchmarks/hermes_ml_checkpoint_v6/core.py'
            original=module.read_bytes();module.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError,'sources changed'): verify(PARENT,capsule,WORKER,TASKS)
            module.write_bytes(original);(module.parent/'alias').symlink_to(module)
            with self.assertRaisesRegex(ValueError,'Unexpected capsule'): verify(PARENT,capsule,WORKER,TASKS)

    def test_forged_pass_or_report_or_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('passed.json','report.json','manifest.json'):
                shutil.copyfile(WORKER/name,root/name)
            for name in ('passed.json','report.json','manifest.json'):
                p=root/name; original=p.read_bytes();p.write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError,'proof required|report or runtime changed'): verify(PARENT,CAPSULE,root,TASKS)
                p.write_bytes(original)
            verify(PARENT,CAPSULE,root,TASKS)


if __name__=='__main__': unittest.main()
