"""Budget and recovery checks for the prospective structured-tool worker."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.ledger_optimization.worker_093 import drive


class WorkerPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.work=self.root/'work';self.work.mkdir()
        self.output=self.root/'output';self.output.mkdir()
        (self.work/'TASK.md').write_text('Synthetic structured-tool task.')
        self.budget={'remaining_requests':16,'deadline_epoch':10**12,'forwarded_requests':0}
        self.save();self.calls=[]

    def save(self):
        (self.work/'agent-budget.json').write_text(json.dumps(self.budget))

    def factory(self,remaining,seconds):
        self.assertEqual(remaining,self.budget['remaining_requests'])
        def converse(**kwargs):
            self.calls.append(kwargs)
            self.budget['remaining_requests']-=1;self.budget['forwarded_requests']+=1;self.save()
            return {'turn_exit_reason':'text_response','messages':[{'role':'assistant','content':'prior'}]}
        return SimpleNamespace(run_conversation=converse)

    def test_correction_uses_structured_tools_and_preserves_history_budget(self):
        checks=iter([False,True])
        result=drive(self.factory,self.work,self.output,is_complete=lambda _:next(checks))
        self.assertEqual(result['stop_reason'],'checkpoint_workflow_complete')
        self.assertEqual(result['corrections'],1);self.assertEqual(result['final_budget']['remaining_requests'],14)
        self.assertEqual(self.calls[1]['conversation_history'],[{'role':'assistant','content':'prior'}])
        self.assertIn('actual lab tool',self.calls[1]['user_message'])
        self.assertNotIn('terminal',self.calls[1]['user_message'])

    def test_at_most_two_corrections_without_resetting_budget(self):
        result=drive(self.factory,self.work,self.output,is_complete=lambda _:False)
        self.assertEqual(result['stop_reason'],'correction_limit_reached')
        self.assertEqual(result['attempts'],3);self.assertEqual(result['corrections'],2)
        self.assertEqual(result['final_budget']['remaining_requests'],13)

    def test_last_request_does_not_trigger_an_extra_continuation(self):
        self.budget['remaining_requests']=1;self.save()
        result=drive(self.factory,self.work,self.output,is_complete=lambda _:False)
        self.assertEqual(result['stop_reason'],'request_budget_exhausted');self.assertEqual(len(self.calls),1)

    def test_deadline_stops_before_agent_creation(self):
        self.budget['deadline_epoch']=0;self.save()
        result=drive(self.factory,self.work,self.output,is_complete=lambda _:False)
        self.assertEqual(result['stop_reason'],'deadline_reached');self.assertFalse(self.calls)

    def test_runtime_failure_is_not_retried_as_agent_repair(self):
        factory=lambda *args:SimpleNamespace(run_conversation=lambda **kwargs:{'failed':True,'error':'provider timeout'})
        result=drive(factory,self.work,self.output,is_complete=lambda _:False)
        self.assertEqual(result['stop_reason'],'incomplete_without_recoverable_termination')
        self.assertEqual(result['attempts'],1);self.assertEqual(result['corrections'],0)


if __name__=='__main__':unittest.main()
