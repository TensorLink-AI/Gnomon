import unittest
from benchmarks.ledger_optimization.ml_service_audit import classify


class ServiceAuditTest(unittest.TestCase):
    def test_all_service_failures(self):
        r=classify({'workflow_complete':False,'numerical_attempts':0},['service_error']*3)
        self.assertTrue(r['no_execution_service_failure'])
        self.assertTrue(r['service_interrupted_before_execution'])

    def test_one_response_then_outage_is_not_all_failed(self):
        r=classify({'workflow_complete':False,'numerical_attempts':0},['success']+['service_error']*3)
        self.assertFalse(r['no_execution_service_failure'])
        self.assertTrue(r['service_interrupted_before_execution'])
        self.assertEqual(r['successful_api_responses'],1)

    def test_other_errors_not_reclassified_as_service(self):
        r=classify({'workflow_complete':False,'numerical_attempts':0},['other_error']*3)
        self.assertFalse(r['no_execution_service_failure'])
        self.assertFalse(r['service_interrupted_before_execution'])

    def test_existing_numerical_work_not_service_only(self):
        r=classify({'workflow_complete':False,'numerical_attempts':60},['service_error']*3)
        self.assertFalse(r['service_interrupted_before_execution'])

    def test_complete_checkpoint_stays_complete_despite_service_errors(self):
        r=classify({'workflow_complete':True,'numerical_attempts':7},['success']+['service_error']*3)
        self.assertFalse(r['no_execution_service_failure'])
        self.assertFalse(r['service_interrupted_before_execution'])


if __name__=='__main__':unittest.main()
