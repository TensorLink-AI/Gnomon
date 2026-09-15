from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.ledger_optimization import launch_planning_107 as launch


class AdmissionTests(unittest.TestCase):
    def proof(self):
        return {'passed': True, 'engy_calls': 0, 'synthetic_sessions': 312,
                'retained_pilot_sessions': 72, 'new_continuation_sessions': 240,
                'numerical_attempts': 2496, 'scripted_responses': 1248,
                'final_gate_opened': False, 'host_sources': {'host.py': 'a'*64},
                'pilot_terminal_sha256': 'b'*64, 'complete_terminal_sha256': 'c'*64}

    def test_complete_proof_contract(self):
        proof = self.proof()
        launch.validate_host_proof(proof, proof['host_sources'])

    def test_pilot_only_cannot_authorize_paid_run(self):
        proof = self.proof(); proof['synthetic_sessions'] = 72
        with self.assertRaisesRegex(ValueError, r'72\+240'):
            launch.validate_host_proof(proof, proof['host_sources'])

    def test_changed_host_source_rejected(self):
        proof = self.proof()
        with self.assertRaisesRegex(ValueError, 'exact source'):
            launch.validate_host_proof(proof, {'host.py': 'z'*64})

    def test_missing_continuation_archive_rejected(self):
        proof = self.proof(); proof.pop('complete_terminal_sha256')
        with self.assertRaisesRegex(ValueError, 'terminal'):
            launch.validate_host_proof(proof, proof['host_sources'])

    def test_boolean_cannot_replace_numerical_count(self):
        proof = self.proof(); proof['engy_calls'] = False
        with self.assertRaises(ValueError): launch.validate_host_proof(proof, proof['host_sources'])

    def test_no_automatic_second_reservation_or_worker(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); binding = {'plan_sha256': 'a'*64, 'output': '/example'}
            launch.reserve('pilot', binding, registry=root)
            with self.assertRaises(FileExistsError): launch.reserve('pilot', binding, registry=root)
            launch.consume('pilot', binding, registry=root)
            with self.assertRaises(FileExistsError): launch.consume('pilot', binding, registry=root)

    def test_changed_binding_cannot_consume_reserved_worker(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); binding = {'plan_sha256': 'a'*64, 'output': '/example'}
            launch.reserve('pilot', binding, registry=root)
            other = deepcopy(binding); other['output'] = '/changed'
            with self.assertRaisesRegex(ValueError, 'differs'):
                launch.consume('pilot', other, registry=root)
            self.assertFalse((root/'pilot/worker.json').exists())

    def test_pilot_reservation_cannot_be_used_for_continuation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); binding = {'plan_sha256': 'a'*64}
            launch.reserve('pilot', binding, registry=root)
            with self.assertRaises(FileNotFoundError): launch.consume('complete', binding, registry=root)


if __name__ == '__main__': unittest.main()
