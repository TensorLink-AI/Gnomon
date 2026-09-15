from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization.control_contrast_100 import FIELDS,main


class ControllerTests(unittest.TestCase):
    def argv(self,root,audit):
        argv=['controller','--launch-directory',str(root/'launch')]
        for field in FIELDS:argv+=['--'+field,str(audit if field=='predecessor-audit' else root/field)]
        return argv

    def test_recheck_evidence_must_be_in_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('sys.argv',self.argv(root,root/'unarchived')),patch('benchmarks.ledger_optimization.control_contrast_100.supervise') as run:
                with self.assertRaisesRegex(ValueError,'so it is archived'):main()
                run.assert_not_called()

    def test_valid_controller_uses_non_restarting_supervisor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('sys.argv',self.argv(root,root/'launch/recheck')),patch('benchmarks.ledger_optimization.control_contrast_100.supervise',return_value={'complete':True}) as run,patch('builtins.print'):
                main()
                run.assert_called_once()
                command=run.call_args.args[0]
                self.assertIn('benchmarks.ledger_optimization.launch_contrast_100',command)
                self.assertEqual(command[command.index('--predecessor-audit')+1],str(root/'launch/recheck'))
