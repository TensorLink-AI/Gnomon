import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_seed_capsule as module


def fixture(root):
    source = root/'parent'; package = source/'benchmarks/hermes_ml_checkpoint_v6'
    package.mkdir(parents=True)
    values = {
        'worker.py': "request_overrides={'temperature':0.2,'seed':7}\n",
        'transport.py': 'payload.update(model=MODEL, temperature=0.2, seed=7, max_tokens=3072, stream=False)\n',
        'analyze.py': "expected={'temperature': .2, 'seed': 7, 'max_tokens': 3072, 'stream': False}\ncanary={'temperature':0,'seed':7,'max_tokens':16,'stream':False}\n",
        'service_admission.py': "canary={'seed':7}\n",
        'numerical.py': 'random_state=17\n', 'PROTOCOL.md': 'Original protocol.\n',
    }
    for name, text in values.items(): (package/name).write_text(text)
    contract = b'{"synthetic":true}\n'; (source/'cohort-contract.json').write_bytes(contract)
    manifest = {'sources': {n: module.digest(t.encode()) for n,t in values.items()},
                'cohort_contract_sha256':module.digest(contract),'common_to_all_arms':True,
                'provider_calls':0,'engy_calls':0}
    (source/'capsule.json').write_text(json.dumps(manifest))
    return source,module.digest((source/'capsule.json').read_bytes())


class SeedCapsuleTests(unittest.TestCase):
    def test_exact_change_set_preserves_canary_numerical_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source,digest = fixture(root)
            before = {str(p.relative_to(source)):module.digest(p.read_bytes()) for p in source.rglob('*') if p.is_file()}
            with patch.object(module,'PARENT_SHA256',digest):
                seed7 = module.build(source,root/'seven',7)
                seed19 = module.build(source,root/'nineteen',19)
                repeat = module.build(source,root/'again',19)
            self.assertEqual(seed7['seed_changed_files'],[])
            self.assertEqual(seed19['seed_changed_files'],['PROTOCOL.md','analyze.py','transport.py','worker.py'])
            self.assertEqual(seed19,repeat)
            self.assertEqual(seed7['sources']['numerical.py'],seed19['sources']['numerical.py'])
            self.assertEqual(seed7['sources']['service_admission.py'],seed19['sources']['service_admission.py'])
            text=(root/'nineteen/benchmarks/hermes_ml_checkpoint_v6/analyze.py').read_text()
            self.assertIn("'seed': 19",text);self.assertIn("'seed':7",text)
            self.assertFalse(seed19['execution_authorized'])
            self.assertEqual(before,{str(p.relative_to(source)):module.digest(p.read_bytes()) for p in source.rglob('*') if p.is_file()})

    def test_invalid_seed_changed_parent_or_contract_reject_before_output(self):
        for kind in ('seed','manifest','worker','contract','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source,digest=fixture(root);package=source/'benchmarks/hermes_ml_checkpoint_v6'
                if kind=='manifest':(source/'capsule.json').write_text('{}')
                if kind=='worker':(package/'worker.py').write_text('changed=1\n')
                if kind=='contract':(source/'cohort-contract.json').write_text('{}')
                if kind=='symlink':(package/'extra.py').symlink_to(package/'worker.py')
                with patch.object(module,'PARENT_SHA256',digest),self.assertRaises(ValueError):
                    module.build(source,root/'out',True if kind=='seed' else 19)
                self.assertFalse((root/'out').exists())

    def test_overwrite_and_nested_output_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,digest=fixture(root)
            with patch.object(module,'PARENT_SHA256',digest):
                module.build(source,root/'out',19)
                for output in (root/'out',source/'nested'):
                    with self.assertRaises(ValueError):module.build(source,output,19)


if __name__=='__main__':unittest.main()
