import hashlib
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from benchmarks.ledger_optimization.build_planning_bundle_107 import verify_archive


class BundleIntegrityTests(unittest.TestCase):
    def check_archive(self, entries, expected):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'bundle.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                for name, body in entries:
                    member = tarfile.TarInfo(name)
                    if body is None:
                        member.type = tarfile.SYMTYPE; member.linkname = '/outside'
                        archive.addfile(member)
                    else:
                        member.size = len(body); archive.addfile(member, io.BytesIO(body))
            verify_archive(path, expected)

    def test_exact_bytes_pass(self):
        self.check_archive([('plan.json', b'frozen')], {'plan.json': hashlib.sha256(b'frozen').hexdigest()})

    def test_changed_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            self.check_archive([('plan.json', b'changed')], {'plan.json': hashlib.sha256(b'frozen').hexdigest()})

    def test_duplicate_member_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.check_archive([('plan.json', b'frozen')]*2, {'plan.json': hashlib.sha256(b'frozen').hexdigest()})

    def test_omitted_member_rejected(self):
        with self.assertRaisesRegex(ValueError, 'omits'):
            self.check_archive([], {'plan.json': hashlib.sha256(b'frozen').hexdigest()})

    def test_unlisted_file_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            self.check_archive([('unlisted.py', b'code')], {})

    def test_link_cannot_substitute_for_regular_file(self):
        with self.assertRaisesRegex(ValueError, 'non-regular'):
            self.check_archive([('plan.json', None)], {'plan.json': hashlib.sha256(b'frozen').hexdigest()})


if __name__ == '__main__': unittest.main()
