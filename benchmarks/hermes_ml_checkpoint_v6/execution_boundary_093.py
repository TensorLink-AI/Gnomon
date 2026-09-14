"""Prototype host-owned execution boundary for a future fair-budget trial.

No paid run uses this yet. All model work is dispatched to the verified lab;
there is no arbitrary shell/Python tool. Pure data inspection never fits models.
A host must intercept *all* Hermes dispatch paths, keep this object inaccessible
to agent writes, and start clean workspaces. This is not an OS security sandbox.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import threading


class BoundaryRejected(ValueError):
    pass


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise BoundaryRejected(f'{name} must be an integer in [{low}, {high}].')
    return value


def fields(args, allowed, required=()):
    if type(args) is not dict:
        raise BoundaryRejected('Arguments must be an object.')
    if set(args) - set(allowed) or set(required) - set(args):
        raise BoundaryRejected('Unexpected or missing fields; use the declared tool schema.')


class LabBoundary:
    """Equal-arm tool dispatcher. Only the host provides paths, hashes and runner."""

    BOOTSTRAP = ("import runpy,sys; p=sys.argv.pop(1); sys.path.insert(0,p); "
                 "sys.argv[0]=p+'/lab.py'; runpy.run_path(sys.argv[0],run_name='__main__')")
    OPERATIONS = {'start', 'status', 'review', 'backtest', 'commit', 'sync'}

    def __init__(self, work, python, protected_hashes, *, runner=subprocess.run):
        self.work = Path(work).resolve(strict=True)
        # Python discovers pyvenv.cfg from the invoked path. Resolving the
        # executable symlink first silently leaves a virtual environment.
        candidate = Path(os.path.abspath(python))
        candidate.resolve(strict=True)  # Validate existence, retain venv identity.
        self.python = str(candidate)
        self.protected = dict(protected_hashes)
        if 'lab.py' not in self.protected:
            raise ValueError('Host manifest must include lab.py and every imported project module.')
        self.runner = runner
        self.lock = threading.RLock()
        self.events = []
        self.verify_sources()

    def path(self, name):
        if not isinstance(name, str) or not name or '\x00' in name:
            raise BoundaryRejected('A relative project path is required.')
        p = Path(name)
        if p.is_absolute() or any(x in ('..', '') for x in p.parts):
            raise BoundaryRejected('Only paths inside the current project are readable.')
        cursor = self.work
        for part in p.parts:
            cursor /= part
            if cursor.is_symlink():
                raise BoundaryRejected('Symlink paths are not permitted.')
        return cursor

    def read_bytes(self, name, maximum=32 * 1024 * 1024):
        p = self.path(name)
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
                raise BoundaryRejected('Expected a bounded regular project file.')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                return stream.read(maximum + 1)
        finally:
            os.close(fd)

    def verify_sources(self):
        for name, expected in self.protected.items():
            if hashlib.sha256(self.read_bytes(name)).hexdigest() != expected:
                raise BoundaryRejected('Protected project content changed; execution refused.')
        # Disallow unmanifested import shadowing, even from a reused workspace.
        for p in self.work.rglob('*'):
            if p.is_symlink():
                raise BoundaryRejected('Project symlinks are not permitted.')
            if p.suffix in {'.py', '.pyc', '.pth', '.so'} and str(p.relative_to(self.work)) not in self.protected:
                raise BoundaryRejected('Unmanifested executable project content.')

    def dispatch(self, tool, arguments):
        with self.lock:  # Same serialization for single and multi-tool batches.
            self.execution_started = False
            try:
                handler = {'lab': self.lab, 'evidence_read': self.evidence_read,
                           'project_list': self.project_list, 'notes_write': self.notes_write,
                           'data_summary': self.data_summary}.get(tool)
                if handler is None:
                    raise BoundaryRejected('Only declared evidence tools and metered lab operations may execute.')
                result = handler(arguments)
                self.events.append({'tool': tool, 'admitted': True})
                return {'status': 'ok', 'result': result}
            except (BoundaryRejected, OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
                self.events.append({'tool': tool, 'admitted': False, 'cause_type': type(error).__name__})
                return {'status': 'error', 'error': {'code': 'EXECUTION_BOUNDARY_REJECTED',
                        'message': str(error), 'next_call': {'name': 'lab', 'arguments': {'operation': 'status'}},
                        'next_call_scope': 'Inspect existing evidence; does not complete the original task.'},
                        'execution_started': self.execution_started}

    def lab(self, args):
        fields(args, {'operation', 'config', 'execution_id', 'offset', 'limit', 'pair'}, {'operation'})
        op = args['operation']
        if not isinstance(op, str) or op not in self.OPERATIONS:
            raise BoundaryRejected('Unsupported lab operation.')
        allowed = {'backtest': {'config'}, 'commit': {'config', 'execution_id'},
                   'review': {'offset', 'limit', 'pair'}}.get(op, set())
        if set(args) - {'operation'} - allowed:
            raise BoundaryRejected('Fields do not apply to this operation.')
        argv = [op]
        if 'config' in args:
            if type(args['config']) is not dict:
                raise BoundaryRejected('config must be an object.')
            argv += ['--config', json.dumps(args['config'], allow_nan=False)]
        if op == 'backtest' and 'config' not in args:
            raise BoundaryRejected('Backtest requires an explicit configuration.')
        for key in ('execution_id',):
            if key in args:
                v = args[key]
                if not isinstance(v, str) or not v or len(v) > 256 or v.startswith('-'):
                    raise BoundaryRejected(f'{key} must be a bounded identifier.')
                argv += ['--' + key.replace('_', '-'), v]
        if 'pair' in args:
            pair = args['pair']
            if (not isinstance(pair, list) or len(pair) != 2 or
                    any(not isinstance(v, str) or not v or len(v) > 256 or v.startswith('-') for v in pair)):
                raise BoundaryRejected('pair must contain exactly two bounded configuration IDs from review cards.')
            argv += ['--pair', *pair]
        for key, lo, hi in [('offset', 0, 1000000), ('limit', 1, 100)]:
            if key in args:
                argv += ['--' + key, str(integer(args[key], lo, hi, key))]
        self.verify_sources()
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
               'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        self.execution_started = True
        result = self.runner([self.python, '-I', '-B', '-c', self.BOOTSTRAP, str(self.work), *argv],
                             cwd=self.work, env=env, text=True, capture_output=True, timeout=60)
        self.verify_sources()
        return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
                'execution_scope': 'verified_metered_lab'}

    def evidence_read(self, args):
        fields(args, {'path', 'offset', 'max_chars'}, {'path'})
        offset = integer(args.get('offset', 0), 0, 100000000, 'offset')
        size = integer(args.get('max_chars', 4096), 1, 16384, 'max_chars')
        raw = self.read_bytes(args['path']);text = raw.decode('utf-8')
        end = min(len(text), offset + size)
        return {'path': args['path'], 'text': text[offset:end], 'total_chars': len(text),
                'next_offset': end if end < len(text) else None, 'sha256': hashlib.sha256(raw).hexdigest()}

    def project_list(self, args):
        fields(args, set())
        return {'files': sorted(str(p.relative_to(self.work)) for p in self.work.rglob('*')
                                if p.is_file() and not p.is_symlink())}

    def notes_write(self, args):
        fields(args, {'path', 'text'}, {'path', 'text'})
        name = args['path'];p = self.path(name);text = args['text']
        note = Path(name)
        if not (name == 'decision.json' or (len(note.parts) == 2 and note.parts[0] == 'notes' and note.suffix == '.md')):
            raise BoundaryRejected('Write only decision.json or notes/*.md; executable and evidence files are protected.')
        if not isinstance(text, str) or len(text.encode('utf-8')) > 16384:
            raise BoundaryRejected('Note must be UTF-8 text of at most 16384 bytes.')
        if name == 'decision.json' and type(json.loads(text)) is not dict:
            raise BoundaryRejected('Decision summary must be a JSON object.')
        if name in self.protected:
            raise BoundaryRejected('Protected files are not writable.')
        p.parent.mkdir(exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BoundaryRejected('Notes must be regular files with no hard links.')
            os.ftruncate(fd, 0)
            with os.fdopen(fd, 'w', closefd=False) as f:f.write(text)
        finally:os.close(fd)
        return {'path': name, 'bytes_written': len(text.encode('utf-8'))}

    def data_summary(self, args):
        fields(args, {'column', 'start_row', 'end_row', 'group_by'}, {'column'})
        rows = list(csv.DictReader(io.StringIO(self.read_bytes('history.csv').decode('utf-8'))))
        start = integer(args.get('start_row', 0), 0, len(rows), 'start_row')
        end = integer(args.get('end_row', len(rows)), start, len(rows), 'end_row')
        column = args['column'];group = args.get('group_by')
        if column not in (rows[0] if rows else {}) or (group is not None and group not in rows[0]):
            raise BoundaryRejected('Choose exact available history columns.')
        groups = {}
        for row in rows[start:end]:
            value = float(row[column])
            if not math.isfinite(value):raise BoundaryRejected('Summary requires finite numeric values.')
            groups.setdefault(row[group] if group else '__all__', []).append(value)
        return {'source': 'history.csv', 'start_row': start, 'end_row': end,
                'column': column, 'group_by': group, 'model_fits': 0,
                'groups': {k: {'n': len(v), 'mean': statistics.mean(v), 'minimum': min(v),
                               'maximum': max(v), 'zero_count': v.count(0)} for k, v in groups.items()}}
