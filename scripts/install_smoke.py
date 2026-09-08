#!/usr/bin/env python3
"""Exercise managed environment replacement, dependency retention and live leases."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


def main():
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='gnomon-install-smoke-') as directory:
        root = Path(directory)
        command = root / 'bin/gnomon'
        env = {**os.environ, 'PIP_CACHE_DIR': str(root / 'pip-cache')}
        env.pop('GNOMON_LOCAL', None)
        env.pop('GNOMON_UPDATE_REF', None)

        def run(*args, check=True):
            result = subprocess.run(list(map(str, args)), text=True, capture_output=True, env=env)
            if check and result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            return result

        def cli(*args):
            return json.loads(run(command, *args).stdout)

        installer = ['bash', source / 'install.sh', '--local', '--install-root', root / 'installs',
                     '--bin-dir', root / 'bin']
        run(*installer)
        first = command.resolve().parent.parent
        build = cli('environment')['build']
        if build['commit'] and build['dirty'] is False:
            before = cli('releases')['releases']
            result = cli('update', '--version', build['commit'])
            assert not result['changed'] and result['reason'] == 'already_up_to_date'
            assert len(cli('releases')['releases']) == len(before)

        wheel = root / 'gnomon_smoke_extra-0.1.0-py3-none-any.whl'
        metadata = 'gnomon_smoke_extra-0.1.0.dist-info/'
        files = {'gnomon_smoke_extra.py': 'VALUE = 42\n',
                 metadata + 'METADATA': 'Metadata-Version: 2.1\nName: gnomon-smoke-extra\nVersion: 0.1.0\n',
                 metadata + 'WHEEL': 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
        files[metadata + 'RECORD'] = ''.join(name + ',,\n' for name in [*files, metadata + 'RECORD'])
        with zipfile.ZipFile(wheel, 'w') as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        run(command, 'python', '-m', 'pip', 'install', '--no-index', wheel)
        # Replace only Git transport with this local checkout. The installed
        # update implementation still exports dependencies and activates normally.
        update = """
import json,sys
from gnomon import installation
original = installation.subprocess.run
installation._resolve_ref = lambda *args: 'f' * 40
def local_install(args, **kwargs):
    if args[0] == 'bash':
        args = ['bash', sys.argv[1], '--local', *args[2:]]
    return original(args, **kwargs)
installation.subprocess.run = local_install
print(json.dumps(installation.update('fixture-next')))
"""
        result = json.loads(run(command, 'python', '-c', update, source / 'install.sh').stdout)
        assert result['changed'] is True
        second = command.resolve().parent.parent
        assert first != second
        assert run(command, 'python', '-c', 'import gnomon_smoke_extra; print(gnomon_smoke_extra.VALUE)').stdout.strip() == '42'

        live = subprocess.Popen([str(first / 'bin/python'), '-I', '-c',
            'import sys,time; assert "_gnomon_release_lease" in sys.modules; print("ready",flush=True); time.sleep(60)'],
            stdout=subprocess.PIPE, text=True, env=env)
        try:
            assert live.stdout.readline().strip() == 'ready'
            result = cli('releases', '--prune', '--keep', '0', '--apply')
            assert first.exists() and first.name not in result['removed']
            assert next(e['usage'] for e in result['releases'] if e['release_id'] == first.name) == 'in_use'
        finally:
            live.terminate()
            live.wait()
        result = cli('releases', '--prune', '--keep', '0', '--apply')
        assert result['removed'] == [first.name] and not first.exists()
        assert command.resolve() == second / 'bin/gnomon'

        # Restore failure happens after the core wheel installed, before activation.
        wheel.unlink()
        result = run(command, 'python', '-c', update, source / 'install.sh', check=False)
        assert result.returncode != 0
        assert command.resolve() == second / 'bin/gnomon'
        assert len(cli('releases')['releases']) == 1
        assert run(command, 'python', '-c', 'import gnomon_smoke_extra').returncode == 0
        print(json.dumps({'status': 'passed', 'dependency_restoration': 'passed', 'live_process_pruning': 'passed',
                          'failed_restore_keeps_active': 'passed',
                          'unchanged_update': 'passed' if build['commit'] and build['dirty'] is False else 'requires_clean_checkout'}))


if __name__ == '__main__':
    main()
