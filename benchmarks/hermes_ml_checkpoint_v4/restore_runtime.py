"""Restore frozen wheel/Hermes runtimes into a new durable directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    assets = args.assets.resolve(); root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    setup = root/'setup'; setup.mkdir()
    hermes = setup/'hermes'; hermes.mkdir()
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    with tarfile.open(assets/'hermes-frozen.tar.gz') as archive:
        archive.extractall(hermes, filter='data')
    source = setup/'recovered-task-source'; source.mkdir()
    shutil.copyfile(assets/'host-jobs-original.json', source/'host-jobs.json')
    def run(*argv):
        subprocess.run(list(map(str, argv)), env=env, check=True)
    plain = root/'plain-venv'
    run(sys.executable, '-m', 'venv', plain)
    python = plain/'bin/python'
    run(python, '-m', 'pip', 'install', '--disable-pip-version-check',
        '-r', assets/'requirements-matched.txt')
    run(python, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps', '-e', hermes)
    gnomon = root/'gnomon-venv'
    shutil.copytree(plain, gnomon, symlinks=True)
    run(gnomon/'bin/python', '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps',
        assets/'gnomon_forecast-1.2.0-py3-none-any.whl')
    code = ('import json,importlib.metadata as m,importlib.util as u;'
            'print(json.dumps({"gnomon_available":u.find_spec("gnomon") is not None,'
            '"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))')
    evidence = {}
    for name in ('plain','gnomon'):
        expected = json.loads((assets/(name+'-runtime.json')).read_text())
        if 'pip' not in expected['packages']:
            run(root/(name+'-venv/bin/python'), '-m', 'pip', 'uninstall', '-y', 'pip')
        value = json.loads(subprocess.check_output([str(root/(name+'-venv/bin/python')), '-I', '-c', code], env=env))
        assert value['packages'] == expected['packages'], (name, 'Frozen package versions differ')
        assert value['gnomon_available'] == (name == 'gnomon')
        evidence[name] = value
    (root/'VERIFIED.json').write_text(json.dumps(evidence, indent=2)+'\n')


if __name__ == '__main__':
    main()
