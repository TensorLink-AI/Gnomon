"""Supervise the 097 pilot with the tested non-restarting archival controller."""
import argparse
import json
from pathlib import Path
import sys

from .control_collection_096 import supervise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-directory', type=Path, required=True)
    fields = ('capsule', 'plan', 'preflight', 'task-source', 'runtime', 'previous-pilot',
              'previous-launch', 'output', 'credentials-file')
    for field in fields:
        parser.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    command = [sys.executable, '-m', 'benchmarks.ledger_optimization.launch_workflow_097']
    for field in fields:
        command += ['--'+field, str(getattr(args, field.replace('-', '_')).absolute())]
    result = supervise(command, args.launch_directory, args.output, args.credentials_file)
    print(json.dumps(result))
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
