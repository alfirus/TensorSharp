#!/usr/bin/env python3
"""Observe default first-PDF setup on a fresh, model-free TensorAgent host.

Start eng/validation/TensorAgentHost with a new --root and no OCR overrides.
This observer uses its connection file without printing the authentication cookie.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    connection = json.loads(Path(args.connection).read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'os': platform.platform(), 'architecture': platform.machine(),
        'systemTools': {name: shutil.which(name) for name in ('pdftoppm', 'tesseract')},
        'inference': 'not_run', 'events': [], 'pollErrors': []}
    stop = threading.Event()
    started = time.perf_counter()

    def status():
        request = urllib.request.Request(connection['baseUrl'] + '/api/knowledge/ocr/status',
            headers={'Cookie': connection['cookie']})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def record(value):
        if not report['events'] or value['status'] != report['events'][-1]['status']:
            report['events'].append({'seconds': time.perf_counter() - started, **value})

    def poll():
        while not stop.wait(0.1):
            try:
                record(status())
            except Exception as error:
                report['pollErrors'].append(type(error).__name__ + ': ' + str(error))

    thread = None
    try:
        if platform.system() not in ('Linux', 'Windows'):
            raise ValueError('This observer requires the actual managed target OS')
        if any(report['systemTools'].values()):
            raise ValueError('System OCR tools exist; use a clean machine')
        if any(key.startswith('TS_KNOWLEDGE_') and value for key, value in os.environ.items()):
            raise ValueError('Unset explicit knowledge environment overrides')
        first = status()
        record(first)
        if not first['managed'] or first['status']['state'] != 'not_started':
            raise ValueError('Start a fresh host with default managed OCR')
        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        command = [sys.executable, str(repo / 'eng/validation/knowledge/pdf_store_probe.py'),
            '--connection', str(Path(args.connection).resolve()), '--output', str(output / 'http')]
        with (output / 'http.log').open('wb') as stream:
            result = subprocess.run(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
        stop.set()
        thread.join()
        record(status())
        states = {event['status']['state'] for event in report['events']}
        report['probeExitCode'] = result.returncode
        if result.returncode or not {'downloading', 'installing', 'ready'} <= states:
            raise ValueError('HTTP checks or actual automatic provisioning transitions failed')
        report['status'] = 'passed'
        return 0
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        return 1
    finally:
        stop.set()
        if thread:
            thread.join(timeout=11)
        report['elapsedSeconds'] = time.perf_counter() - started
        (output / 'managed-http.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Managed HTTP validation status=' + report['status'])


if __name__ == '__main__':
    raise SystemExit(main())
