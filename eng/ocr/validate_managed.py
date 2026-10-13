#!/usr/bin/env python3
"""Validate actual automatic OCR download, private install, and offline reuse.

Run on the OS being claimed. No system package installation or model inference.
The first cache must be empty. Network-denied checks use a refused local proxy.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dotnet', default='dotnet')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--configuration', choices=('Release', 'Debug'), default='Release')
    parser.add_argument('--require-no-system-tools', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'startedUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'failed',
        'os': platform.platform(), 'architecture': platform.machine(), 'configuration': args.configuration, 'steps': [],
        'nativeGgml': 'not_run', 'inference': 'not_run', 'networkDenyMethod': 'HTTP(S)/ALL_PROXY=http://127.0.0.1:9; NO_PROXY empty',
        'limitations': ['Synthetic fixtures and one real host only.', 'Other OS runs are not implied by this result.']}
    environment = dict(os.environ, TENSORSHARP_GGML_NATIVE_SKIP='true', DOTNET_NOLOGO='true')

    def save():
        (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command, env, expected_success=True):
        started = time.perf_counter()
        with (output / (label + '.log')).open('wb') as stream:
            result = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report['steps'].append({'name': label, 'command': command, 'exitCode': result.returncode,
            'seconds': time.perf_counter() - started, 'expectedSuccess': expected_success})
        save()
        if (result.returncode == 0) != expected_success:
            raise RuntimeError(label + ' returned unexpected status ' + str(result.returncode))

    try:
        if platform.system() not in ('Linux', 'Windows'):
            raise ValueError('Managed OCR validation requires actual Linux or Windows')
        if not 1 <= args.samples <= 20:
            raise ValueError('--samples must be 1..20')
        if cache.exists() and any(cache.iterdir()):
            raise ValueError('First-run cache must be empty; choose a new directory')
        report['cacheInitiallyEmpty'] = True
        report['systemTools'] = {name: shutil.which(name) for name in ('pdftoppm', 'tesseract')}
        if args.require_no_system_tools and any(report['systemTools'].values()):
            raise ValueError('A system OCR executable is present; this is not a clean-machine run')
        overrides = {key: value for key, value in environment.items() if key in
            ('TS_KNOWLEDGE_PDFTOPPM_PATH', 'TS_KNOWLEDGE_TESSERACT_PATH', 'TS_KNOWLEDGE_TESSDATA_DIRECTORY', 'TS_KNOWLEDGE_OCR_LANGUAGES') and value}
        if overrides:
            raise ValueError('Unset explicit OCR dependency overrides before managed validation')
        dotnet = shutil.which(args.dotnet)
        if not dotnet:
            raise FileNotFoundError('dotnet executable missing')
        run('build', [dotnet, 'build', 'eng/validation/PortablePdfBench/PortablePdfBench.csproj', '-c', args.configuration, '--nologo'], environment)
        dll = repo / 'eng/validation/PortablePdfBench/bin' / args.configuration / 'net10.0/PortablePdfBench.dll'

        def command(folder, directory, samples):
            return [dotnet, str(dll), '--managed', 'true', '--cache', str(directory), '--output', str(folder),
                '--fixtures', str(repo / 'InferenceWeb.Tests/Fixtures/Knowledge'), '--samples', str(samples)]

        run('first-run', command(output / 'first-run', cache, args.samples), environment)
        first = json.loads((output / 'first-run/report.json').read_text())
        if first['status'] != 'passed' or not any(event['State'] == 'downloading' for event in first['setupEvents']):
            raise ValueError('First run did not prove automatic setup and OCR success')
        denied = dict(environment, HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9',
            ALL_PROXY='http://127.0.0.1:9', NO_PROXY='', http_proxy='http://127.0.0.1:9',
            https_proxy='http://127.0.0.1:9', all_proxy='http://127.0.0.1:9', no_proxy='')
        run('offline-reuse', command(output / 'offline-reuse', cache, 1), denied)
        reused = json.loads((output / 'offline-reuse/report.json').read_text())
        if reused['status'] != 'passed' or any(event['State'] in ('downloading', 'installing') for event in reused['setupEvents']):
            raise ValueError('Offline reuse attempted a download or failed OCR')
        missing = output / 'offline-empty-cache'
        run('offline-empty-cache', command(output / 'offline-empty', missing, 1), denied, expected_success=False)
        negative = json.loads((output / 'offline-empty/report.json').read_text())
        if negative['status'] != 'failed' or (missing / 'ready.json').exists():
            raise ValueError('Empty offline cache incorrectly reported ready')
        report['firstSetupMilliseconds'] = first['setupMilliseconds']
        report['offlineReuseSetupMilliseconds'] = reused['setupMilliseconds']
        report['status'] = 'passed'
        return 0
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        print(report['error'], flush=True)
        return 1
    finally:
        report['completedUtc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print('Managed validation status=' + report['status'], flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
