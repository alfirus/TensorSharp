#!/usr/bin/env python3
"""Run actual local Poppler/Tesseract tests and OCR benchmark; missing tools fail.

No chat/embedding models, package installs, downloads, remote hosts, or native ggml execution.
Provision tools and language data explicitly before calling this script.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET


def executable(value):
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError('Required executable is unavailable: ' + value)
    return str(Path(resolved).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixtures', default='InferenceWeb.Tests/Fixtures/Knowledge')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--dotnet', default='dotnet')
    parser.add_argument('--pdftoppm', default='pdftoppm')
    parser.add_argument('--tesseract', default='tesseract')
    parser.add_argument('--tessdata', required=True)
    parser.add_argument('--data-profile', choices=('fast', 'best'), default='best')
    parser.add_argument('--languages', default='eng+chi_sim')
    args = parser.parse_args()
    if not 1 <= args.samples <= 20:
        parser.error('--samples must be 1..20')
    repo = Path(__file__).resolve().parents[3]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'startedUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'os': platform.platform(), 'architecture': platform.machine(), 'python': platform.python_version(),
        'status': 'failed', 'steps': [], 'nativeGgml': 'not_run',
        'models': 'ocr_traineddata_only', 'chatAndEmbeddingModels': 'not_used',
        'limitations': ['OCR and store correctness on original synthetic fixtures only.',
            'One local host with sequential requests; no cross-platform performance equivalence claim.',
            'Windows CI definition is not Windows execution evidence until run on Windows.']}
    environment = dict(os.environ, TENSORSHARP_GGML_NATIVE_SKIP='true',
        TENSORSHARP_GGML_NATIVE_ENABLE_CUDA='OFF', TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN='OFF',
        TENSORSHARP_CUDNN='OFF', DOTNET_NOLOGO='true')

    def save():
        (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(label, command):
        started = time.perf_counter()
        log = output / (label + '.log')
        with log.open('wb') as stream:
            result = subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        report['steps'].append({'name': label, 'command': command, 'exitCode': result.returncode,
            'seconds': time.perf_counter() - started, 'log': log.name})
        save()
        if result.returncode:
            raise RuntimeError(label + ' failed; inspect ' + str(log))
        return log.read_text(errors='replace')

    try:
        dotnet, renderer, ocr = map(executable, (args.dotnet, args.pdftoppm, args.tesseract))
        environment.update(TS_KNOWLEDGE_PDFTOPPM_PATH=renderer, TS_KNOWLEDGE_TESSERACT_PATH=ocr)
        tessdata = Path(args.tessdata).resolve()
        fixtures = Path(args.fixtures).resolve()
        expected = json.loads(Path(__file__).with_name('tessdata-' + args.data_profile + '.json').read_text())
        hashes = {entry['language']: entry['sha256'] for entry in expected['files']}
        report['dataProfile'] = args.data_profile
        report['dataRepository'] = expected['repository']
        report['dataRevision'] = expected['revision']
        if any(language not in hashes for language in args.languages.split('+')):
            raise ValueError('Selected language is not in the pinned data profile')
        report['languageData'] = []
        # Verify companion models too: Chinese models can load *_vert data
        # implicitly even when only the base language is selected on the CLI.
        for entry in expected['files']:
            language = entry['language']
            path = tessdata / (language + '.traineddata')
            data = path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if len(data) != entry['bytes'] or actual != hashes[language]:
                raise ValueError('Language data differs from the pinned validation manifest: ' + language)
            report['languageData'].append({'language': language, 'sha256': actual})
        report['fixtureHashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(fixtures.glob('*.pdf'))}
        if not report['fixtureHashes']:
            raise FileNotFoundError('No PDF fixtures at ' + str(fixtures))
        run('dotnet-info', [dotnet, '--info'])
        run('poppler-version', [renderer, '-v'])
        version = run('tesseract-version', [ocr, '--version'])
        if not re.match(r'^tesseract v?5\.', version.lstrip()):
            raise ValueError('This validation requires Tesseract 5.x')
        run('tesseract-languages', [ocr, '--tessdata-dir', str(tessdata), '--list-langs'])
        run('tests', [dotnet, 'test', 'InferenceWeb.Tests/InferenceWeb.Tests.csproj',
            '--filter', 'FullyQualifiedName~PortablePdfProcessRunnerTests|FullyQualifiedName~PortablePdfPageExtractorTests|FullyQualifiedName~KnowledgePdfFixtureTests|FullyQualifiedName~KnowledgePdfBridgeTests',
            '--logger', 'trx;LogFileName=tests.trx', '--results-directory', str(output)])
        test_results = ET.parse(output / 'tests.trx')
        counters = test_results.find('.//{*}Counters')
        if counters is None or int(counters.get('executed', '0')) == 0:
            raise ValueError('No tests executed; filter or test setup is invalid')
        report['testCounts'] = dict(counters.attrib)
        # Some test adapters leave Counters.notExecuted at zero for skipped
        # tests. Preserve each result's outcome instead of inferring from it.
        report['skippedTests'] = [result.get('testName') for result in
            test_results.findall('.//{*}UnitTestResult')
            if result.get('outcome') == 'NotExecuted']
        report['testSkipsArePasses'] = False
        run('benchmark', [dotnet, 'run', '--project', 'eng/validation/PortablePdfBench/PortablePdfBench.csproj', '--',
            '--output', str(output / 'benchmark'), '--fixtures', str(fixtures), '--provider', 'tesseract',
            '--languages', args.languages, '--samples', str(args.samples), '--pdftoppm', renderer,
            '--tesseract', ocr, '--tessdata', str(tessdata)])
        report['status'] = 'passed'
        return 0
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        print(report['error'], flush=True)
        return 1
    finally:
        report['completedUtc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print('Validation status=' + report['status'] + '; report=' + str(output / 'validation.json'), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
