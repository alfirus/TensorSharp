#!/usr/bin/env python3
"""Actual TensorAgent/standalone PDF upload, retrieval, source and reindex checks.
No inference/model load; uses the shared original fixture assertions in protocol_probe.
"""
import argparse
import pathlib
import time
from protocol_probe import Probe, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixtures', default='InferenceWeb.Tests/Fixtures/Knowledge')
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    args.base_url = None
    args.model = 'not_used_no_inference'
    args.samples = 0
    args.protocols = []
    args.modes = []
    args.skip_pdf = False
    args.require_chinese = True
    probe = Probe(args)
    probe.report['coverage'] = {'inference': 'not_run', 'agent': 'not_run',
        'pdf': 'actual HTTP upload, OCR, scoped retrieval, original download, read and reindex'}
    try:
        probe.pdf()
        assert probe.report['pdf']['status'] == 'passed', probe.report['pdf']
        cid = probe.report['pdf']['collectionId']
        records = []
        for entry in probe.report['pdf']['uploads']:
            old = entry['document']
            started = time.perf_counter()
            current = probe.js(f'/api/knowledge/collections/{cid}/documents/{old["id"]}/reindex', {})['document']
            assert current['version'] > old['version'] and current['chunkCount'] > 0, current
            old_url = f'/api/knowledge/collections/{cid}/documents/{old["id"]}/source?version={old["version"]}'
            status, _, _ = probe.request(old_url)
            assert status == 409, f'Stale version must return409, got{status}'
            new_url = f'/api/knowledge/collections/{cid}/documents/{current["id"]}/source?version={current["version"]}'
            status, raw, _ = probe.request(new_url)
            assert status == 200 and digest(raw) == probe.originals[(cid, old['id'], old['version'])]
            records.append({'file': old['fileName'], 'oldVersion': old['version'], 'newVersion': current['version'],
                'seconds': time.perf_counter() - started, 'staleVersionStatus': 409, 'originalUnchanged': True})
        probe.report['reindex'] = records
        other = probe.collection('PDF scope exclusion')
        for scope in ([], [other]):
            hits = probe.js('/api/knowledge/search', {'query': 'CEDAR-4821', 'collectionIds': scope})['results']
            assert not hits, 'Evidence escaped collection scope'
        probe.report['passed'] = True
        return 0
    except Exception as error:
        probe.report.update(passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for cid in reversed(probe.collections):
            try:
                probe.js('/api/knowledge/collections/' + cid, method='DELETE')
            except Exception as error:
                probe.report.setdefault('cleanupErrors', []).append(str(error))
        probe.save()
        print(f'PDF HTTP checks passed={probe.report["passed"]}; report={probe.out / "report.json"}')


if __name__ == '__main__':
    raise SystemExit(main())
