#!/usr/bin/env python3
"""Validate actual N1 selected-expert host timing windows from a saved log.

Times are nested host wall intervals and process-global host API submission
counters. They are not CUDA kernel time, exclusive CPU time, physical SSD I/O,
or a performance/quality comparison. No native/device/model APIs are called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


PREFIX = '[GLM-SELECTED-EXPERT-TIMING]'
SCOPE = 'host_wall_and_nested_process_global_host_api_submission_not_kernel_or_cpu_exclusive'
DEFAULT_WINDOWS = ((1872, 32), (33, 128))
EXPECTED_BOUNDARIES = 42
ROUND_TOLERANCE_MS = 0.000003
_U64 = (1 << 64) - 1
_UINT = re.compile(r'0|[1-9][0-9]*')
_COUNTS = ('profile', 'rows', 'p0', 'valid', 'boundaries', 'asks', 'ready',
           'completed', 'errors', 'dsv4_available')
_TIMES = ('ready_ms', 'ids_ms', 'helper_ms', 'drain_ms', 'callback_ms', 'tail_ms', 'compute_ms')
_DS_COUNTS = tuple(f'dsv4_{part}_{name}' for part in ('total', 'helper', 'outer')
                   for name in ('calls', 'views', 'fused', 'nodes'))
_DS_TIMES = tuple(f'dsv4_{part}_submit_ms' for part in ('total', 'helper', 'outer'))
_KEYS = set(_COUNTS + _TIMES + _DS_COUNTS + _DS_TIMES + ('outcome', 'scope'))


def _uint(value, name):
    if not _UINT.fullmatch(value) or int(value) > _U64:
        raise ValueError('Invalid uint64 ' + name)
    return int(value)


def _milliseconds(value, name):
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError('Invalid milliseconds ' + name) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError('Nonfinite/negative milliseconds ' + name)
    return result


def parse_records(text):
    """Parse the exact native schema. Malformed tagged lines always fail closed."""
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith(PREFIX):
            continue
        if not line.startswith(PREFIX + ' '):
            raise ValueError('Malformed selected-expert timing at line%d' % line_number)
        fields = {}
        for token in line[len(PREFIX)+1:].split(' '):
            if token.count('=') != 1:
                raise ValueError('Malformed timing field at line%d' % line_number)
            key, value = token.split('=')
            if not key or not value or key in fields:
                raise ValueError('Empty/duplicate timing field at line%d' % line_number)
            fields[key] = value
        if set(fields) != _KEYS or fields['scope'] != SCOPE:
            raise ValueError('Missing/unknown timing fields or scope at line%d' % line_number)
        value = {key: _uint(fields[key], key) for key in _COUNTS + _DS_COUNTS}
        value.update({key: _milliseconds(fields[key], key) for key in _TIMES + _DS_TIMES})
        if value['profile'] != 1 or value['rows'] != 1:
            raise ValueError('Unsupported timing profile/width at line%d' % line_number)
        if value['valid'] not in (0, 1) or value['dsv4_available'] not in (0, 1):
            raise ValueError('Invalid timing Boolean at line%d' % line_number)
        if fields['outcome'] not in ('completed', 'failed'):
            raise ValueError('Unknown timing outcome at line%d' % line_number)
        value.update(outcome=fields['outcome'], scope=SCOPE, line=line_number)
        records.append(value)
    return records


def _validate(record):
    position = record['p0']
    if record['outcome'] != 'completed' or record['valid'] != 1:
        raise ValueError('Failed/invalid actual N1 timing at p0=%d' % position)
    if any(record[key] != EXPECTED_BOUNDARIES for key in ('boundaries', 'asks', 'ready', 'completed')) or record['errors'] != 0:
        raise ValueError('Selected boundary/callback protocol mismatch at p0=%d' % position)
    for component in ('calls', 'views', 'fused', 'nodes'):
        total, helper, outer = (record[f'dsv4_{part}_{component}'] for part in ('total', 'helper', 'outer'))
        if helper > total or outer != total-helper:
            raise ValueError('DSV4 component/subset mismatch at p0=%d' % position)
    total, helper, outer = (record[f'dsv4_{part}_submit_ms'] for part in ('total', 'helper', 'outer'))
    if helper > total + ROUND_TOLERANCE_MS or abs(outer-(total-helper)) > ROUND_TOLERANCE_MS:
        raise ValueError('DSV4 submission time decomposition mismatch at p0=%d' % position)
    if record['dsv4_available'] == 0 and any(record[key] != 0 for key in _DS_COUNTS + _DS_TIMES):
        raise ValueError('Unavailable DSV4 metrics must all be zero at p0=%d' % position)
    callback_parts = record['ids_ms'] + record['helper_ms'] + record['drain_ms']
    compute_parts = record['ready_ms'] + record['callback_ms'] + record['tail_ms']
    if record['callback_ms'] + ROUND_TOLERANCE_MS < callback_parts:
        raise ValueError('Callback nested intervals exceed callback wall at p0=%d' % position)
    if record['compute_ms'] + ROUND_TOLERANCE_MS < compute_parts:
        raise ValueError('Compute nested intervals exceed compute wall at p0=%d' % position)


def _sum_time(values):
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('Window timing sum overflow') from exc
    if not math.isfinite(result):
        raise ValueError('Window timing sum nonfinite')
    return result


def _sum_count(values):
    result = sum(values)
    if result > _U64:
        raise ValueError('Window counter sum exceeds uint64')
    return result


def _ratio(numerator, denominator):
    if not denominator:
        return None
    result = numerator/denominator
    if not math.isfinite(result):
        raise ValueError('Nonfinite window timing ratio')
    return result


def _summary(records):
    totals = {key: _sum_time(record[key] for record in records) for key in _TIMES + _DS_TIMES}
    totals.update({key: _sum_count(record[key] for record in records)
                   for key in ('boundaries', 'asks', 'ready', 'completed', 'errors') + _DS_COUNTS})
    available = sum(record['dsv4_available'] for record in records)
    observed = sum(bool(record['dsv4_available'] and record['dsv4_total_calls'] > 0
        and record['dsv4_outer_calls'] > 0 and record['dsv4_total_submit_ms'] > 0
        and record['dsv4_outer_submit_ms'] > 0) for record in records)
    facade_observed = observed == len(records)
    return dict(totals=totals, callback_host_wall_fractions=dict(
        ready_over_compute=_ratio(totals['ready_ms'], totals['compute_ms']),
        callback_over_compute=_ratio(totals['callback_ms'], totals['compute_ms']),
        tail_over_compute=_ratio(totals['tail_ms'], totals['compute_ms']),
        ids_over_callback=_ratio(totals['ids_ms'], totals['callback_ms']),
        helper_over_callback=_ratio(totals['helper_ms'], totals['callback_ms']),
        drain_over_callback=_ratio(totals['drain_ms'], totals['callback_ms'])),
        dsv4_available_records=available, dsv4_observed_records=observed,
        dsv4_submission_observed_for_every_record=facade_observed,
        nested_process_global_host_submission_fractions=(dict(
            total_submit_over_compute=_ratio(totals['dsv4_total_submit_ms'], totals['compute_ms']),
            helper_submit_over_helper=_ratio(totals['dsv4_helper_submit_ms'], totals['helper_ms']),
            outer_submit_over_compute=_ratio(totals['dsv4_outer_submit_ms'], totals['compute_ms']))
            if facade_observed else None))


def qualify_records(records, windows=DEFAULT_WINDOWS):
    """Require every completed actual N1 event in each explicit position window.

    Failed/valid0 startup events outside those windows receive no credit. Tagged
    syntax was already checked globally. Overlap, duplicate positions or an
    interleaved N1 record makes window association unavailable, never a pass.
    """
    used, results = set(), []
    for start, count in windows:
        if (isinstance(start, bool) or isinstance(count, bool) or not isinstance(start, int)
            or not isinstance(count, int) or start < 0 or count <= 0 or start+count-1 > _U64):
            raise ValueError('Invalid requested N1 position window')
        if count > len(records):
            raise ValueError('Requested N1 window has more positions than captured records')
        positions = range(start, start+count)
        if used.intersection(positions):
            raise ValueError('Requested N1 windows overlap')
        used.update(positions)
        indexed = [(index, record) for index, record in enumerate(records) if record['p0'] in positions]
        if [record['p0'] for _, record in indexed] != list(positions):
            raise ValueError('Missing/duplicate/out-of-order N1 positions in window%d..%d' % (start, start+count-1))
        if any(b[0] != a[0]+1 for a, b in zip(indexed, indexed[1:])):
            raise ValueError('Other N1 timing interleaves requested position window')
        selected = [record for _, record in indexed]
        for record in selected:
            _validate(record)
        results.append(dict(first_p0=start, last_p0=start+count-1, records=count,
            first_line=selected[0]['line'], last_line=selected[-1]['line'],
            callback_timing_qualified=True, **_summary(selected)))
    if not results:
        raise ValueError('No N1 timing window requested')
    return dict(schema_version=1, passed=True, expected_boundaries=EXPECTED_BOUNDARIES,
        round_tolerance_ms=ROUND_TOLERANCE_MS, windows=results,
        scope=SCOPE, cpu_exclusive_time_qualified=False, kernel_time_qualified=False,
        physical_io_qualified=False, performance_qualified=False, numerical_quality_qualified=False,
        external_process_model_provenance_qualified=False,
        limitations=['Only actual completed N1 events at the requested positions qualify; startup records supply no credit.',
            'Log events have no request/model ID. Fresh isolated process, matching native/model hashes and completed API/canonical quality must be verified separately.',
            'Intervals and process-global facade submissions are nested; they are not additive exclusive costs or GPU kernel durations.',
            'Observed facade fractions require positive total/outer calls and submission time for every record; availability alone supplies no credit.',
            'No physical SSD traffic or speedup is inferred from this diagnostic.'])


def qualify_log(text, windows=DEFAULT_WINDOWS):
    return qualify_records(parse_records(text), windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-position', type=int, help='Custom N1 window instead of both default quick-plan windows')
    parser.add_argument('--calls', type=int, help='Number of events in custom N1 window; requires --start-position')
    args = parser.parse_args()
    if (args.start_position is None) != (args.calls is None):
        parser.error('--start-position and --calls must be supplied together')
    raw = args.log.read_bytes()
    windows = DEFAULT_WINDOWS if args.start_position is None else ((args.start_position, args.calls),)
    try:
        result = qualify_log(raw.decode('utf-8-sig', errors='strict'), windows)
    except ValueError as exc:
        parser.error(str(exc))
    result['log'] = dict(path=str(args.log.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    result['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(passed=result['passed'], windows=result['windows'], performance_qualified=False)))


if __name__ == '__main__':
    main()
