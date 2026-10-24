#!/usr/bin/env python3
"""Replay one measured pooled-expert cold reset against preserved warm slots.

Routes belong to the FOLLOWING native forward timing record, not the preceding
record. Only a contiguous completed ordinary N1 window is supported. Metadata
comes from an existing header-only replay receipt; this tool never opens weights.
Results are logical GPU-miss payload, not physical I/O, time or numerical quality.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


_U64 = (1 << 64) - 1
_UINT = re.compile(r'0|[1-9][0-9]*')
_FORWARD = '[GLM-MTP-FORWARD-TIMING]'
_ROUTE = '[HOSTMOE-ROUTE]'
_CREATE = '[HOSTMOE-ROUTE-CREATE]'
_FILE = '[HOSTMOE-CACHE-FILE]'
_CACHE_TIME = '[HOSTMOE-CACHE-TIMING]'
_MUTATIONS = ('[HOSTMOE-CACHE-GROW]', '[HOSTMOE-CACHE-SHRINK]', '[HOSTMOE-REBALANCE]')
_FORWARD_KEYS = {'kind', 'rows', 'p0', 'nout', 'wanth', 'logits', 'reused', 'outcome',
                 'acquire_ms', 'input_ms', 'compute_ms', 'output_ms', 'total_ms', 'scope'}
_FORWARD_SCOPE = 'host_wall_including_existing_sync_and_expert_callbacks'
_CACHE_SCOPE = 'nested_host_wall_not_cuda_kernel_or_physical_io'


@lru_cache(maxsize=2)
def _module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'),
        Path(__file__).with_name(name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _uint(value, name):
    if not isinstance(value, str) or not _UINT.fullmatch(value) or int(value) > _U64:
        raise ValueError('Invalid uint64 ' + name)
    return int(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _U64:
        raise ValueError('Invalid integer ' + name)
    return value


def _fields(line, prefix):
    if not line.startswith(prefix + ' '):
        raise ValueError('Malformed ' + prefix)
    result = {}
    for token in line[len(prefix) + 1:].split(' '):
        if token.count('=') != 1:
            raise ValueError('Malformed ' + prefix + ' field')
        key, value = token.split('=')
        if not key or not value or key in result:
            raise ValueError('Missing/duplicate ' + prefix + ' field')
        result[key] = value
    return result


def _forward(line):
    result = _fields(line, _FORWARD)
    if set(result) != _FORWARD_KEYS or result['scope'] != _FORWARD_SCOPE:
        raise ValueError('Unsupported native forward timing schema')
    for key in ('kind', 'rows', 'p0', 'nout', 'wanth', 'logits'):
        result[key] = _uint(result[key], key)
    if result['reused'] not in ('-1', '0', '1') or result['outcome'] not in ('completed', 'failed'):
        raise ValueError('Invalid native forward outcome/reuse')
    for key in ('acquire_ms', 'input_ms', 'compute_ms', 'output_ms', 'total_ms'):
        try:
            value = float(result[key])
        except ValueError as exc:
            raise ValueError('Invalid native timing') from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError('Nonfinite/negative native timing')
    return result


def associate_forward_calls(text, line_offset=0):
    """Keep chronological records until their POSTCALL timer, including token0.

    Native logging has no request ID here. Exact position/layer validation in
    select_window is mandatory, and single-model/nonconcurrent capture is an
    explicit input contract. Unfinished trailing records are never a passed call.
    Wide route IDs stay as strings until the selected small N1 window is known.
    """
    calls, pending = [], []
    for number, line in enumerate(text.splitlines(), line_offset + 1):
        if line.startswith(_FORWARD):
            calls.append(dict(ordinal=len(calls), line=number,
                              timing=_forward(line), events=pending))
            pending = []
        elif line.startswith(_ROUTE):
            value = _fields(line, _ROUTE)
            if set(value) != {'layer', 'ids'} or not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', value['ids']):
                raise ValueError('Malformed route line')
            pending.append(dict(kind='route', line=number,
                                layer=_uint(value['layer'], 'layer'), ids=value['ids']))
        elif line.startswith(_CREATE) or line.startswith(_FILE):
            kind, prefix = ('create', _CREATE) if line.startswith(_CREATE) else ('retire', _FILE)
            value = _fields(line, prefix)
            expected = {'layer', 'slots', 'experts'} if kind == 'create' else {'layer', 'calls', 'file_bytes'}
            if set(value) != expected:
                raise ValueError('Unsupported ' + prefix + ' schema')
            pending.append(dict(kind=kind, line=number,
                **{key: _uint(item, key) for key, item in value.items()}))
        elif line.startswith(_CACHE_TIME):
            value = _fields(line, _CACHE_TIME)
            required = {'layer', 'rows', 'joint', 'bulk', 'outcome', 'file_bytes', 'scope'}
            if not required <= set(value) or value['scope'] != _CACHE_SCOPE:
                raise ValueError('Unsupported cache timing schema')
            pending.append(dict(kind='cache_timing', line=number, outcome=value['outcome'],
                **{key: _uint(value[key], key) for key in ('layer', 'rows', 'joint', 'bulk', 'file_bytes')}))
        elif any(line.startswith(prefix) for prefix in _MUTATIONS):
            prefix = next(prefix for prefix in _MUTATIONS if line.startswith(prefix))
            value = _fields(line, prefix)
            pending.append(dict(kind='mutation', mutation=prefix, line=number,
                                layer=_uint(value.get('layer'), 'layer')))
    return calls, pending


def validate_metadata(metadata):
    """Recheck exact quantized extents from saved descriptors, without model I/O."""
    groups = metadata.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('Saved metadata has no groups')
    blocks = _module('local-model-preflight').BLOCKS
    lookup, seen_groups, layouts = {}, set(), set()
    for group in groups:
        group_id = _integer(group.get('id'), 'group')
        layers = group.get('layers')
        parts = group.get('projections')
        if group_id in seen_groups or not isinstance(layers, list) or not layers or not isinstance(parts, list) or len(parts) != 3:
            raise ValueError('Invalid/duplicate group metadata')
        seen_groups.add(group_id)
        total, expert_count, layout = 0, None, []
        for part in parts:
            shape, kind = part.get('shape'), part.get('type')
            if not isinstance(shape, list) or len(shape) != 3 or isinstance(kind, bool) or not isinstance(kind, int) or kind not in blocks:
                raise ValueError('Unsupported projection metadata')
            for extent in shape:
                _integer(extent, 'dimension', 1)
            block, width = blocks[kind]
            expected = shape[0] // block * shape[1] * width
            if shape[0] % block or expected * shape[2] > _U64 or part.get('expert_bytes') != expected:
                raise ValueError('Saved quantized extent differs from dtype/shape')
            if expert_count is not None and shape[2] != expert_count:
                raise ValueError('Projection source expert counts differ')
            expert_count = shape[2]
            total += expected
            layout.append((kind, tuple(shape)))
        if parts[0]['shape'] != parts[1]['shape'] or parts[2]['shape'][:2] != list(reversed(parts[0]['shape'][:2])):
            raise ValueError('Unsupported Gate/Up/Down source geometry')
        if total > _U64 or group.get('expert_bytes') != total:
            raise ValueError('Saved group extent differs from projections')
        if tuple(layout) in layouts:
            raise ValueError('Identical quantized layouts must form one pooled group')
        layouts.add(tuple(layout))
        for layer in layers:
            _integer(layer, 'layer')
            if layer in lookup:
                raise ValueError('Duplicate source layer')
            lookup[layer] = group
        if group_id != min(layers):
            raise ValueError('Group ID must be its first source layer')
    identity = hashlib.sha256(json.dumps(dict(groups=groups,
        model_files=metadata.get('model_files', [])), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return groups, lookup, identity


def select_window(calls, lookup, start_position, count, top_k):
    _integer(start_position, 'start_position')
    _integer(count, 'call count', 2)
    _integer(top_k, 'top_k', 1)
    if start_position + count - 1 > _U64:
        raise ValueError('Position window overflow')
    positions = range(start_position, start_position + count)
    selected = [call for call in calls if call['timing']['kind'] == 0
                and call['timing']['rows'] == 1 and call['timing']['p0'] in positions]
    if [call['timing']['p0'] for call in selected] != list(positions):
        raise ValueError('N1 position window is missing, duplicated or out of order')
    if any(b['ordinal'] != a['ordinal'] + 1 for a, b in zip(selected, selected[1:])):
        raise ValueError('Other native forward interleaves selected N1 window')
    layers = sorted(lookup)
    result = []
    for call in selected:
        timing = call['timing']
        if timing['outcome'] != 'completed' or timing['nout'] != 1 or timing['wanth'] != 0 or timing['logits'] != 1:
            raise ValueError('Window contains unsupported/failed native forward')
        routes = [event for event in call['events'] if event['kind'] == 'route']
        timings = [event for event in call['events'] if event['kind'] == 'cache_timing']
        if [event['layer'] for event in routes] != layers or [event['layer'] for event in timings] != layers:
            raise ValueError('Ambiguous/missing full-model route or cache timing sequence')
        for index, (route, cache) in enumerate(zip(routes, timings)):
            ids = tuple(_uint(value, 'expert') for value in route['ids'].split(','))
            experts = lookup[route['layer']]['projections'][0]['shape'][2]
            if len(ids) != top_k or len(set(ids)) != top_k or any(expert >= experts for expert in ids):
                raise ValueError('Wide/duplicate/out-of-range selected route')
            if cache['rows'] != 1 or cache['joint'] != 0 or cache['bulk'] != 0 or cache['outcome'] != 'completed':
                raise ValueError('Unsupported cache execution in N1 window')
            if not route['line'] < cache['line'] < call['line']:
                raise ValueError('Cache timing does not follow its route')
            if index + 1 < len(routes) and cache['line'] >= routes[index+1]['line']:
                raise ValueError('Next layer route precedes completion of this cache call')
            route = dict(route, ids=ids)
            result.append((call['ordinal'], route, cache))
    return selected, result


def replay_retention(text, metadata, *, group_id, start_position, count,
                     initial_capacity, cold_capacity, retained_capacity, top_k=8,
                     begin_line=1, end_line=None):
    """Strictly validate one fresh->cold chronology, then replay two online LRUs.

    Only the initial populated LOW prefix is retained. An occupied truncated tail
    refuses this narrow scenario instead of inventing a survivor/compaction oracle.
    Subsequent routes are observations; neither policy uses future routes.
    """
    _integer(top_k, 'top_k', 1)
    for name, value in (('group', group_id), ('initial capacity', initial_capacity),
                        ('cold capacity', cold_capacity), ('retained capacity', retained_capacity)):
        _integer(value, name, 1 if name != 'group' else 0)
    if not top_k <= retained_capacity <= cold_capacity < initial_capacity:
        raise ValueError('Expected positive smaller cold/preserved capacities')
    groups, lookup, identity = validate_metadata(metadata)
    group = next((group for group in groups if group['id'] == group_id), None)
    if group is None or initial_capacity > len(group['layers']) * group['projections'][0]['shape'][2]:
        raise ValueError('Unknown group or capacity beyond its source extent')
    _integer(begin_line, 'begin_line', 1)
    lines = text.splitlines()
    if end_line is None:
        end_line = len(lines)
    _integer(end_line, 'end_line', begin_line)
    if end_line > len(lines):
        raise ValueError('Capture line bounds exceed trace')
    calls, trailing = associate_forward_calls('\n'.join(lines[begin_line-1:end_line]), begin_line-1)
    selected, rows = select_window(calls, lookup, start_position, count, top_k)
    group_layers = set(group['layers'])
    for index, call in enumerate(selected):
        relevant = [event for event in call['events'] if event.get('layer') in group_layers]
        creates = [event for event in relevant if event['kind'] == 'create']
        retires = [event for event in relevant if event['kind'] == 'retire']
        if any(event['kind'] == 'mutation' for event in relevant):
            raise ValueError('Additional target-group growth/shrink/rebalance is unsupported')
        expected = initial_capacity if index == 0 else cold_capacity
        if index < 2:
            if len(creates) != 1 or creates[0]['slots'] != expected or creates[0]['experts'] != group['projections'][0]['shape'][2]:
                raise ValueError('Measured target creation differs from requested chronology')
            first_route = next(event for event in relevant if event['kind'] == 'route')
            if creates[0]['line'] >= first_route['line']:
                raise ValueError('Target table is not fresh before its first route')
        elif creates or retires:
            raise ValueError('Unexpected later target table retirement/recreation')
        if index == 1:
            if len(retires) != 1 or retires[0]['line'] >= creates[0]['line']:
                raise ValueError('Missing/ambiguous physical target retirement before cold recreation')
            expected_calls = len(group_layers)
            expected_bytes = expected_calls * top_k * group['expert_bytes']
            if retires[0]['calls'] != expected_calls or retires[0]['file_bytes'] != expected_bytes:
                raise ValueError('Retired fresh owner counters do not cover exactly the first N1')
        # Initial inherited-owner retirement is allowed only before the NEW table.
        if index == 0 and any(event['line'] >= creates[0]['line'] for event in retires):
            raise ValueError('Target owner retired after initial fresh creation')
    cache_class = _module('glm-route-cache-replay').Cache
    cold = cache_class(initial_capacity, False)
    retained = cache_class(initial_capacity, False)
    reports, first_slots = [], None
    row_index = 0
    for index, call in enumerate(selected):
        if index == 1:
            first_slots = list(cold.slots)
            if any(key is not None for key in retained.slots[retained_capacity:]):
                raise ValueError('Preserved scenario would discard occupied tail; compaction not modelled')
            cold = cache_class(cold_capacity, False)
            retained.capacity = retained_capacity
            retained.slots = retained.slots[:retained_capacity]
            retained.ages = retained.ages[:retained_capacity]
        demands = cold_misses = kept_misses = restored = regressed = measured = 0
        for layer in sorted(lookup):
            ordinal, route, cache = rows[row_index]
            row_index += 1
            if ordinal != call['ordinal'] or route['layer'] != layer:
                raise AssertionError('Internal route association drift')
            if layer not in group_layers:
                continue
            source = (identity, layer)
            actual = {key for key, _ in cold.visit_with_evictions(source, route['ids'])}
            preserved = {key for key, _ in retained.visit_with_evictions(source, route['ids'])}
            expected_bytes = len(actual) * group['expert_bytes']
            if cache['file_bytes'] != expected_bytes:
                raise ValueError('Measured file bytes disagree with cold LRU at p0=%d layer=%d: %d != %d'
                    % (call['timing']['p0'], layer, cache['file_bytes'], expected_bytes))
            demands += len(route['ids'])
            cold_misses += len(actual)
            kept_misses += len(preserved)
            restored += len(actual - preserved)
            regressed += len(preserved - actual)
            measured += cache['file_bytes']
        reports.append(dict(p0=call['timing']['p0'], forward_line=call['line'],
            first_route_line=next(event['line'] for event in call['events'] if event['kind'] == 'route'),
            demands=demands, cold_capacity=cold.capacity, retained_capacity=retained.capacity,
            cold_hits=demands-cold_misses, retained_hits=demands-kept_misses,
            cold_misses=cold_misses, retained_misses=kept_misses,
            restored_hits=restored, regressed_hits=regressed,
            cold_logical_source_bytes=measured,
            retained_logical_source_bytes=kept_misses * group['expert_bytes'],
            net_logical_source_bytes_saved=(cold_misses-kept_misses) * group['expert_bytes']))
    expected_keys = len(group_layers) * top_k
    occupied = [key for key in first_slots if key is not None]
    if len(occupied) != expected_keys or len(set(occupied)) != expected_keys:
        raise ValueError('First fresh owner did not contain the exact source-qualified union')
    totals = {key: sum(row[key] for row in reports) for key in (
        'demands', 'cold_hits', 'retained_hits', 'cold_misses', 'retained_misses',
        'restored_hits', 'regressed_hits', 'cold_logical_source_bytes',
        'retained_logical_source_bytes', 'net_logical_source_bytes_saved')}
    return dict(schema_version=1, trace_association='pending_records_to_following_native_postcall_timer',
        source_identity=identity, source_identity_scope='Saved immutable-model descriptor namespace plus layer plus expert; capture model identity must be independently matched',
        group_id=group_id, layers=sorted(group_layers), expert_bytes=group['expert_bytes'],
        positions=dict(first=start_position, last=start_position+count-1, count=count),
        capture_lines=dict(first=begin_line, last=end_line),
        initial_capacity=initial_capacity, cold_capacity=cold_capacity, retained_capacity=retained_capacity,
        preserved_initial_keys=len(occupied), dropped_initial_keys=0,
        chronology_and_measured_cold_payload_matched=True,
        candidate_physical_admission_qualified=False, physical_io_qualified=False,
        performance_qualified=False, numerical_quality_qualified=False,
        totals=totals, calls=reports, trailing_unassociated_records=len(trailing),
        scope='Offline online-LRU counterfactual from exact completed N1 routes. Only logical selected payload; no runtime candidate admission, SSD, wall-time or quality qualification. Other groups are unchanged and not replayed.',
        limitations=['Single immutable model and nonconcurrent capture are required; native postcall records have no request/source-generation ID.',
            'Only one measured fresh-first-call then cold-second-call transition is supported; other target-group mutation fails closed.',
            'Candidate keeps the complete occupied low prefix, without compaction or an oracle survivor choice.',
            'Measured cold payload must match every target-layer cache timer; RAM hits/prefetch/other unmodelled traffic cannot silently qualify.',
            'Counterfactual capacity is caller-specified; rounded VMM pages, scratch, tickets and actual ceiling must be qualified separately.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True, help='Existing saved header-only receipt containing groups/model_files')
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--start-position', type=int, required=True)
    parser.add_argument('--calls', type=int, required=True)
    parser.add_argument('--initial-capacity', type=int, required=True)
    parser.add_argument('--cold-capacity', type=int, required=True)
    parser.add_argument('--retained-capacity', type=int, required=True)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--begin-line', type=int, default=1,
        help='Inclusive original line BEFORE first N1 CREATE/route, after any undelimited layer-major PP')
    parser.add_argument('--end-line', type=int,
        help='Inclusive final original capture line; must include last requested postcall timer')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trace, metadata = args.trace.read_bytes(), args.metadata.read_bytes()
    result = replay_retention(trace.decode('utf-8-sig', errors='strict'),
        json.loads(metadata.decode('utf-8-sig', errors='strict')),
        group_id=args.group, start_position=args.start_position, count=args.calls,
        initial_capacity=args.initial_capacity, cold_capacity=args.cold_capacity,
        retained_capacity=args.retained_capacity, top_k=args.top_k,
        begin_line=args.begin_line, end_line=args.end_line)
    result['trace'] = dict(path=str(args.trace.resolve()), sha256=hashlib.sha256(trace).hexdigest())
    result['metadata'] = dict(path=str(args.metadata.resolve()), sha256=hashlib.sha256(metadata).hexdigest(),
        checkpoint_opened=False)
    result['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['cache_replay_sha256'] = hashlib.sha256(Path(__file__).with_name('glm-route-cache-replay.py').read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(dict(positions=result['positions'], totals=result['totals'],
        chronology_and_measured_cold_payload_matched=True, performance_qualified=False)))


if __name__ == '__main__':
    main()
