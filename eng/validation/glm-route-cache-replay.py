#!/usr/bin/env python3
"""Replay recorded GLM decode routes under fixed pooled LRU/LFU capacities.

This is an offline miss-payload estimate, not a timing or numerical-quality
benchmark. Only GGUF tensor headers are read. Layer+expert identities stand for
the source-qualified payload identities of one immutable model.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


def module(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def model_groups(model):
    headers = module('gguf-tensor-types')
    blocks = module('local-model-preflight').BLOCKS
    layers = {}
    files = []
    for shard in headers.shards(str(model)):
        path = Path(shard)
        files.append(dict(path=str(path.resolve()), bytes=path.stat().st_size,
                          mtime_ns=path.stat().st_mtime_ns))
        for name, shape, kind in headers.tensors(shard):
            match = re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', name)
            if match:
                if len(shape) != 3 or kind not in blocks:
                    raise ValueError('Unsupported expert geometry: ' + name)
                block, width = blocks[kind]
                if shape[0] % block:
                    raise ValueError('Unaligned quantized expert row: ' + name)
                layers.setdefault(int(match[1]), {})[match[2]] = dict(
                    shape=list(shape), type=kind,
                    expert_bytes=math.prod(shape[:2]) // block * width)
    grouped = {}
    for layer, projections in sorted(layers.items()):
        if set(projections) != {'gate', 'up', 'down'}:
            raise ValueError('Incomplete expert source triple')
        parts = [projections[name] for name in ('gate', 'up', 'down')]
        key = tuple((part['type'], tuple(part['shape'])) for part in parts)
        if key not in grouped:
            grouped[key] = dict(id=layer, layers=[], projections=parts,
                                expert_bytes=sum(part['expert_bytes'] for part in parts))
        grouped[key]['layers'].append(layer)
    if not grouped:
        raise ValueError('No routed expert tensors found')
    return list(grouped.values()), files


def parse_routes(text, top_k=8, after_marker=None):
    started = after_marker is None
    rows, skipped = [], Counter()
    for line in text.splitlines():
        if after_marker is not None and after_marker in line:
            started = True
        if not started:
            continue
        match = re.fullmatch(r'\[HOSTMOE-ROUTE\] layer=(\d+) ids=([0-9,]+)', line)
        if not match:
            continue
        ids = tuple(int(value) for value in match[2].split(','))
        if len(ids) != top_k:
            skipped[len(ids)] += 1
            continue
        if len(set(ids)) != len(ids):
            raise ValueError('Decode trace contains duplicate selected experts')
        rows.append((int(match[1]), ids))
    if not rows:
        raise ValueError('No matching decode rows')
    return rows, dict(skipped)


def capacity_epoch(capacity):
    epoch = 32
    while epoch < 1024 and epoch // 2 < capacity:
        epoch *= 2
    return epoch


class Cache:
    def __init__(self, capacity, lfu, epoch_multiplier=1):
        if capacity <= 0 or epoch_multiplier <= 0:
            raise ValueError('Positive capacity and aging multiplier required')
        self.capacity, self.lfu = capacity, lfu
        self.epoch = capacity_epoch(capacity) * epoch_multiplier
        self.slots = [None] * capacity
        self.ages = [0] * capacity
        self.frequency = Counter()
        self.rows = self.clock = 0

    def visit(self, layer, ids):
        return len(self.visit_with_evictions(layer, ids))

    def visit_with_evictions(self, layer, ids):
        """Return (missed key, evicted key or None) in selected-route order."""
        keys = [(layer, expert) for expert in ids]
        if len(set(keys)) > self.capacity:
            raise ValueError('Selected union exceeds replay capacity')
        if self.lfu:
            self.rows += 1
            if self.rows == self.epoch:
                self.frequency = Counter({key: value >> 1 for key, value in self.frequency.items()})
                self.rows = 0
            for key in keys:
                self.frequency[key] = min(65535, self.frequency[key] + 1)
        locations = {key: slot for slot, key in enumerate(self.slots) if key is not None}
        protected = {locations[key] for key in keys if key in locations}
        misses = []
        for key in keys:
            if key in locations:
                continue
            def priority(slot):
                expert = self.slots[slot]
                frequency = -1 if expert is None else self.frequency[expert]
                return (frequency if self.lfu else 0, self.ages[slot], slot)
            victim = min((slot for slot in range(self.capacity) if slot not in protected), key=priority)
            old = self.slots[victim]
            if old is not None:
                locations.pop(old)
            self.slots[victim] = key
            locations[key] = victim
            protected.add(victim)
            misses.append((key, old))
        for key in keys:
            self.clock += 1
            self.ages[locations[key]] = self.clock
        return misses


class HostLru:
    """Byte-bounded RAM LRU. The keys include the original model source."""
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError('Host capacity cannot be negative')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.evicted_bytes = 0

    def get(self, key, remove=False):
        if key not in self.entries:
            return False
        if remove:
            self.bytes -= self.entries.pop(key)
        else:
            self.entries.move_to_end(key)
        return True

    def put(self, key, size):
        if size <= 0:
            raise ValueError('Host payload must have a positive size')
        if size > self.capacity:
            return False
        if key in self.entries:
            if self.entries[key] != size:
                raise ValueError('A source payload changed size')
            self.entries.move_to_end(key)
            return True
        while self.bytes + size > self.capacity:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old
            self.evicted_bytes += old
        self.entries[key] = size
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return True


def replay_host(rows, groups, capacities, host_bytes, mode, policy='lru'):
    """Cold RAM tier behind the unchanged GPU policy; no prefill warming.

    Inclusive retains source reads in RAM without touching RAM recency on GPU
    hits. Exclusive removes promotions and demotes GPU victims to RAM, charging
    their full payload as D2H traffic. It is a different architecture, not a
    prediction for an ordinary read-through cache or a free victim-byte oracle.
    """
    if mode not in ('inclusive', 'exclusive'):
        raise ValueError('Unknown host cache mode')
    if policy not in ('lru', 'lfu'):
        raise ValueError('Unknown GPU cache policy')
    lookup = {layer: group for group in groups for layer in group['layers']}
    caches = {g['id']: Cache(capacities[g['id']], policy == 'lfu') for g in groups}
    host = HostLru(host_bytes)
    file_bytes = hit_bytes = gpu_miss_bytes = demotion_bytes = host_hits = 0
    last_half_file_bytes = last_half_gpu_miss_bytes = 0
    for index, (layer, ids) in enumerate(rows):
        group = lookup[layer]
        if any(expert < 0 or expert >= group['projections'][0]['shape'][2] for expert in ids):
            raise ValueError('Expert ID outside GGUF source')
        size = group['expert_bytes']
        for key, victim in caches[group['id']].visit_with_evictions(layer, ids):
            gpu_miss_bytes += size
            hit = host.get(key, remove=mode == 'exclusive')
            if hit:
                host_hits += 1
                hit_bytes += size
            else:
                file_bytes += size
            if index >= len(rows) // 2:
                last_half_gpu_miss_bytes += size
                last_half_file_bytes += 0 if hit else size
            if mode == 'inclusive':
                host.put(key, size)
            elif victim is not None and host.put(victim, lookup[victim[0]]['expert_bytes']):
                demotion_bytes += lookup[victim[0]]['expert_bytes']
    return dict(mode=mode, gpu_policy=policy, host_capacity_bytes=host_bytes,
                host_retained_bytes=host.bytes, host_peak_bytes=host.peak_bytes,
                host_evicted_bytes=host.evicted_bytes, host_hits=host_hits,
                host_hit_bytes=hit_bytes, gpu_miss_bytes=gpu_miss_bytes,
                logical_file_bytes=file_bytes, h2d_payload_bytes=gpu_miss_bytes,
                d2h_victim_payload_bytes=demotion_bytes,
                last_half_logical_file_bytes=last_half_file_bytes,
                last_half_gpu_miss_bytes=last_half_gpu_miss_bytes,
                logical_file_reduction_percent=100 * hit_bytes / gpu_miss_bytes if gpu_miss_bytes else 0)


class VictimAdmission:
    """Past-only demand history; no future route or host-hit feedback.

    residence-hits stores one saturating counter per live GPU key. The rolling
    policies retain exactly the last window demand accesses (not tokens), with
    None placeholders aging misses out of recent-hits at the same rate. Their
    counters contain only keys with a live event in that bounded window.
    """
    def __init__(self, policy, threshold, window=0):
        if policy not in ('residence-hits', 'recent-demands', 'recent-hits'):
            raise ValueError('Unknown victim admission policy')
        if threshold <= 0 or (policy != 'residence-hits' and window <= 0):
            raise ValueError('Positive threshold and rolling window required')
        if policy == 'residence-hits' and window != 0:
            raise ValueError('GPU residence counters do not use a history window')
        self.policy, self.threshold, self.window = policy, threshold, window
        self.resident = {}
        self.history, self.counts = deque(), Counter()
        self.peak_resident = self.peak_history = self.peak_keys = 0

    def observe(self, keys, missed):
        # Every selected route in this row is already known when slots are
        # chosen. Hits are protected before demotions, as in the native path.
        for key in keys:
            hit = key not in missed
            if self.policy == 'residence-hits':
                if hit:
                    if key not in self.resident:
                        raise ValueError('GPU hit has no resident admission state')
                    self.resident[key] = min(self.threshold, self.resident[key] + 1)
            else:
                if len(self.history) == self.window:
                    old = self.history.popleft()
                    if old is not None:
                        self.counts[old] -= 1
                        if not self.counts[old]:
                            del self.counts[old]
                event = key if self.policy == 'recent-demands' or hit else None
                self.history.append(event)
                if event is not None:
                    self.counts[event] += 1
        self.peak_history = max(self.peak_history, len(self.history))
        self.peak_keys = max(self.peak_keys, len(self.counts))

    def replace(self, key, victim):
        if self.policy == 'residence-hits':
            if victim is not None and victim not in self.resident:
                raise ValueError('GPU victim has no resident admission state')
            score = self.resident.pop(victim, 0)
            self.resident[key] = 0
            self.peak_resident = max(self.peak_resident, len(self.resident))
        else:
            score = self.counts.get(victim, 0)
        return victim is not None and score >= self.threshold

    def stats(self):
        return dict(policy=self.policy, threshold=self.threshold,
                    history_event_limit=self.window,
                    peak_history_events=self.peak_history,
                    peak_history_keys=self.peak_keys,
                    peak_resident_keys=self.peak_resident)


def gpu_route_events(rows, groups, capacities):
    """Host policy cannot change GPU LRU; reuse its transitions across sweeps.

    Materializing the trace is an offline optimization only. Admission receives
    one current row at a time; it never receives this list or its suffix.
    """
    lookup = {layer: group for group in groups for layer in group['layers']}
    caches = {g['id']: Cache(capacities[g['id']], False) for g in groups}
    events = []
    for layer, ids in rows:
        group = lookup[layer]
        if len(set(ids)) != len(ids):
            raise ValueError('Selective replay requires distinct selected experts')
        if any(expert < 0 or expert >= group['projections'][0]['shape'][2] for expert in ids):
            raise ValueError('Expert ID outside GGUF source')
        keys = tuple((layer, expert) for expert in ids)
        misses = tuple(caches[group['id']].visit_with_evictions(layer, ids))
        events.append((keys, misses))
    return events


def replay_selective_host(events, groups, host_bytes, policy, threshold, window=0):
    lookup = {layer: group for group in groups for layer in group['layers']}
    admission = VictimAdmission(policy, threshold, window)
    host = HostLru(host_bytes)
    gpu_bytes = hit_bytes = d2h_bytes = host_hits = 0
    considered = admitted = rejected = capacity_bypasses = 0
    last_gpu = last_hits = last_d2h = 0
    for index, (keys, misses) in enumerate(events):
        admission.observe(keys, {key for key, _ in misses})
        for key, victim in misses:
            size = lookup[key[0]]['expert_bytes']
            gpu_bytes += size
            hit = host.get(key, remove=True)
            if hit:
                hit_bytes += size
                host_hits += 1
            demoted = 0
            selected = admission.replace(key, victim)
            if victim is not None:
                considered += 1
                if not selected:
                    rejected += 1
                elif host.put(victim, lookup[victim[0]]['expert_bytes']):
                    admitted += 1
                    demoted = lookup[victim[0]]['expert_bytes']
                    d2h_bytes += demoted
                else:
                    capacity_bypasses += 1
            if index >= len(events) // 2:
                last_gpu += size
                last_hits += size if hit else 0
                last_d2h += demoted
    return dict(mode='selective-exclusive', gpu_policy='lru', admission=admission.stats(),
                host_capacity_bytes=host_bytes, host_retained_bytes=host.bytes,
                host_peak_bytes=host.peak_bytes, host_evicted_bytes=host.evicted_bytes,
                host_hits=host_hits, host_hit_bytes=hit_bytes, gpu_miss_bytes=gpu_bytes,
                logical_file_bytes=gpu_bytes - hit_bytes, h2d_payload_bytes=gpu_bytes,
                d2h_victim_payload_bytes=d2h_bytes,
                d2h_per_avoided_file_byte=d2h_bytes / hit_bytes if hit_bytes else None,
                victims_considered=considered, victims_admitted=admitted,
                victims_rejected=rejected, victims_capacity_bypassed=capacity_bypasses,
                last_half_logical_file_bytes=last_gpu - last_hits,
                last_half_gpu_miss_bytes=last_gpu, last_half_host_hit_bytes=last_hits,
                last_half_d2h_victim_payload_bytes=last_d2h,
                logical_file_reduction_percent=100 * hit_bytes / gpu_bytes if gpu_bytes else 0)


def host_pareto_frontier(runs):
    """Keep nondominated read-saving/D2H points within each RAM ceiling.

    This is a descriptive frontier on the supplied trace, not a deployable
    future-aware chooser. Equal outcomes retain their distinct policy labels.
    """
    return [candidate for candidate in runs if not any(
        other['host_capacity_bytes'] == candidate['host_capacity_bytes']
        and other['host_hit_bytes'] >= candidate['host_hit_bytes']
        and other['d2h_victim_payload_bytes'] <= candidate['d2h_victim_payload_bytes']
        and (other['host_hit_bytes'] > candidate['host_hit_bytes']
             or other['d2h_victim_payload_bytes'] < candidate['d2h_victim_payload_bytes'])
        for other in runs)]


def replay(rows, groups, capacities, policy, epoch_mode='capacity'):
    lookup = {layer: group for group in groups for layer in group['layers']}
    caches = {group['id']: Cache(capacities[group['id']], policy == 'lfu',
                len(group['layers']) if epoch_mode == 'per-source' else 1) for group in groups}
    totals = {group['id']: dict(rows=0, accesses=0, misses=0, missed_bytes=0) for group in groups}
    chunks = []
    for layer, ids in rows:
        group = lookup[layer]
        if any(expert < 0 or expert >= group['projections'][0]['shape'][2] for expert in ids):
            raise ValueError('Expert ID outside GGUF source')
        misses = caches[group['id']].visit(layer, ids)
        total = totals[group['id']]
        total['rows'] += 1
        total['accesses'] += len(ids)
        total['misses'] += misses
        total['missed_bytes'] += misses * group['expert_bytes']
        chunks.append(misses * group['expert_bytes'])
    return dict(policy=policy, epoch_mode=epoch_mode,
                groups={key: dict(value, capacity=capacities[key], epoch=caches[key].epoch)
                        for key, value in totals.items()},
                accesses=sum(value['accesses'] for value in totals.values()),
                misses=sum(value['misses'] for value in totals.values()),
                missed_bytes=sum(chunks),
                last_half_missed_bytes=sum(chunks[len(chunks)//2:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capacities', required=True, help='First layer of each group:slots, e.g. 3:225,4:100,11:312')
    parser.add_argument('--after-marker', help='Only read routes after this literal log marker')
    parser.add_argument('--host-gib', type=int, nargs='+', default=[],
                        help='Also replay cold inclusive/exclusive host LRU tiers at these GiB capacities')
    parser.add_argument('--selective-host-gib', type=int, nargs='+', default=[],
                        help='Sweep past-only selective victim policies at these RAM GiB ceilings')
    parser.add_argument('--residence-hit-thresholds', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--history-events', type=int, nargs='+', default=[4096, 16384, 65536],
                        help='Rolling demand-access windows; each scalar top-8 layer row adds 8 events')
    parser.add_argument('--recent-demand-thresholds', type=int, nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--recent-hit-thresholds', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups, files = model_groups(args.model)
    capacities = {int(key): int(value) for key, value in (item.split(':') for item in args.capacities.split(','))}
    trace = args.trace.read_bytes()
    rows, skipped = parse_routes(trace.decode('utf-8-sig', errors='strict'), after_marker=args.after_marker)
    active_layers = {layer for layer, _ in rows}
    groups = [dict(group, layers=[layer for layer in group['layers'] if layer in active_layers]) for group in groups]
    groups = [group for group in groups if group['layers']]
    if set(capacities) != {group['id'] for group in groups}:
        raise ValueError('Capacities must specify each routed group exactly')
    runs = [replay(rows, groups, capacities, 'lru'), replay(rows, groups, capacities, 'lfu'),
            replay(rows, groups, capacities, 'lfu', 'per-source')]
    baseline = runs[0]['missed_bytes']
    for run in runs:
        run['missed_bytes_reduction_percent'] = 100 * (1 - run['missed_bytes'] / baseline)
    report = dict(scope='Offline cold-start fixed-capacity single-model decode route replay. No GPU, timings, prefetch, capacity changes, prefill warming, numerical quality or physical I/O claim.',
                  trace=dict(path=str(args.trace.resolve()), sha256=hashlib.sha256(trace).hexdigest(),
                             after_marker=args.after_marker, rows=len(rows),
                             rows_per_layer=dict(Counter(layer for layer, _ in rows)), skipped_widths=skipped),
                  model_files=files, model_identity_scope='Header geometry and size/mtime only; no full weight hash.',
                  groups=groups, runs=runs,
                  aging='capacity matches ExpertEvictionPolicy global group-row epoch; per-source multiplies that epoch by source count as a separate experimental longer-history policy. Both saturate uint16 and halve periodically; LRU breaks frequency ties.')
    if args.host_gib:
        report['host_runs'] = [replay_host(rows, groups, capacities, size * 2**30, mode)
                              for size in args.host_gib for mode in ('inclusive', 'exclusive')]
        report['host_scope'] = ('Cold byte-bounded host LRU behind unchanged pooled GPU LRU. '
            'Inclusive stores each file-sourced GPU miss and touches RAM only on GPU misses. '
            'Exclusive removes RAM promotions and demotes every admitted GPU victim through D2H. '
            'Reports logical source and transfer payload only; D2H, H2D, copies, memory pressure, '
            'page-cache interference and host-cache lookup costs are not timings. '
            'No scan resistance, prefill warming, physical disk traffic or performance claim.')
    if args.selective_host_gib:
        events = gpu_route_events(rows, groups, capacities)
        configurations = [('residence-hits', threshold, 0) for threshold in args.residence_hit_thresholds]
        configurations += [(policy, threshold, window)
                           for window in args.history_events
                           for policy, thresholds in [('recent-demands', args.recent_demand_thresholds),
                                                      ('recent-hits', args.recent_hit_thresholds)]
                           for threshold in thresholds]
        report['selective_host_runs'] = [
            replay_selective_host(events, groups, size * 2**30, policy, threshold, window)
            for size in args.selective_host_gib for policy, threshold, window in configurations]
        report['selective_host_frontier'] = host_pareto_frontier(
            report['selective_host_runs'] + report.get('host_runs', []))
        report['selective_host_scope'] = (
            'Past-only GPU-residence hit counters or bounded rolling demand/GPU-hit counters. '
            'The history window counts selected expert accesses across all layers, not tokens. '
            'RAM promotions remove entries; only selected GPU victims pay full D2H. '
            'GPU LRU decisions remain unchanged. History counters contain no future routes or '
            'host-hit feedback. Peak key/event counts bound metadata cardinality, not Python or '
            'native allocation bytes. The descriptive Pareto frontier is evaluated on this same '
            'trace; it is not an oracle admission policy, held-out validation, or a speedup claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('scope', 'trace', 'runs', 'host_runs', 'host_scope',
                                                  'selective_host_frontier', 'selective_host_scope') if key in report}))


if __name__ == '__main__':
    main()
