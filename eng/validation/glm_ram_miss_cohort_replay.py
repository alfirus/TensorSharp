#!/usr/bin/env python3
"""Compare equal-total RAM LRU with cohorts frozen from past GPU misses.

The CLI deliberately accepts only the already-audited v27 cold D128 capture.
It never opens a checkpoint or predicts time, physical SSD traffic, guaranteed
RAM residency, or a different GPU trajectory. The callable analysis supports
small independently generated cold captures for portable CPU tests.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

import glm_route_retention_replay as retention


GIB = 1 << 30
MAX_TRACE_BYTES = 32 << 20
MAX_METADATA_BYTES = 4 << 20
MAX_CALLS = 512
MAX_LAYERS = 256
MAX_SOURCE_KEYS = 65536
MAX_WINDOW_LINES = 65536
V27_TRACE_SHA256 = '8e603d4c4718ae8367c744a5dc27d367fb170c0118ff2053b325332df7a94cfe'
V27_METADATA_SHA256 = '5cbb184adce69d042c6db180e65a1bff3a71419cdb67e5f77e06c92e38a4a8c7'
V27_WINDOW = dict(start_position=33, count=128, train_tokens=32, top_k=8,
                  begin_line=4075, end_line=15225, capacities={3: 299, 4: 750, 11: 87})
V27_BASELINE = dict(gpu_misses=20018, logical_source_bytes=139102257152)
LIMITATIONS = [
    'Logical immutable Gate/Up/Down triple payload only; no physical SSD, time, throughput or numerical-quality claim.',
    'RAM budgets count selected raw payload only. Metadata, page padding, pinning, OS cache displacement, copies, and vendor memory are excluded; no guaranteed residency or physical-admission certificate.',
    'GPU LRU, capacities and cold seed stay fixed. Inclusive static RAM can duplicate GPU-resident payload; no exclusive-capacity or D2H benefit is claimed.',
    'Selection happens after training. The static reference charges every selected triple once; equal-total comparisons use ordinary RAM-LRU during training and separately charge selected triples absent at freeze. No training work is retrospectively credited.',
    'Source generations have no explicit native IDs in this trace. The CLI pins the known single-immutable-model capture and saved descriptor hashes; keys namespace descriptor identity plus layer plus expert.',
    'Frequency ranking uses training GPU misses only, descending count then ascending layer/expert, with whole-triple greedy fitting. This is not a knapsack optimum or selection on holdout results.',
    'One diagnostic prompt suffix is held out chronologically; no cross-prompt/model generalization or production policy is qualified.',
    'Equal-total comparisons are counterfactual private whole-triple payload caches. They do not reconstruct the actual OS cache, readahead, file-page sharing, RAM availability, or physical SSD traffic.',
    'At cohort freeze, only selected triples absent from the training RAM LRU incur incremental admission reads. Noncohort entries are retained newest-first when their entire size fits the remaining payload budget; no future reuse is consulted.',
]


def _bounded(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Invalid bounded ' + name)
    return value


def reconstruct_misses(text, metadata, *, capacities, start_position, count, top_k,
                       begin_line=1, end_line=None):
    """Require every pool's fresh creation and reconcile EVERY layer timer."""
    _bounded(count, 'call count', 2, MAX_CALLS)
    _bounded(top_k, 'top_k', 1, 64)
    groups, lookup, identity = retention.validate_metadata(metadata)
    if (not 1 <= len(lookup) <= MAX_LAYERS or type(capacities) is not dict
            or set(capacities) != {group['id'] for group in groups}
            or any(type(key) is not int for key in capacities)):
        raise ValueError('Every routed pool requires one explicit cold capacity')
    key_space = 0
    for group in groups:
        experts = group['projections'][0]['shape'][2]
        key_space += len(group['layers']) * experts
        _bounded(capacities[group['id']], 'pool capacity', top_k,
                 min(8192, len(group['layers']) * experts))
    if key_space > MAX_SOURCE_KEYS:
        raise ValueError('Source-key namespace exceeds the bounded replay profile')
    lines = text.splitlines()
    if end_line is None:
        end_line = len(lines)
    _bounded(begin_line, 'begin line', 1, len(lines))
    _bounded(end_line, 'end line', begin_line, len(lines))
    if end_line - begin_line + 1 > MAX_WINDOW_LINES:
        raise ValueError('Capture window exceeds bounded line count')
    capture = '\n'.join(lines[begin_line - 1:end_line])
    # This narrow model has no policy/source/copy-side mutation not represented
    # by the original parser. The actual CLI additionally pins full provenance.
    if '[HOSTMOE-VMM-' in capture or '[HOSTMOE-PREFETCH]' in capture:
        raise ValueError('Unsupported cache policy or prefetch activity')
    calls, trailing = retention.associate_forward_calls(capture, begin_line - 1)
    selected, rows = retention.select_window(calls, lookup, start_position, count, top_k)
    if len(calls) != len(selected) or trailing:
        raise ValueError('Window must contain exactly the completed cold N1 calls')
    for index, call in enumerate(selected):
        if any(event['kind'] == 'mutation' and event.get('layer') not in lookup for event in call['events']):
            raise ValueError('Mutation of an unknown GPU source is unsupported')
        for group in groups:
            relevant = [event for event in call['events'] if event.get('layer') in group['layers']]
            creates = [event for event in relevant if event['kind'] == 'create']
            retires = [event for event in relevant if event['kind'] == 'retire']
            mutations = [event for event in relevant if event['kind'] == 'mutation']
            if index == 0:
                first_route = next(event for event in relevant if event['kind'] == 'route')
                if (len(creates) != 1 or creates[0]['layer'] != group['id']
                        or creates[0]['slots'] != capacities[group['id']]
                        or creates[0]['experts'] != group['projections'][0]['shape'][2]
                        or creates[0]['line'] >= first_route['line']
                        or any(event['line'] >= creates[0]['line'] for event in retires + mutations)):
                    raise ValueError('Missing exact fresh pool seed before its first demand')
                # Actual v27 rebalance retires inherited PP owners first. Their
                # old keys/ages are never used: every pool is recreated before
                # its first selected demand, providing the observable cold seed.
            elif creates or retires or mutations:
                raise ValueError('Later pool recreation/retirement would invalidate the fixed GPU seed')
    cache_class = retention._module('glm-route-cache-replay').Cache
    caches = {group['id']: cache_class(capacities[group['id']], False) for group in groups}
    reports = []
    for ordinal, route, cache in rows:
        group = lookup[route['layer']]
        # Namespace identity does not affect ordering or admission. Each layer
        # identifies one immutable source generation with all three projections.
        misses = caches[group['id']].visit_with_evictions((identity, route['layer']), route['ids'])
        missed_ids = [key[1] for key, _ in misses]
        predicted = len(missed_ids) * group['expert_bytes']
        if predicted != cache['file_bytes']:
            raise ValueError('GPU miss bytes disagree with layer timer at p0=%d layer=%d: %d != %d'
                % (selected[ordinal]['timing']['p0'], route['layer'], predicted, cache['file_bytes']))
        reports.append(dict(p0=selected[ordinal]['timing']['p0'], layer=route['layer'], group=group['id'],
            gpu_miss_experts=missed_ids, gpu_hits=len(route['ids']) - len(missed_ids),
            expert_bytes=group['expert_bytes'], logical_source_bytes=predicted,
            route_line=route['line'], cache_timer_line=cache['line']))
    return dict(source_identity=identity, groups=groups, rows=reports,
        positions=dict(first=start_position, last=start_position + count - 1, count=count),
        capture_lines=dict(first=begin_line, last=end_line), capacities=capacities,
        source_key_space=key_space, layer_timer_checks=len(rows),
        gpu_misses=sum(len(row['gpu_miss_experts']) for row in reports),
        logical_source_bytes=sum(row['logical_source_bytes'] for row in reports),
        gpu_hits=sum(row['gpu_hits'] for row in reports),
        capture_text_sha256=hashlib.sha256(capture.encode()).hexdigest())


def freeze_cohort(training, budget_bytes):
    """Only training rows are supplied to selection; no holdout feedback."""
    _bounded(budget_bytes, 'RAM payload budget', 1, retention._U64)
    counts, sizes, group_ids = Counter(), {}, {}
    for row in training:
        layer = row['layer']
        for expert in row['gpu_miss_experts']:
            key = (layer, expert)
            if key in sizes and sizes[key] != row['expert_bytes']:
                raise ValueError('Immutable source triple changed size')
            counts[key] += 1
            sizes[key] = row['expert_bytes']
            group_ids[key] = row['group']
    selected, used = [], 0
    for layer, expert in sorted(counts, key=lambda key: (-counts[key], key[0], key[1])):
        key = (layer, expert)
        size = sizes[key]
        if size <= budget_bytes - used:
            selected.append(dict(layer=layer, expert=expert, group=group_ids[key],
                payload_bytes=size, training_gpu_misses=counts[key]))
            used += size
    return dict(budget_bytes=budget_bytes, selected_payload_bytes=used,
        selected_triples=len(selected), selected=selected,
        selection_sha256=hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


class PayloadLru:
    """Metadata-only byte-weighted LRU; no payload allocation or OS-cache model."""
    def __init__(self, capacity):
        self.capacity = _bounded(capacity, 'RAM LRU capacity', 0, retention._U64)
        self.entries = OrderedDict()  # oldest to newest, exact source-qualified key -> bytes
        self.owned_bytes = self.peak_bytes = 0

    def demand(self, key, size):
        _bounded(size, 'immutable triple bytes', 1, retention._U64)
        if key in self.entries:
            if self.entries[key] != size:
                raise ValueError('A source-qualified RAM payload changed size')
            self.entries.move_to_end(key)
            return True
        if size > self.capacity:
            return False  # An uncacheable miss must not flush otherwise useful entries.
        while self.owned_bytes > self.capacity - size:
            _, old_size = self.entries.popitem(last=False)
            self.owned_bytes -= old_size
        self.entries[key] = size
        self.owned_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.owned_bytes)
        return False

    def clone(self):
        result = PayloadLru(self.capacity)
        result.entries = self.entries.copy()
        result.owned_bytes, result.peak_bytes = self.owned_bytes, self.peak_bytes
        return result

    def newest_fit(self, capacity, excluded):
        """Retain existing bytes only, greedily newest first, no admission I/O."""
        result = PayloadLru(capacity)
        chosen = []
        remaining = capacity
        for key, size in reversed(self.entries.items()):
            if key not in excluded and size <= remaining:
                chosen.append((key, size))
                remaining -= size
        result.entries = OrderedDict(reversed(chosen))
        result.owned_bytes = result.peak_bytes = capacity - remaining
        return result


def _serve_ram(lru, rows, source_identity, pinned=None):
    pinned = {} if pinned is None else pinned
    pinned_bytes = sum(pinned.values())
    if set(lru.entries).intersection(pinned):
        raise ValueError('Pinned and dynamic RAM payload must never duplicate a key')
    totals = dict(gpu_misses=0, gpu_miss_bytes=0, cohort_hits=0, cohort_hit_bytes=0,
                  lru_hits=0, lru_hit_bytes=0, ram_misses=0, ram_miss_logical_source_bytes=0)
    per_group = {}
    for row in rows:
        group = per_group.setdefault(row['group'], dict(gpu_miss_bytes=0,
            cohort_hit_bytes=0, lru_hit_bytes=0, ram_miss_logical_source_bytes=0))
        size = row['expert_bytes']
        for expert in row['gpu_miss_experts']:
            key = (source_identity, row['layer'], expert)
            totals['gpu_misses'] += 1
            totals['gpu_miss_bytes'] += size
            group['gpu_miss_bytes'] += size
            if key in pinned:
                if pinned[key] != size:
                    raise ValueError('A frozen source-qualified payload changed size')
                totals['cohort_hits'] += 1
                totals['cohort_hit_bytes'] += size
                group['cohort_hit_bytes'] += size
            elif lru.demand(key, size):
                totals['lru_hits'] += 1
                totals['lru_hit_bytes'] += size
                group['lru_hit_bytes'] += size
            else:
                totals['ram_misses'] += 1
                totals['ram_miss_logical_source_bytes'] += size
                group['ram_miss_logical_source_bytes'] += size
    if (totals['gpu_misses'] != totals['cohort_hits'] + totals['lru_hits'] + totals['ram_misses']
            or totals['gpu_miss_bytes'] != totals['cohort_hit_bytes'] + totals['lru_hit_bytes']
                + totals['ram_miss_logical_source_bytes']
            or set(lru.entries).intersection(pinned)
            or lru.owned_bytes != sum(lru.entries.values())
            or not 0 <= lru.owned_bytes <= lru.peak_bytes <= lru.capacity):
        raise ValueError('Counterfactual RAM conservation or deduplication failed')
    return dict(**totals, per_group=per_group, pinned_payload_bytes=pinned_bytes,
        dynamic_capacity_bytes=lru.capacity, dynamic_resident_bytes=lru.owned_bytes,
        dynamic_peak_bytes=lru.peak_bytes, owned_payload_peak_bytes=pinned_bytes + lru.peak_bytes,
        h2d_payload_bytes_unchanged=totals['gpu_miss_bytes'])


def compare_same_total_ram(training, holdout, cohorts, source_identity, total_budgets):
    """Train ordinary RAM LRU, then compare exact same-total frozen alternatives."""
    if (type(cohorts) not in (list, tuple) or not 1 <= len(cohorts) <= 16
            or not isinstance(source_identity, str) or not 1 <= len(source_identity) <= 256):
        raise ValueError('Expected bounded frozen cohorts and an immutable source namespace')
    if (type(total_budgets) not in (list, tuple) or not total_budgets or len(total_budgets) > 16
            or any(type(value) is not int for value in total_budgets)
            or len(set(total_budgets)) != len(total_budgets)):
        raise ValueError('Total RAM budgets must be a bounded distinct integer list')
    comparisons = []
    for total in total_budgets:
        _bounded(total, 'total RAM payload budget', 1, retention._U64)
        seed = PayloadLru(total)
        train = _serve_ram(seed, training, source_identity)
        seed_items = list(seed.entries.items())
        ordinary = seed.clone()
        baseline = _serve_ram(ordinary, holdout, source_identity)
        baseline_whole = train['ram_miss_logical_source_bytes'] + baseline['ram_miss_logical_source_bytes']
        alternatives = []
        for cohort in cohorts:
            ceiling = _bounded(cohort['budget_bytes'], 'cohort ceiling', 1, retention._U64)
            if ceiling > total:
                raise ValueError('A cohort ceiling cannot exceed the compared total RAM budget')
            if not isinstance(cohort['selected'], list) or len(cohort['selected']) > MAX_SOURCE_KEYS:
                raise ValueError('Frozen cohort metadata exceeds the bounded source namespace')
            _bounded(cohort['selected_payload_bytes'], 'selected payload bytes', 0, retention._U64)
            pinned = {}
            for item in cohort['selected']:
                key = (source_identity, item['layer'], item['expert'])
                if key in pinned:
                    raise ValueError('Frozen cohort contains a duplicate source-qualified triple')
                pinned[key] = _bounded(item['payload_bytes'], 'cohort triple bytes', 1, retention._U64)
            selected_bytes = sum(pinned.values())
            if selected_bytes != cohort['selected_payload_bytes'] or selected_bytes > cohort['budget_bytes']:
                raise ValueError('Frozen cohort payload charge is inconsistent')
            already_cached = 0
            for key, size in pinned.items():
                if key in seed.entries:
                    if seed.entries[key] != size:
                        raise ValueError('Cached selected triple changed size at freeze')
                    already_cached += size
            remaining = total - selected_bytes  # All unused cohort allowance remains available to LRU.
            dynamic = seed.newest_fit(remaining, pinned)
            admission = selected_bytes - already_cached
            discarded = seed.owned_bytes - already_cached - dynamic.owned_bytes
            after_freeze = selected_bytes + dynamic.owned_bytes
            if (discarded < 0 or seed.owned_bytes + admission - discarded != after_freeze
                    or after_freeze > total):
                raise ValueError('Freeze must evict discarded bytes before bounded cohort admission')
            candidate = _serve_ram(dynamic, holdout, source_identity, pinned)
            candidate_whole = (train['ram_miss_logical_source_bytes'] + admission
                + candidate['ram_miss_logical_source_bytes'])
            peak = max(train['owned_payload_peak_bytes'], after_freeze,
                       candidate['owned_payload_peak_bytes'])
            if peak > total or list(seed.entries.items()) != seed_items:
                raise ValueError('An alternative changed the shared training seed or exceeded total RAM')
            alternatives.append(dict(cohort_budget_bytes=cohort['budget_bytes'],
                selected_payload_bytes=selected_bytes, selected_triples=len(pinned),
                selection_sha256=cohort['selection_sha256'],
                freeze=dict(training_resident_bytes=seed.owned_bytes,
                    selected_already_cached_bytes=already_cached,
                    incremental_admission_logical_source_bytes=admission,
                    worst_case_full_cold_admission_logical_source_bytes=selected_bytes,
                    retained_dynamic_bytes=after_freeze - selected_bytes,
                    discarded_training_payload_bytes=discarded,
                    owned_payload_after_bytes=after_freeze,
                    ordering='plan from past state; discard nonretained bytes before admitting absent cohort triples'),
                holdout=candidate, whole_window_logical_source_bytes_including_admission=candidate_whole,
                holdout_logical_bytes_saved_before_admission=(baseline['ram_miss_logical_source_bytes']
                    - candidate['ram_miss_logical_source_bytes']),
                net_logical_bytes_saved_after_incremental_admission=baseline_whole - candidate_whole,
                net_logical_bytes_saved_after_worst_case_cold_admission=(baseline['ram_miss_logical_source_bytes']
                    - candidate['ram_miss_logical_source_bytes'] - selected_bytes),
                owned_payload_peak_bytes=peak, total_budget_respected=True,
                cohort_updates_during_holdout=0, duplicate_payload_bytes=0))
        comparisons.append(dict(total_payload_budget_bytes=total, training=train,
            baseline_holdout=baseline, baseline_whole_window_logical_source_bytes=baseline_whole,
            baseline_owned_payload_peak_bytes=baseline['owned_payload_peak_bytes'], alternatives=alternatives))
    return comparisons


def analyse(text, metadata, *, train_tokens, budgets=(4 * GIB, 8 * GIB, 12 * GIB, 16 * GIB),
            total_ram_budgets=(16 * GIB, 20 * GIB), **window):
    baseline = reconstruct_misses(text, metadata, **window)
    _bounded(train_tokens, 'training tokens', 1, baseline['positions']['count'] - 1)
    if (type(budgets) not in (list, tuple) or not budgets or len(budgets) > 16
            or any(type(value) is not int for value in budgets) or len(set(budgets)) != len(budgets)):
        raise ValueError('RAM budgets must be a bounded distinct integer list')
    split = baseline['positions']['first'] + train_tokens
    training = [row for row in baseline['rows'] if row['p0'] < split]
    holdout = [row for row in baseline['rows'] if row['p0'] >= split]
    training_bytes = sum(row['logical_source_bytes'] for row in training)
    holdout_bytes = sum(row['logical_source_bytes'] for row in holdout)
    cohorts = []
    for budget in budgets:
        cohort = freeze_cohort(training, budget)
        cohort['source_identity'] = baseline['source_identity']
        cohort['cohort_sha256'] = hashlib.sha256((baseline['source_identity']
            + ':' + cohort['selection_sha256']).encode()).hexdigest()
        keys = {(item['layer'], item['expert']) for item in cohort['selected']}
        served_count = served_bytes = 0
        per_group = {group['id']: dict(gpu_misses=0, gpu_miss_bytes=0,
                     potentially_served_misses=0, potentially_served_bytes=0) for group in baseline['groups']}
        for row in holdout:
            served = sum((row['layer'], expert) in keys for expert in row['gpu_miss_experts'])
            served_count += served
            served_bytes += served * row['expert_bytes']
            values = per_group[row['group']]
            values['gpu_misses'] += len(row['gpu_miss_experts'])
            values['gpu_miss_bytes'] += row['logical_source_bytes']
            values['potentially_served_misses'] += served
            values['potentially_served_bytes'] += served * row['expert_bytes']
        admission = cohort['selected_payload_bytes']
        cohort.update(holdout_potentially_served_misses=served_count,
            holdout_potentially_served_logical_bytes=served_bytes,
            holdout_unserved_logical_bytes=holdout_bytes - served_bytes,
            holdout_potential_served_fraction=served_bytes / holdout_bytes if holdout_bytes else 0,
            cold_admission_logical_source_bytes=admission,
            cold_training_plus_admission_logical_source_bytes=training_bytes + admission,
            selected_training_gpu_misses=sum(item['training_gpu_misses'] for item in cohort['selected']),
            selected_training_logical_read_bytes=sum(item['training_gpu_misses'] * item['payload_bytes'] for item in cohort['selected']),
            holdout_potential_bytes_minus_cold_admission=served_bytes - admission,
            holdout_gpu_upload_payload_bytes_unchanged=holdout_bytes,
            cohort_updates_during_holdout=0, per_group=per_group)
        cohorts.append(cohort)
    fair = compare_same_total_ram(training, holdout, cohorts, baseline['source_identity'], total_ram_budgets)
    return dict(schema_version=2, scope='same_total_counterfactual_RAM_LRU_and_frozen_cohort_behind_exact_observed_GPU_LRU',
        baseline=baseline, split=dict(training_first=baseline['positions']['first'], training_last=split - 1,
            training_tokens=train_tokens, holdout_first=split, holdout_last=baseline['positions']['last'],
            holdout_tokens=baseline['positions']['count'] - train_tokens),
        training=dict(gpu_misses=sum(len(row['gpu_miss_experts']) for row in training),
            logical_source_bytes=training_bytes, gpu_hits_excluded_from_scores=sum(row['gpu_hits'] for row in training)),
        holdout=dict(gpu_misses=sum(len(row['gpu_miss_experts']) for row in holdout),
            logical_source_bytes=holdout_bytes, gpu_hits=sum(row['gpu_hits'] for row in holdout)),
        cohorts=cohorts, static_cohorts_are_payload_only_reference=True, same_total_ram=fair,
        same_total_ram_scope='Counterfactual cold private whole-triple caches with equal payload budgets; not the actual OS cache, readahead, file-page sharing, physical RAM residency or SSD service',
        every_layer_timer_matched=True, selection_uses_holdout=False,
        physical_admission_qualified=False, guaranteed_residency=False, performance_qualified=False,
        physical_io_qualified=False, numerical_quality_qualified=False, limitations=LIMITATIONS)


def _read_bounded(path, limit, expected_sha256):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Unbounded or unrecognized audited input: ' + str(path))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = _read_bounded(args.trace, MAX_TRACE_BYTES, V27_TRACE_SHA256)
    metadata_raw = _read_bounded(args.metadata, MAX_METADATA_BYTES, V27_METADATA_SHA256)
    result = analyse(raw.decode('utf-8-sig', errors='strict'),
        json.loads(metadata_raw.decode('utf-8-sig', errors='strict')), **V27_WINDOW)
    if any(result['baseline'][key] != value for key, value in V27_BASELINE.items()):
        raise ValueError('Reconstructed v27 cold baseline disagrees with the independently audited totals')
    dependencies = ('glm_route_retention_replay.py', 'glm-route-cache-replay.py', 'local-model-preflight.py')
    result['provenance'] = dict(profile='v27_D128_all_fresh_groups', trace=str(args.trace.resolve()),
        trace_sha256=V27_TRACE_SHA256, metadata=str(args.metadata.resolve()), metadata_sha256=V27_METADATA_SHA256,
        checkpoint_opened=False, tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in dependencies})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(every_layer_timer_matched=True, training=result['training'], holdout=result['holdout'],
        cohorts=[{key: item[key] for key in ('budget_bytes', 'selected_payload_bytes',
            'holdout_potentially_served_logical_bytes', 'cold_admission_logical_source_bytes')}
            for item in result['cohorts']], same_total_ram=[dict(
                total_payload_budget_bytes=item['total_payload_budget_bytes'],
                baseline_holdout_ram_miss_logical_bytes=item['baseline_holdout']['ram_miss_logical_source_bytes'],
                alternatives=[{key: arm[key] for key in ('cohort_budget_bytes',
                    'whole_window_logical_source_bytes_including_admission',
                    'net_logical_bytes_saved_after_incremental_admission')}
                    for arm in item['alternatives']]) for item in result['same_total_ram']],
        performance_qualified=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
