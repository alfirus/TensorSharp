#!/usr/bin/env python3
"""Validate bounded CUPTI activity accounting and summarize diagnostic kernel times."""
import argparse
import json
import math
from pathlib import Path


def summarize(report):
    if report.get('completed') is not True or report.get('diagnostic') is not True or report.get('hardware_counters') is not False:
        raise ValueError('A complete activity-only diagnostic is required')
    for field in ('dropped', 'incomplete', 'callback_exceptions', 'truncated'):
        if report.get(field) != 0:
            raise ValueError(f'{field} must be zero')
    if report.get('errors') != []:
        raise ValueError('CUPTI errors are present')
    events, kernels = report['events'], report['kernels']
    if not events or report.get('kernel_count') != len(events):
        raise ValueError('Kernel event count mismatch')
    observed = {}
    for event in events:
        begin, end = event['start_ns'], event['end_ns']
        if not isinstance(begin, int) or not isinstance(end, int) or end <= begin:
            raise ValueError('Invalid event interval')
        item = observed.setdefault(event['name'], dict(count=0, sum_ns=0))
        item['count'] += 1
        item['sum_ns'] += end - begin
    if len(kernels) != len(observed) or len({k['name'] for k in kernels}) != len(kernels):
        raise ValueError('Kernel aggregation names mismatch')
    categories = {}
    for kernel in kernels:
        item = observed[kernel['name']]
        if kernel['count'] != item['count'] or not math.isclose(kernel['sum_ms'], item['sum_ns'] / 1e6, abs_tol=1e-5):
            raise ValueError('Kernel aggregation count or time mismatch')
        name = kernel['name']
        category = 'owned_vision_attention' if ('vision_kernel<' in name or 'vision_kernel_minblocks3<' in name) else 'gemm' if ('gemm' in name.lower() or 'splitKreduce' in name) else 'other'
        group = categories.setdefault(category, dict(count=0, sum_kernel_ms=0))
        group['count'] += item['count']
        group['sum_kernel_ms'] += item['sum_ns'] / 1e6
    total = sum(item['sum_ns'] for item in observed.values()) / 1e6
    if not math.isclose(total, report['sum_kernel_ms'], abs_tol=1e-5):
        raise ValueError('Total kernel duration mismatch')
    for group in categories.values():
        group['fraction_of_sum'] = group['sum_kernel_ms'] / total
    return dict(valid_accounting=True, diagnostic=True, paired_performance_claim=False,
                kernel_count=len(events), sum_kernel_ms=total, categories=categories,
                top_kernels=sorted(kernels, key=lambda k: k['sum_ms'], reverse=True)[:10],
                scope='Includes warmup; sums overlapping device intervals and excludes CPU/copies/service overhead.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    result['input'] = str(args.input.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
