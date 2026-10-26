#!/usr/bin/env python3
"""Sample device/container evidence; sampled values are not allocator peaks.

Uses the existing cgroup reader. Container counters include every process,
including other tenants/tools visible in this container, and must not be
attributed to one request without its independent time boundaries.
"""
import argparse
import csv
import json
from pathlib import Path
import signal
import subprocess
import time

from inference_resource_limits import cgroup_snapshot


def query(fields, kind='gpu'):
    command = ['nvidia-smi', f'--query-{kind}=' + ','.join(fields), '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        if result.returncode:
            return {'available': False, 'error': result.stderr.strip(), 'exit_code': result.returncode}
        return {'available': True, 'rows': [dict(zip(fields, (v.strip() for v in row)))
                                            for row in csv.reader(result.stdout.splitlines())]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'error': str(error)}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--duration', type=float, default=0)
    args = parser.parse_args()
    if args.interval < .25 or args.duration < 0:
        parser.error('interval must be >= .25s and duration nonnegative')
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with args.output.open('x', encoding='utf-8') as stream:
        while running and (not args.duration or time.monotonic() - start < args.duration):
            tick = time.monotonic()
            apps = query(['pid', 'process_name', 'used_gpu_memory'], 'compute-apps')
            pids = sorted({r['pid'] for r in apps.get('rows', []) if r.get('pid', '').isdigit()})
            sample = {'unix_time': time.time(), 'elapsed_seconds': tick - start,
                      'gpu': query(['index', 'uuid', 'memory.used', 'utilization.gpu', 'power.draw', 'clocks.sm']),
                      'compute_processes': apps, 'cgroup': cgroup_snapshot(),
                      'cgroup_cpu_stat': read('/sys/fs/cgroup/cpu.stat') or read('/sys/fs/cgroup/cpu/cpu.stat'),
                      'processes': {pid: {name: read(f'/proc/{pid}/{name}') for name in ('status', 'io', 'stat')}
                                    for pid in pids}}
            stream.write(json.dumps(sample) + '\n')
            stream.flush()
            time.sleep(max(0, args.interval - (time.monotonic() - tick)))


if __name__ == '__main__':
    main()
