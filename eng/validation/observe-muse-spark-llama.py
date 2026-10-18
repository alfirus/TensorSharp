#!/usr/bin/env python3
"""Verify an actual Linux llama-server launch and capture read-only provenance.

Run after launching the prepared Muse Spark reference server. This tool checks
the live process, exact profile arguments, template and executable/library
hashes, then waits for GET /v1/models. It never submits inference requests.
Use a new output file for each launch; startup/prewarm are not benchmark time.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import urllib.request
from urllib.parse import urlsplit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def argument(argv, name):
    positions = [index for index, value in enumerate(argv) if value == name]
    require(len(positions) == 1 and positions[0] + 1 < len(argv),
            f'Expected one explicit {name} argument')
    return argv[positions[0] + 1]


def verify_arguments(argv, planned):
    settings = planned['settings']
    require(Path(argv[0]).name == 'llama-server', 'PID is not llama-server')
    ranges = settings['layer_ranges']
    layers = [last - first + 1 for first, last in ranges]
    require(len(layers) == 5 and sum(layers) == 62, 'Expected five-device 62-layer profile')
    layers[-1] += 1  # llama.cpp also places the output projection.
    url = urlsplit(planned['server_url'])
    expected = {
        '--n-cpu-moe': settings['cpu_expert_layer_count'],
        '--threads': settings['cpu_threads'], '--threads-batch': settings['cpu_threads'],
        '--cache-type-k': settings['key_cache_dtype'], '--cache-type-v': settings['value_cache_dtype'],
        '--parallel': settings['max_running_requests'], '--ctx-size': settings['aggregate_kv_pool_tokens'],
        '--kv-unified-per-slot': settings['context_per_request'],
        '--batch-size': settings['max_batched_tokens'], '--ubatch-size': settings['prefill_chunk_tokens'],
        '--predict': settings['max_output_tokens'], '--tensor-split': ','.join(map(str, layers)),
        '--device': 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4', '--gpu-layers': 'all', '--split-mode': 'layer',
        '--mmproj-device': 'CUDA0', '--override-tensor': r'token_embd\.weight=CUDA0',
        '--fit': 'off', '--flash-attn': 'on', '--reasoning': 'off',
        '--host': url.hostname, '--port': url.port,
    }
    for name, value in expected.items():
        require(argument(argv, name) == str(value), f'{name} differs from the planned profile')
    for flag in ('--kv-unified', '--cont-batching', '--jinja'):
        require(argv.count(flag) == 1, f'Missing or repeated {flag}')
    cache_flags = [value for value in argv if value in ('--cache-prompt', '--no-cache-prompt')]
    require(type(settings.get('prefix_cache')) is bool, 'Explicit prefix_cache setting required')
    expected_cache_flag = '--cache-prompt' if settings['prefix_cache'] else '--no-cache-prompt'
    require(bool(cache_flags) and cache_flags[-1] == expected_cache_flag,
            'Final effective prefix-cache flag differs from planned profile')
    if settings['prefix_cache'] is False:
        # cache_prompt=false skips prefix reuse, but llama's independent RAM
        # prompt cache can still save/load idle slot state. These copies are
        # unnecessary in this profile and are not measured by cached_tokens.
        options = planned.get('engine_options', {})
        require(type(options.get('prompt_cache_ram_mib')) is int and options['prompt_cache_ram_mib'] == 0
                and options.get('cache_idle_slots') is False,
                'No-cache reference requires explicit disabled RAM/idle-slot cache provenance')
        require(argument(argv, '--cache-ram') == '0', 'No-cache reference must disable the RAM prompt cache')
        idle_flags = [value for value in argv if value in ('--cache-idle-slots', '--no-cache-idle-slots')]
        require(bool(idle_flags) and idle_flags[-1] == '--no-cache-idle-slots',
                'No-cache reference must disable idle-slot state saves')
        expected['--cache-ram'] = 0
        expected['--no-cache-idle-slots'] = True
    # Reject aliases/negative flags that could override a checked long option.
    conflicting = {'-c', '-b', '-ub', '-t', '-tb', '-np', '-ngl', '-ts',
                   '--no-cont-batching', '--no-kv-unified'}
    require(not conflicting.intersection(argv), 'Conflicting profile override/alias in argv')
    first = [name for name in planned['weights_sha256'] if '-00001-of-' in name]
    require(len(first) == 1, 'Expected one verified first shard')
    require(argument(argv, '--model').replace('\\', '/').endswith('/' + first[0]),
            'Actual model path does not select the verified checkpoint')
    first_path = Path(planned['tokenizer_metadata']['file'])
    require(Path(argument(argv, '--model')).as_posix() == first_path.as_posix(),
            'Actual model path differs from the verified metadata source')
    require(Path(argument(argv, '--mmproj')).as_posix() == (first_path.parent.parent / 'mmproj-Q8_0.gguf').as_posix(),
            'Unexpected projector path')
    adapter = planned['reference_prompt_adapter']
    require(argument(argv, '--chat-template-file') == adapter['template'], 'Unexpected template path')
    return expected


def process(pid):
    directory = Path('/proc') / str(pid)
    argv = (directory / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    environment = dict(item.split('=', 1) for item in
        (directory / 'environ').read_bytes().decode().split('\0') if '=' in item)
    stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    return argv, environment, int(stat[19])


def verify_binaries(planned):
    verified, resolved = {}, {}
    require(bool(planned.get('binary_hashes')), 'No compiled binary manifest')
    for name, expected in planned['binary_hashes'].items():
        path = Path(name)
        target = path.resolve(strict=True)
        if target not in resolved:
            resolved[target] = sha256(target)
        actual = resolved[target]
        require(actual == expected, f'Compiled binary changed: {name}')
        verified[name] = actual
    return verified


def owns_listen_port(pid, port, proc_root=Path('/proc')):
    """Do not accept another process's healthy endpoint while this one loads."""
    directory = proc_root / str(pid)
    inodes = set()
    for descriptor in (directory / 'fd').iterdir():
        try:
            target = os.readlink(descriptor)
        except OSError:
            continue
        if target.startswith('socket:[') and target.endswith(']'):
            inodes.add(target[8:-1])
    for name in ('tcp','tcp6'):
        path = directory / 'net' / name
        if not path.exists():
            continue
        for row in path.read_text().splitlines()[1:]:
            fields = row.split()
            if len(fields) >= 10 and fields[3] == '0A' and fields[9] in inodes:
                if int(fields[1].rsplit(':',1)[1],16) == port:
                    return True
    return False


def main(engine='llamacpp', profile_verifier=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planned', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    require(args.timeout > 0 and args.pid > 0, 'Positive timeout and PID required')
    require(not args.output.exists(), 'Use a new output path; observed launch evidence is not overwritten')
    planned = json.loads(args.planned.read_text(encoding='utf-8'))
    require(planned.get('launch_observed') is False, 'Input must be planned provenance')
    argv, env, start = process(args.pid)
    if profile_verifier is None:
        checked_arguments = verify_arguments(argv, planned)
        require(env.get('OMP_NUM_THREADS') == str(planned['settings']['cpu_threads']), 'OMP thread count differs')
        checked_environment = {'OMP_NUM_THREADS': env['OMP_NUM_THREADS']}
    else:
        checked_arguments, checked_environment = profile_verifier(argv, env, planned)
    require(args.server_log.is_file() and args.server_log.stat().st_size > 0, 'Startup log missing/empty')
    template = planned.get('reference_prompt_adapter')
    if template:
        require(sha256(Path(template['template'])) == template['template_sha256'], 'Reference template changed')
    verified = verify_binaries(planned)
    executable = Path(f'/proc/{args.pid}/exe').resolve(strict=True)
    if engine == 'tensorsharp':
        host_dll = next(Path(value) for value in argv if Path(value).name == 'TensorSharp.Server.Host.dll')
        cwd = Path(f'/proc/{args.pid}/cwd').resolve(strict=True)
        target = host_dll if host_dll.is_absolute() else cwd / host_dll
        require(any(Path(name).resolve() == target.resolve() for name in verified),
                'Live managed host is absent from binary manifest')
    else:
        require(any(Path(name).resolve() == executable for name in verified), 'Live executable is absent from binary manifest')
    runtime_executable_hash = sha256(executable)
    deadline = time.monotonic() + args.timeout
    last_error = None
    while time.monotonic() < deadline:
        current_argv, current_environment, current_start = process(args.pid)
        require(start == current_start and argv == current_argv, 'Process identity changed')
        require(all(current_environment.get(key) == value for key, value in checked_environment.items()),
                'Checked environment changed before readiness')
        try:
            require(owns_listen_port(args.pid,urlsplit(planned['server_url']).port),
                    'Observed PID does not own the expected listening port')
            with urllib.request.urlopen(planned['server_url'].rstrip('/') + '/v1/models', timeout=3) as response:
                models = json.load(response)
            require(any('muse-spark' in str(model.get('id', '')).lower() for model in models.get('data', [])),
                    'Ready endpoint did not advertise Muse Spark')
            break
        except (OSError, ValueError) as error:
            last_error = str(error)
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    else:
        raise TimeoutError('Readiness not observed: ' + str(last_error))
    observed = datetime.now(timezone.utc).isoformat()
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    started = datetime.fromtimestamp(boot + start / os.sysconf('SC_CLK_TCK'), timezone.utc).isoformat()
    result = dict(planned)
    result.update(launch_recipe=planned['server_command'], launch_observed=True,
        launch_observed_utc=observed, server_pid=args.pid, server_command=shlex.join(argv),
        server_log=str(args.server_log), observed_environment=checked_environment,
        observed_checked_arguments=checked_arguments, ready_response=models,
        binaries_reverified_at_launch=verified,
        runtime_executable={'path':str(executable), 'sha256':runtime_executable_hash},
        phase_timestamps={'process_start_kernel_estimate_utc': started, 'http_ready_first_observed_utc': observed},
        phase_note='Readiness polling and binary verification precede inference. Record CPU expert prewarm separately; no inference requests were sent by this observer.',
        status='Actual arguments, binary/template identity and HTTP readiness verified; quality and performance not implied.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pid': args.pid, 'launch_observed': True, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
