#!/usr/bin/env python3
"""Observe a prepared TensorSharp Muse Spark launch without sending inference.

Uses the same --planned/--pid/--server-log/--output/--timeout arguments as the
llama observer. engine_options.environment must explicitly state stream,
source-page release and residual-fusion controls; those remain outside common
comparison settings. Binary hashes and process identity are checked before
readiness is accepted. Never run this against an unrelated server.
"""
import importlib.util
from pathlib import Path
from urllib.parse import urlsplit

# Share process/readiness/hash checks with the reference observer. Its hyphenated
# executable filename needs an explicit module loader when reused as a library.
SPEC = importlib.util.spec_from_file_location('spark_launch_observer',
    Path(__file__).with_name('observe-muse-spark-llama.py'))
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def verify_profile(argv, env, planned):
    require, argument = observer.require, observer.argument
    settings = planned['settings']
    require(Path(argv[0]).name == 'dotnet', 'PID is not the dotnet host')
    require(sum(Path(value).name == 'TensorSharp.Server.Host.dll' for value in argv) == 1,
            'Expected exactly one TensorSharp server assembly')
    require(settings['key_cache_dtype'] == settings['value_cache_dtype'], 'TS profile requires equal K/V precision')
    layers = [last - first + 1 for first,last in settings['layer_ranges']]
    require(len(layers) == 5 and sum(layers) == 62, 'Expected five-device 62-layer profile')
    first = Path(planned['tokenizer_metadata']['file'])
    url = urlsplit(planned['server_url'])
    expected = {'--model':str(first), '--mmproj':str(first.parent.parent / 'mmproj-Q8_0.gguf'),
        '--backend':'ggml_cuda', '--layer-split':'5', '--n-cpu-moe':settings['cpu_expert_layer_count'],
        '--cpu-moe-threads':settings['cpu_threads'], '--kv-cache-dtype':settings['key_cache_dtype'],
        '--host':url.hostname, '--port':url.port, '--max-tokens':settings['max_output_tokens'],
        '--sampling-precedence':'request'}
    for name,value in expected.items():
        actual = argument(argv,name)
        if name in ('--model','--mmproj'):
            require(Path(actual).as_posix() == Path(value).as_posix(), f'{name} differs from verified path')
        else:
            require(actual == str(value), f'{name} differs from planned setting')
    require(settings.get('speculation') == 'disabled' and argv.count('--no-spec') == 1,
            'This observer requires explicit disabled speculation')
    require(argv.count('--no-webui') == 1, 'API-only startup flag required')
    require(type(settings.get('prefix_cache')) is bool, 'Explicit prefix_cache setting required')
    require(('--no-prefix-cache' in argv) == (not settings['prefix_cache']), 'Prefix-cache flag differs')
    require(argv.count('--no-prefix-cache') <= 1, 'Repeated prefix-cache flag')
    pool = settings['aggregate_kv_pool_tokens']
    require(type(pool) is int and pool > 0 and pool % 16 == 0, 'Expected aggregate pool in 16-token blocks')
    require(env.get('TS_SCHED_BLOCK_SIZE') in (None,'0','16'), 'Unexpected scheduler block size')
    expected_env = {'MAX_CONTEXT':settings['context_per_request'], 'TS_SCHED_NUM_BLOCKS':pool//16,
        'TS_SCHED_MAX_RUNNING_SEQS':settings['max_running_requests'],
        'TS_SCHED_MAX_BATCHED_TOKENS':settings['max_batched_tokens'],
        'TS_MUSE_SPARK_PREFILL_CHUNK':settings['prefill_chunk_tokens'],
        'TS_SCHED_PREFILL_CHUNK':settings['prefill_chunk_tokens'],
        'TS_SCHED_SOLO_PREFILL_CHUNK':settings['prefill_chunk_tokens'],
        'TS_MUSE_SPARK_LAYER_SPLIT':','.join(map(str,layers)),
        'TS_CPU_MOE_THREADS':settings['cpu_threads'], 'TS_GGML_CPU_THREADS':settings['cpu_threads'],
        'OMP_NUM_THREADS':settings['cpu_threads']}
    options = planned.get('engine_options',{}).get('environment',{})
    controls = ('TS_MUSE_SPARK_STREAM_EXPERTS','TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES','TS_MUSE_SPARK_FUSE_RESIDUALS')
    require(all(options.get(name) in ('0','1') for name in controls), 'Explicit 0/1 engine controls required')
    require(all(isinstance(key,str) and isinstance(value,str) for key,value in options.items()),
            'engine_options.environment must contain string values')
    for key,value in options.items():
        require(key not in expected_env or str(expected_env[key]) == value,
                'Engine option conflicts with common setting: ' + key)
        expected_env[key] = value
    expected_env = {key:str(value) for key,value in expected_env.items()}
    for key,value in expected_env.items():
        require(env.get(key) == value, 'Observed environment differs: ' + key)
    return expected, expected_env


if __name__ == '__main__':
    observer.main(engine='tensorsharp',profile_verifier=verify_profile)
