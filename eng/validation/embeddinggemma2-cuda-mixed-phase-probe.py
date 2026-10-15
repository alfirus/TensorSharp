#!/usr/bin/env python3
"""Untimed exact-ragged CUDA phase/lifetime/isolation diagnostic, not a benchmark."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from embeddinggemma2_evidence import command_artifacts

spec = importlib.util.spec_from_file_location('embedding_bench', ROOT / 'benchmarks/EmbeddingBench/embedding_bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
spec = importlib.util.spec_from_file_location('embedding_http', Path(__file__).with_name('embeddinggemma2-http.py'))
http = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', type=json.loads, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18380')
    parser.add_argument('--model', default='embeddinggemma-2-Q8_0')
    parser.add_argument('--model-file', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'InferenceWeb.Tests/Fixtures/EmbeddingTokenizer/embeddinggemma2-tokenization.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('Require at least one cold and one repeated exact geometry')
    for flag in ('TS_EMBEDDING_CUDA_PHASES=1', 'TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION=1'):
        if flag not in args.command:
            parser.error('Require explicit command flag ' + flag)
    fixture = json.loads(args.fixture.read_text())
    model_hash = hashlib.sha256(args.model_file.read_bytes()).hexdigest()
    if fixture['architecture'] != 'gemma-embedding2' or fixture['model_sha256'] != model_hash:
        raise ValueError('Require the exact model and independently verified tokenizer fixture')
    seed = next(case['tokens'] for case in fixture['cases'] if case['text'] == http.TEXTS[1])
    bos, eos = fixture['bos_token_id'], fixture['eos_token_id']
    content = seed[1:-1]
    if seed[0] != bos or seed[-1] != eos or not content:
        raise ValueError('Require verified nonempty BOS/EOS tokenizer seed')
    lengths = [31, 511, 513, 1025]
    inputs = [[bos] + [content[index % len(content)] for index in range(length - 2)] + [eos] for length in lengths]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(completed=False, command=args.command, lengths=lengths, inputs=inputs,
                  model_sha256=model_hash,
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(), repeats=args.repeats,
                  binary_artifacts=command_artifacts(bench.command_artifacts, args.command),
                  limitations=['Untimed phase diagnostic with GPU events; not formal latency evidence.',
                               'GPU intervals include host-issue idle and exclude initial queued HtoD; final CPU normalization is outside phases.'])
    destination = args.output / 'report.json'
    bench.KEEP_ALIVE = True
    with (args.output / 'server.log').open('w') as log:
        process = subprocess.Popen(args.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            start = time.monotonic()
            while time.monotonic() - start < 180:
                if process.poll() is not None:
                    raise RuntimeError('Server exited at startup')
                try:
                    status, discovery = bench.request(args.url, '/api/models')
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError('Server startup')
            if discovery['loadedBackend'] != 'cuda':
                raise ValueError('Require actual direct CUDA backend')
            report['actual_backend'] = discovery['loadedBackend']
            steps = []
            for _ in range(args.repeats):
                vectors, usage = bench.embedding(args.url, args.model, inputs)
                bench.assert_vectors(vectors, 768)
                if usage['prompt_tokens'] != sum(lengths):
                    raise ValueError('Exact ragged token accounting changed')
                steps.append(vectors)
            singles = [bench.embedding(args.url, args.model, [tokens])[0][0] for tokens in inputs]
            reversed_vectors = bench.embedding(args.url, args.model, inputs[::-1])[0][::-1]
            comparisons = dict(single_batch_cosines=[bench.cosine(a, b) for a, b in zip(singles, steps[-1])],
                               reversed_cosines=[bench.cosine(a, b) for a, b in zip(reversed_vectors, steps[-1])],
                               repeat_cosines=[bench.cosine(a, b) for step in steps[:-1] for a, b in zip(step, steps[-1])],
                               min_cosine=.99999)
            report.update(steps=steps, single_vectors=singles, comparisons=comparisons)
            report['quality_pass'] = all(value >= comparisons['min_cosine'] for key, values in comparisons.items()
                                         if isinstance(values, list) for value in values)
        finally:
            for connection in getattr(bench.CONNECTIONS, 'connections', {}).values():
                connection.close()
            bench.CONNECTIONS.connections = {}
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            report['server_exit_code'] = process.returncode
            report['completed'] = process.returncode == 0 and 'quality_pass' in report
            destination.write_text(json.dumps(report, indent=2) + '\n')
    if not report['completed'] or not report['quality_pass']:
        raise RuntimeError('Exact ragged quality/process-cleanup diagnostic failed')
    print(json.dumps(dict(completed=report['completed'], comparisons=report['comparisons']), indent=2))


if __name__ == '__main__':
    main()
