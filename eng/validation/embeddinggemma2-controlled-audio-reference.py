#!/usr/bin/env python3
"""Explicit llama-clamp audio diagnostic against the independent F32 model.

This never qualifies the default HF frontend. It checks both completed engines
from a controlled media pair without replacing failed cross-engine gates.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_inputs(pair, audio, model_hash):
    if not pair.get('completed'):
        raise ValueError('Both owned pair engines must have completed')
    if pair['model']['sha256'] != model_hash or pair['projector']['sha256'] != audio['projector_sha256']:
        raise ValueError('Controlled pair text model/projector differs from the independent oracle')
    if audio.get('audio_log_policy') != 'llama_clamp' or audio.get('past_distance_inclusive') != 11:
        raise ValueError('Require explicitly clamped mel and the pinned exclusive-left distance11 oracle')
    case = next(case for case in pair['inputs'] if case['name'] == 'audio')
    if len(case['parts']) != 1 or case['parts'][0]['kind'] != 'audio':
        raise ValueError('This diagnostic requires the unprefixed single waveform sequence')
    wave_name = Path(case['parts'][0]['value']).name
    if not any(Path(record['path']).name == wave_name and record['sha256'] == audio['waveform_sha256']
               for record in pair['media']):
        raise ValueError('Controlled pair waveform differs from the independent oracle')
    for name in ('tensorsharp', 'llama'):
        engine = pair['engines'][name]
        if not engine.get('completed') or engine.get('server_exit_code') != 0:
            raise ValueError(name + ' did not complete with a zero graceful server exit')
        row = engine['benchmarks']['audio']
        if row['parts'] != case['parts']:
            raise ValueError(name + ' actual waveform sequence differs from the manifest')
    command = pair['engines']['tensorsharp']['command']
    if 'TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp' not in command:
        raise ValueError('Candidate command must explicitly select the controlled clamp policy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--audio-oracle', type=Path, required=True)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-cosine', type=float, default=.9999)
    parser.add_argument('--max-absolute-error', type=float, default=.002)
    args = parser.parse_args()
    if not math.isfinite(args.min_cosine) or not -1 <= args.min_cosine <= 1 or not math.isfinite(args.max_absolute_error) or args.max_absolute_error < 0:
        parser.error('Require finite valid numerical gates')
    pair, audio = json.loads(args.pair.read_text()), json.loads(args.audio_oracle.read_text())
    model_hash = digest(args.model)
    validate_inputs(pair, audio, model_hash)
    source = Path(__file__).resolve().parents[1] / 'embeddinggemma2-reference.py'
    spec = importlib.util.spec_from_file_location('embedding_reference', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = reference.Encoder(args.model)
    vocabulary = model.metadata('tokenizer.ggml.tokens')
    leading = [int(model.metadata('tokenizer.ggml.bos_token_id')), vocabulary.index('<|audio>')]
    trailing = [vocabulary.index('<audio|>'), int(model.metadata('tokenizer.ggml.eos_token_id'))]
    soft = np.asarray(audio['soft_rows'], dtype=np.float32)
    if soft.ndim != 2 or list(soft.shape) != audio['shape'] or soft.shape[1] != model.dim or not np.all(np.isfinite(soft)):
        raise ValueError('Require complete finite independent audio rows')
    inputs = np.concatenate((model.token_rows(leading), soft, model.token_rows(trailing)), axis=0)
    expected = model.encode_inputs(inputs)
    if expected.size != 768 or not np.all(np.isfinite(expected)) or abs(float(np.linalg.norm(expected)) - 1) > 1e-5:
        raise ValueError('Independent model must return a finite unit 768-dimensional embedding')
    comparisons = {}
    for name in ('tensorsharp', 'llama'):
        row = pair['engines'][name]['benchmarks']['audio']
        actual = np.asarray(row['embedding'], dtype=np.float64)
        if row['prompt_tokens'] != inputs.shape[0] or actual.shape != expected.shape or not np.all(np.isfinite(actual)) or abs(float(np.linalg.norm(actual)) - 1) > 1e-5:
            raise ValueError(name + ' token count or normalized output differs from the complete input')
        left = expected.astype(np.float64)
        cosine = float(np.dot(left, actual) / (np.linalg.norm(left) * np.linalg.norm(actual)))
        error = float(np.max(np.abs(left - actual)))
        comparisons[name] = dict(cosine=cosine, max_absolute_error=error,
                                 passed=cosine >= args.min_cosine and error <= args.max_absolute_error)
    report = dict(passed=all(check['passed'] for check in comparisons.values()), comparisons=comparisons,
        thresholds=dict(min_cosine=args.min_cosine, max_absolute_error=args.max_absolute_error),
        model_sha256=model_hash, projector_sha256=audio['projector_sha256'], waveform_sha256=audio['waveform_sha256'],
        audio_oracle=dict(path=str(args.audio_oracle.resolve()), sha256=digest(args.audio_oracle)),
        pair=dict(path=str(args.pair.resolve()), sha256=digest(args.pair)),
        encoder_source_sha256=digest(source), source_revision=audio['source_revision'],
        oracle_source_hashes={name: digest(Path(__file__).with_name(name)) for name in
                             ('embeddinggemma2-audio-reference.py', 'generate-gemma4-audio-frontend.py')},
        media=pair['media'], binary_artifacts=pair['binary_artifacts'],
        dependency_revisions=pair['dependency_revisions'],
        dependency_tracked_changes=pair['dependency_tracked_changes'],
        server_exit_codes={name: pair['engines'][name]['server_exit_code'] for name in ('tensorsharp', 'llama')},
        original_pair_passed=pair['passed'], original_pair_quality_pass=pair['quality_pass'],
        leading_token_ids=leading, trailing_token_ids=trailing, total_input_rows=int(inputs.shape[0]),
        input_rows_sha256=hashlib.sha256(inputs.astype('<f4').tobytes()).hexdigest(), embedding=expected.tolist(),
        method='Explicit controlled llama-clamp frontend, independent complete F32 audio and bidirectional text encoders',
        limitations=['Diagnostic clamp policy only; default HF frontend quality remains separately checked.',
                     'Original cross-engine and latency failures remain failed; this report does not replace them.',
                     'Only this waveform, quantized projector, text model and saved actual backends are checked.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], comparisons=comparisons), indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
