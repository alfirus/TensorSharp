#!/usr/bin/env python3
"""Run a pinned Maya executable on exact tokens from a completed Direct probe.

The Maya process is owned and bounded. Each request disables prefix reuse and
forced loop endings. Its own production tokenizer/parser checks the final answer
independently of complete-token parity against the Direct probe reference.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('maya_bench_metrics', HERE / 'multimodel-quality-bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def parse_done(line):
    parts = line.split()
    if len(parts) != 11 or parts[0] != 'DONE':
        raise ValueError('Require the pinned Maya 11-field DONE record')
    row = dict(zip(('generated', 'prompt_tokens', 'prompt_ms', 'decode_ms', 'finish',
                    'drafts_accepted', 'drafts_offered', 'reused', 'hits', 'lookups'), parts[1:]))
    for key in ('generated', 'prompt_tokens', 'drafts_accepted', 'drafts_offered', 'reused', 'hits', 'lookups'):
        row[key] = int(row[key])
        if row[key] < 0:
            raise ValueError('Negative DONE count')
    for key in ('prompt_ms', 'decode_ms'):
        row[key] = float(row[key])
        if not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError('Nonfinite or negative DONE time')
    if row['finish'] not in ('stop', 'length', 'cancel'):
        raise ValueError('Unknown finish reason')
    return row


def qualify(row, reference, eos_ids, semantic=None):
    tokens = row['tokens']
    exact = tokens == reference['generated_tokens']
    eos = bool(tokens) and tokens[-1] in eos_ids and all(t not in eos_ids for t in tokens[:-1])
    evidence_ok = (len(tokens) == row['generated'] and row['prompt_tokens'] == len(reference['prompt_tokens'])
                   and row['reused'] == 0 and row['drafts_offered'] == 0 and row['drafts_accepted'] == 0)
    complete = evidence_ok and eos and row['finish'] == 'stop' and reference.get('passed') is True and reference.get('eos') is True
    passed = complete and (semantic['passed'] if semantic is not None else exact)
    return dict(passed=passed, exact_reference_tokens=exact, checkpoint_eos_observed=eos,
                wire_evidence_valid=evidence_ok, semantic_inferred_from_exact_reference=passed and semantic is None,
                scope='Complete answer-only task with checkpoint EOS; production Maya parser result and exact token parity are separate.')


def load_maya_text(source, pack):
    modules = []
    for name, file in (('maya_tokenizer_for_baseline', Path(source) / 'tools/strata_tokenizer.py'),
                       ('maya_frontend_for_baseline', Path(source) / 'serve/frontend.py')):
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclass resolves postponed annotations here
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[0].Tokenizer.from_dir(Path(pack) / 'tokenizer'), modules[1].OutputParser


def semantic_result(raw, reference, parser_class):
    thinking = reference['rendered_prompt'].endswith('<think>')
    parser = parser_class(thinking=thinking, tools=None, stream_tools=False)
    events = parser.feed(raw) + parser.finish()
    content = ''.join(e.text for e in events if e.kind == 'content')
    reasoning = ''.join(e.text for e in events if e.kind == 'reasoning')
    calls = [dict(name=e.call.name, arguments=e.call.arguments) for e in events if e.kind == 'tool_call']
    closed = not thinking or '</think>' in raw
    return dict(raw_text=raw, content=content, reasoning=reasoning, tool_calls=calls, thinking_closed=closed,
                parser='Maya production serve/frontend.py OutputParser',
                reasoning_exact_reference=reasoning == reference['reasoning_text'],
                passed=closed and not calls and content.strip() == reference['parsed_text'].strip(),
                scope='Exact expected final answer for this answer-only task; reasoning is retained and compared separately, not semantically graded.')


def cleaned_environment(overrides):
    prefixes = ('TS_', 'TENSORSHARP_', 'GGML_', 'GLM_', 'STRATA_', 'OMP_', 'GOMP_', 'NCCL_')
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)
           and k not in ('MAX_CONTEXT', 'KV_CACHE_DTYPE', 'CUDA_MPS_PIPE_DIRECTORY')}
    env.update(overrides)
    return env


def prefill_argument(plan):
    """Preserve the historical auto default; explicit bounds are token counts."""
    value = plan.get('prefill', 'auto')
    if value == 'auto':
        return value
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('prefill must be auto or a positive integer token capacity')
    text = str(value)
    if not text.isascii() or not text.isdecimal() or not 0 < int(text) <= 2147483647:
        raise ValueError('prefill must be auto or a positive integer token capacity')
    return str(int(text))


def verify_inputs(plan):
    source = Path(plan['source_path'])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if revision != plan['source_revision'] or dirty:
        raise ValueError('Maya source revision or clean state changed')
    if bench.sha(plan['exe']) != plan['exe_sha256']:
        raise ValueError('Maya executable changed')
    evidence_path = Path(plan['pack_evidence'])
    if bench.sha(evidence_path) != plan['pack_evidence_sha256']:
        raise ValueError('Pack evidence changed')
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    if evidence.get('exit_code') != 0 or not evidence.get('original_checkpoint_stat_unchanged') or not evidence.get('source_unchanged'):
        raise ValueError('Pack was not successfully qualified')
    pack = Path(plan['pack']).resolve()
    expected_paths = {Path(item['path']).resolve() for item in evidence['generated_files']}
    if not expected_paths or any(not p.is_relative_to(pack) for p in expected_paths):
        raise ValueError('Pack path differs from the qualified derived files')
    # In particular, an unrecorded experts.bin supersedes the GGUF expert views.
    if {p.resolve() for p in pack.rglob('*') if p.is_file()} != expected_paths:
        raise ValueError('Unrecorded or missing pack files (including expert/profile overrides)')
    # Hash derived files, never reread the 93 GB original checkpoint here. This
    # occurs outside timing, but warms the derived-weight cache: no cold-cache claim.
    files = []
    for item in evidence['generated_files']:
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or bench.sha(p) != item['sha256']:
            raise ValueError(f'Derived pack changed: {p}')
        files.append(item)
    for item in evidence['model_stat_after']:
        if Path(item['path']).resolve().parent != pack.parent:
            raise ValueError('The indexed shards are not alongside the qualified pack')
        stat = Path(item['path']).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError(f'Original checkpoint changed: {item["path"]}')
    manifest = evidence['original_checkpoint_identity']
    if bench.sha(manifest['path']) != manifest['sha256']:
        raise ValueError('Original full-checkpoint identity manifest changed')
    reference_path = Path(plan['direct_report'])
    if bench.sha(reference_path) != plan['direct_report_sha256']:
        raise ValueError('Direct reference changed')
    report = json.loads(reference_path.read_text(encoding='utf-8'))
    cases = [c for c in report['quality'] if c['name'] == plan['case']]
    if len(cases) != 1 or cases[0].get('parser') != 'production' or cases[0].get('passed') is not True or cases[0].get('eos') is not True:
        raise ValueError('Require exactly one completed production-parsed reference')
    return cases[0], files


def cleanup_owned(process, watchdog, reader, sampler):
    errors = []
    def attempt(label, action):
        try:
            return action()
        except Exception as error:
            errors.append(f'{label}: {error!r}')
            return None
    def join_thread(label, thread, timeout):
        # A start() allocation failure can leave an assigned, unstarted thread.
        if thread.ident is not None:
            attempt(label + '.join', lambda: thread.join(timeout=timeout))
        if attempt(label + '.is_alive', thread.is_alive) is not False:
            errors.append(label + ' did not exit')
    if watchdog is not None:
        attempt('watchdog.cancel', watchdog.cancel)
        join_thread('watchdog', watchdog, 10)
    exit_code = None
    if process is not None:
        if attempt('process.poll', process.poll) is None:
            attempt('process.kill', process.kill)
            attempt('process.wait', lambda: process.wait(timeout=20))
        exit_code = attempt('process.final_poll', process.poll)
        if exit_code is None:
            errors.append('Owned process did not exit')
    if reader is not None:
        join_thread('reader', reader, 10)
    if sampler is not None:
        attempt('sampler.stop', sampler.stop.set)
        join_thread('sampler', sampler.thread, 15)
    return exit_code, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive and finite')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['repetitions'] < 1 or plan['max_new'] < 1:
        parser.error('positive repetitions and max_new required')
    try:
        prefill = prefill_argument(plan)
    except ValueError as error:
        parser.error(str(error))
    out = bench.evidence_directory(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, process_success=False, quality_passed=False, rows=[], plan=plan,
                  harness_sha256=bench.sha(__file__), helper_sha256=bench.sha(bench.__file__), errors=[],
                  limitations=['Compatibility pack converts listed small projections to BF16; original expert bytes unchanged.',
                    'Same hardware, not an equal physical hard-budget assertion.',
                    'Derived files are hashed before timing; OS file cache is uncontrolled. First request is not a cold-disk claim.',
                    'Sequential requests in one process are not independent balanced A/B performance trials.',
                    'One-second RSS and board VRAM sampling can miss peaks; board VRAM includes other processes; observer overhead unquantified.',
                    'Exact loaded executable identity checked once after READY; shared CUDA DLL identities not enumerated.',
                    'DONE decode time includes first-token sampling from prefill and only generated-1 model forwards; last EOS is not forwarded.'])
    def save():
        p = out / 'report.json.tmp'
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        p.replace(out / 'report.json')
    process = sampler = watchdog = reader = None
    timed_out = threading.Event()
    lines = queue.Queue()
    save()
    try:
        reference, files = verify_inputs(plan)
        tokenizer, parser_class = load_maya_text(plan['source_path'], plan['pack'])
        if tokenizer.encode(reference['rendered_prompt'], parse_special=True) != reference['prompt_tokens']:
            raise ValueError('Maya tokenizer does not reproduce the exact Direct prompt IDs')
        reference_raw = tokenizer.decode(reference['generated_tokens'][:-1])
        if reference_raw != reference['raw_text'] or not semantic_result(reference_raw, reference, parser_class)['passed']:
            raise ValueError('Maya text pipeline disagrees on the qualified reference output')
        result['text_pipeline_source'] = {str(p): bench.sha(p) for p in
            (Path(plan['source_path']) / 'tools/strata_tokenizer.py', Path(plan['source_path']) / 'serve/frontend.py')}
        result['prompt_tokenizer_exact'] = True
        result['reference'] = reference
        result['derived_files_before'] = files
        command = [plan['exe'], '--glm-pack', plan['pack'], '--max-context', str(plan['max_context']),
                   '--prefill', prefill, '--serve']
        result['command'] = command
        started = time.monotonic()
        deadline = started + args.timeout
        result['process_start_unix_ns'] = time.time_ns()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', env=cleaned_environment(plan['environment']),
                                   creationflags=bench.PROCESS_FLAGS)
        result['process_id'] = process.pid
        def expire():
            if process.poll() is None:
                timed_out.set()
                try: process.kill()
                except OSError as error: result['errors'].append(f'deadline kill: {error!r}')
        watchdog = threading.Timer(args.timeout, expire)
        watchdog.start()
        def read_output():
            try:
                with (out / 'stdout.log').open('w', encoding='utf-8') as log:
                    for line in process.stdout:
                        log.write(line); log.flush()
                        lines.put((time.monotonic(), line.rstrip('\r\n')))
            except Exception as error:
                lines.put((time.monotonic(), 'ERR reader: ' + repr(error)))
            finally: lines.put((time.monotonic(), None))
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        sampler = bench.Sampler(out / 'telemetry.jsonl')
        sampler.process = bench.psutil.Process(process.pid)
        sampler.thread.start()
        def next_line():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or timed_out.is_set(): raise TimeoutError('Owned Maya deadline')
            stamp, text = lines.get(timeout=remaining)
            if time.monotonic() >= deadline or timed_out.is_set(): raise TimeoutError('Late Maya response')
            if text is None: raise RuntimeError('Maya exited before completing the protocol')
            if text.startswith('ERR '): raise RuntimeError(text)
            return stamp, text
        while True:
            _, line = next_line()
            if line.startswith('READY '): break
        result['ready_seconds'] = time.monotonic() - started
        actual_exe = Path(sampler.process.exe())
        if actual_exe.resolve() != Path(plan['exe']).resolve() or bench.sha(actual_exe) != plan['exe_sha256']:
            raise ValueError('Actual owned executable identity mismatch')
        result['actual_executable'] = {'path': str(actual_exe), 'sha256': bench.sha(actual_exe)}
        for repetition in range(plan['repetitions']):
            request = 'GEN ' + str(plan['max_new']) + ' temperature=0 ' + ','.join(map(str, reference['prompt_tokens'])) + '\n'
            row = dict(repetition=repetition, tokens=[], submitted_unix_ns=time.time_ns(), first_token_ms=None)
            t0 = time.monotonic()
            process.stdin.write(request); process.stdin.flush()
            while True:
                stamp, line = next_line()
                if line.startswith('T '):
                    fields = line.split()
                    if len(fields) != 2: raise ValueError('Malformed token wire row')
                    token = int(fields[1])
                    if token < 0 or token >= plan['vocab_size']: raise ValueError('Out-of-vocabulary output')
                    row['tokens'].append(token)
                    if row['first_token_ms'] is None: row['first_token_ms'] = (stamp - t0) * 1000
                    if len(row['tokens']) > plan['max_new']: raise ValueError('Generation exceeded bound')
                elif line.startswith('DONE '):
                    row.update(parse_done(line)); row['wall_ms'] = (stamp - t0) * 1000
                    break
            decoded = row['tokens'][:-1] if row['tokens'] and row['tokens'][-1] in plan['eos_ids'] else row['tokens']
            row['semantic'] = semantic_result(tokenizer.decode(decoded), reference, parser_class)
            row['quality'] = qualify(row, reference, plan['eos_ids'], row['semantic'])
            row['post_prefill_forward_tokens_per_second'] = ((row['generated'] - 1) * 1000 / row['decode_ms']
                if row['generated'] > 1 and row['decode_ms'] > 0 else None)
            row['prefill_tokens_per_second'] = row['prompt_tokens'] * 1000 / row['prompt_ms'] if row['prompt_ms'] > 0 else None
            result['rows'].append(row); save()
        process.stdin.write('QUIT\n'); process.stdin.flush()
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if time.monotonic() >= deadline or timed_out.is_set(): raise TimeoutError('Late Maya process completion')
        result['completed'] = True
    except Exception as error:
        if isinstance(error, (TimeoutError, queue.Empty, subprocess.TimeoutExpired)):
            timed_out.set()
        result['errors'].append(repr(error))
    finally:
        result['exit_code'], cleanup_errors = cleanup_owned(process, watchdog, reader, sampler)
        result['errors'].extend(cleanup_errors)
        if sampler is not None:
            result['sampling_errors'] = sampler.errors
            result['telemetry_samples'] = len(sampler.rows)
            result['performance_observation_complete'] = bool(sampler.rows) and not sampler.errors and not cleanup_errors
        result['timed_out'] = timed_out.is_set()
        try:
            _, after = verify_inputs(plan)
            result['derived_files_after'] = after
            result['identities_unchanged'] = after == result.get('derived_files_before')
        except Exception as error:
            result['identities_unchanged'] = False
            result['errors'].append('identity after: ' + repr(error))
        result['process_success'] = result['completed'] and result.get('exit_code') == 0 and not result['errors'] and result['identities_unchanged'] and not timed_out.is_set()
        result['quality_passed'] = result['process_success'] and len(result['rows']) == plan['repetitions'] and all(r['quality']['passed'] for r in result['rows'])
        save()
    print(json.dumps({k: result[k] for k in ('completed', 'process_success', 'quality_passed', 'errors')}))
    return 0 if result['quality_passed'] else 2 if result['process_success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
