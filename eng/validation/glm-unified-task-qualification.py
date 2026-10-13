#!/usr/bin/env python3
"""Prepare bound GLM inputs and grade finished public unified-session reports.

Never launches a process/model. Inputs come only from the Direct quality row's
prompt_tokens, cross-checked against the historical Maya input. Reference model
answers are never oracles. --task unknown deliberately exits1 in frozen Probev3;
this checker independently establishes semantics, integrity and cleanup.
"""
import argparse
import calendar
import copy
import datetime
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('glm_fixed_tasks', ROOT / 'fixed-task-semantic.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
CASES = ('extraction', 'code', 'tool_json')
BASE = pathlib.Path('D:/TensorSharp-validation-cache')
PREFIX = '[gMASK]<sop><|system|>Reasoning Effort: Low<|system|>You are a helpful assistant.<|user|>'
SUFFIX = '<|assistant|><think>'
OPTIONS = dict(ContextTokens=1024, PrefillChunkTokens=32, ExpertCachePrefillRows=32,
    MaximumRunningRequests=1, PrefixCheckpointCount=0, RetainedConversationCount=0,
    MaximumHostBytes=12 << 30, MaximumDeviceBytes=14 << 30, MaximumExpertCacheBytes=7000 << 20,
    EnableGlmSelectedExperts=True, EnableAsyncExpertPrefill=False, EnableExpertCacheCapacityGrowth=False)
ARGS = dict(context='1024', **{'max-new': '128', 'prefill': '32', 'expert-prefill-rows': '32',
    'selected': 'true', 'cpu-threads': '8', 'deadline-seconds': '720', 'task': 'unknown',
    'host-bytes': str(12 << 30), 'device-bytes': str(14 << 30), 'expert-bytes': str(7000 << 20),
    'repetitions': '1', 'cache-telemetry': 'true', 'cache-token-snapshots': 'false'})
IDENTITY_FLAGS = ('ModelStatsUnchanged', 'ManagedUnchanged', 'SourceToolsUnchanged',
    'NativeUnchanged', 'InputFileUnchanged', 'HistoricalModelIdentityUnchanged')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def identity(path):
    path = pathlib.Path(path).resolve()
    return dict(Path=str(path), Bytes=path.stat().st_size, Sha256=sha(path))


def load(path):
    path = pathlib.Path(path)
    if path.stat().st_size > 4_194_304:
        raise ValueError('Report/plan exceeds4MiB bound')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=fixed.unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def canonical(path):
    value = re.sub('/+', '/', str(path).replace('\\', '/'))
    return value.casefold() if re.match(r'^[A-Za-z]:/', value) else value


def basename(path):
    return canonical(path).rsplit('/', 1)[-1]


def same_identity(actual, expected):
    return isinstance(actual, dict) and canonical(actual.get('Path', '')) == canonical(expected['Path']) \
        and type(actual.get('Bytes')) is int and actual['Bytes'] == expected['Bytes'] \
        and isinstance(actual.get('Sha256'), str) and actual['Sha256'].lower() == expected['Sha256'].lower()


def utc_ns(text):
    match = re.fullmatch(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?Z', text)
    if not match: raise ValueError('Expected exact UTC timestamp')
    seconds = calendar.timegm(datetime.datetime.fromisoformat(match[1]).timetuple())
    return seconds * 1_000_000_000 + int((match[2] or '').ljust(9, '0'))


def deployment(receipt, app):
    variants = [x for x in receipt if x['Variant'] == 'native16' and canonical(x['Directory']) == canonical(app)]
    if len(variants) != 1: raise ValueError('Require pinned frozen native16 app receipt')
    files = {x['Name']: x for x in variants[0]['Files']}
    if len(files) != len(variants[0]['Files']): raise ValueError('Duplicate deployment file')
    native = dict(Path=str(pathlib.Path(app) / 'GgmlOps.dll'), Bytes=files['GgmlOps.dll']['Bytes'], Sha256=files['GgmlOps.dll']['Sha256'])
    if native['Sha256'] != variants[0]['NativeSha256']: raise ValueError('Native receipt pin differs')
    managed_expected = [dict(Path=str(pathlib.Path(app) / name), Bytes=row['Bytes'], Sha256=row['Sha256'])
        for name, row in files.items() if name.endswith('.dll') and (name.startswith('TensorSharp') or name == 'GlmUnifiedSessionProbe.dll')]
    wrapper_expected = {canonical(str(pathlib.Path(app) / name)): row['Sha256'] for name, row in files.items()
        if name.endswith(('.dll', '.deps.json', '.runtimeconfig.json'))}
    return files, native, managed_expected, wrapper_expected


def command(plan):
    values = ['dotnet', str(pathlib.Path(plan['AppDirectory']) / 'GlmUnifiedSessionProbe.dll'),
        '--output', plan['ReportOutput'], '--model', plan['Model'], '--tokens-file', plan['InputFile']['Path'],
        '--model-identity', plan['ModelIdentity']['Path'], '--native-sha256', plan['Native']['Sha256']]
    for key, value in ARGS.items(): values += ['--' + key, value]
    return values


def tokens(value, vocab):
    if not isinstance(value, list) or not value or len(value) > 1_048_576 \
        or any(type(x) is not int or not 0 <= x < vocab for x in value):
        raise ValueError('Invalid original token array')
    return value


def bind_direct(source, case, vocab=154880):
    if case not in CASES or source.get('synthetic') is not False \
        or source.get('architecture') not in ('glm5-next', 'glm5next') \
        or source.get('dimensions', {}).get('VocabSize') != vocab \
        or source.get('source_dimensions', {}).get('glm5-next.vocab_size') != vocab:
        raise ValueError('Require original nonsynthetic GLM Direct inputs')
    rows = [row for row in source['quality'] if row.get('name') == case]
    if len(rows) != 1:
        raise ValueError('Require exactly one original input row for task')
    row = rows[0]
    rendered = PREFIX + fixed.PROMPTS[case] + SUFFIX
    if row.get('prompt') != fixed.PROMPTS[case] or row.get('rendered_prompt') != rendered \
        or row.get('rendered_prompt_utf8_sha256') != hashlib.sha256(rendered.encode()).hexdigest():
        raise ValueError('Independent task/rendered input binding mismatch')
    return dict(Case=case, Prompt=fixed.PROMPTS[case], RenderedPrompt=rendered,
        PromptTokens=list(tokens(row['prompt_tokens'], vocab)), VocabSize=vocab,
        SourceSelector='quality[name=' + case + '].prompt_tokens', SourceModel=source['model_path'],
        SourceModelFiles=[dict(Path=f['path'], Bytes=f['bytes'], MtimeNs=utc_ns(f['last_write_utc'])) for f in source['model_files']])


def bind_maya(maya, binding):
    row = maya['reference']
    if maya.get('prompt_tokenizer_exact') is not True or row.get('name') != binding['Case'] \
        or row.get('prompt') != binding['Prompt'] or row.get('rendered_prompt') != binding['RenderedPrompt'] \
        or row.get('prompt_tokens') != binding['PromptTokens']:
        raise ValueError('Maya original conditioning differs from Direct')


def managed(rows):
    result = {}
    for row in rows:
        name = basename(row['Path']).casefold()
        if name in result or not re.fullmatch('[0-9a-fA-F]{64}', row['Sha256']):
            raise ValueError('Duplicate/invalid managed identity')
        result[name] = row['Sha256'].lower()
    return result


def metadata_rows(rows, path_key, bytes_key, time_key, ticks=False):
    """Exact integer metadata; .NET ticks preserve100ns, never float seconds."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Require nonempty model metadata rows')
    result = {}
    for row in rows:
        path, size, time = row[path_key], row[bytes_key], row[time_key]
        if not isinstance(path, str) or not path or type(size) is not int or size <= 0 \
            or type(time) is not int or time < (621355968000000000 if ticks else 0):
            raise ValueError('Invalid exact model metadata schema')
        key = canonical(path)
        if key in result:
            raise ValueError('Duplicate canonical model metadata path')
        result[key] = (size, (time - 621355968000000000) * 100 if ticks else time)
    return result


def model_metadata(binding, manifest, before, after):
    expected = metadata_rows(binding['SourceModelFiles'], 'Path', 'Bytes', 'MtimeNs')
    historical = metadata_rows(manifest['files'], 'path', 'bytes', 'mtime_ns')
    actual = []
    for phase in (before, after):
        if not isinstance(phase, dict) or any(not isinstance(key, str) or not isinstance(row, dict)
            or canonical(key) != canonical(row['Path']) for key, row in phase.items()):
            raise ValueError('Runtime model dictionary key/path differs')
        actual.append(metadata_rows(list(phase.values()), 'Path', 'Bytes', 'LastWriteTimeUtcTicks', ticks=True))
    # A full checkpoint manifest may also contain an mmproj, while this text-only
    # session loads only the source's required GGUF shards. Every required shard
    # must match exactly; extra manifest rows are not claimed as live-validated.
    required_historical = {key: historical[key] for key in expected if key in historical}
    return dict(Matches=expected == required_historical == actual[0] == actual[1],
        RequiredModelPaths=sorted(expected), AdditionalManifestPaths=sorted(set(historical) - set(expected)),
        TimestampComparison='Exact integer nanoseconds; .NET100ns ticks converted without float/tolerance.',
        Scope='All required model shards before/after; additional manifest attachments are not live-validated by this text-only probe.')


def verify_revalidation_origin(plan):
    if 'OriginPlan' not in plan:
        return
    if identity(plan['OriginPlan']['Path']) != plan['OriginPlan'] \
        or identity(plan['OriginCheckerSnapshot']['Path']) != plan['OriginCheckerSnapshot']:
        raise ValueError('Original plan/checker snapshot changed')
    origin = load(plan['OriginPlan']['Path'])
    snapshot = plan['OriginCheckerSnapshot']
    if snapshot['Bytes'] != origin['Checker']['Bytes'] or snapshot['Sha256'] != origin['Checker']['Sha256']:
        raise ValueError('Snapshot is not the original executing checker identity')
    restored = copy.deepcopy(plan)
    for key in ('OriginPlan', 'OriginCheckerSnapshot', 'RevalidationScope'):
        restored.pop(key)
    restored['Checker'] = origin['Checker']
    restored['ExecutionStatus'] = origin['ExecutionStatus']
    if restored != origin or plan['ExecutionStatus'] != 'revalidation-only-existing-execution':
        raise ValueError('Revalidation changed original command/input/runtime expectations')


def verify_plan_files(plan):
    verify_revalidation_origin(plan)
    for name in ('Source', 'MayaSource', 'DeploymentReceipt', 'ModelIdentity', 'Oracle', 'Checker', 'InputFile'):
        row = plan[name]
        if identity(row['Path']) != row:
            raise ValueError('Pinned input/tool/provenance changed: ' + name)
    binding = bind_direct(load(plan['Source']['Path']), plan['Case'], plan['Binding']['VocabSize'])
    if binding != plan['Binding']:
        raise ValueError('Plan original binding altered')
    bind_maya(load(plan['MayaSource']['Path']), binding)
    if canonical(plan['Model']) != canonical(binding['SourceModel']): raise ValueError('Actual source model binding differs')
    if not same_identity(plan['Oracle'], identity(fixed.__file__)) or not same_identity(plan['Checker'], identity(__file__)):
        raise ValueError('Plan names a different executing oracle/checker')
    _, native, expected, wrapper = deployment(load(plan['DeploymentReceipt']['Path']), plan['AppDirectory'])
    if plan['Native'] != native or plan['ManagedIdentities'] != expected or plan['WrapperFiles'] != wrapper \
        or plan['Managed'] != managed(expected) or plan['ModelManifest'] != load(plan['ModelIdentity']['Path']) \
        or plan['Command'] != command(plan): raise ValueError('Plan expectations differ from pinned provenance')
    text = pathlib.Path(plan['InputFile']['Path']).read_text(encoding='utf-8')
    if text != ','.join(map(str, binding['PromptTokens'])) + '\n':
        raise ValueError('Original raw input file differs from pinned source')


def qualify(plan, report, wrapper):
    """Pure grading; CLI separately rechecks small pinned source/input files."""
    errors = []
    def check(value, label):
        if value is not True:
            errors.append(label)
    try:
        case = plan['Case']; binding = plan['Binding']; vocab = binding['VocabSize']
        check(plan['SchemaVersion'] == 1 and report['SchemaVersion'] == 2, 'Pinned plan/Probev3 report schema')
        check(case in CASES and binding['Case'] == case and binding['Prompt'] == fixed.PROMPTS.get(case), 'Independent task binding')
        check(binding['RenderedPrompt'] == PREFIX + binding['Prompt'] + SUFFIX, 'Exact rendered conditioning')
        check(canonical(plan['Model']) == canonical(binding['SourceModel']), 'Actual original model binding')
        prompt = tokens(binding['PromptTokens'], vocab); output = tokens(report['OutputTokens'], vocab)
        for name in ('PromptTokens', 'PublishedTokens', 'OutputTokensAfterCleanup'):
            tokens(report[name], vocab)
        eos = tokens(report['CheckpointEosIds'], vocab)
        check(eos == plan['EosIds'], 'Checkpoint EOS identity')
        check(output[-1] in eos and not any(x in eos for x in output[:-1]), 'Natural unique terminal EOS')
        completion = report['Completion']
        check(completion['Status'] == 'FinishedStopped' and completion['FinishReason'] == 'eos', 'Finished natural EOS completion')
        check(completion['PromptTokenCount'] == len(prompt) and completion['OutputTokenCount'] == len(output)
            and completion['PrefixCacheReusedTokens'] == 0 and len(output) <= 128, 'Full output/prompt counts without prefix reuse')
        check(report['PromptTokens'] == prompt and report['DecodedPrompt'] == binding['RenderedPrompt'], 'Actual raw-ID and production decode binding')
        check(report['PublishedTokens'] == output[:-1] and report['ComputedTokens'] == len(prompt) + len(output), 'Complete P+O AR publication/commit')
        check(report['OutputTokensAfterCleanup'] == output and report['ComputedTokensAfterCleanup'] == report['ComputedTokens']
            and report['SequenceStatusAfterCleanup'] == 'FinishedStopped' and report['FinishReasonAfterCleanup'] == 'eos'
            and report['SequenceError'] is None and report['RawTextAfterCleanup'] == report['RawText'], 'No cleanup token/state mutation or sequence error')
        check(report['StructuralPassed'] is True and report['AllRoundsStructuralPassed'] is True
            and report['NaturalEos'] is True and report['NaturalEosAfterCleanup'] is True, 'Probe structural receipts')
        check(report['StatusBeforeDispose'] == 'FinishedStopped', 'Finished before disposal')
        check(report['Status'] == 'completed-quality-unknown' and report['Passed'] is False
            and report['SemanticPassed'] is None and report['AllRoundsSemanticPassed'] is None, 'Unknown remains unassessed in original probe')
        check(report['CompletedRepetitions'] == 1 and report['RequestedRepetitions'] == 1
            and report['EngineCompletionObserved'] is True, 'Exactly one completed request')
        check(all(type(report['SessionOptions'][k]) is type(v) and report['SessionOptions'][k] == v for k, v in OPTIONS.items()), 'Exact public session options')
        check(all(report['Arguments'][k] == v for k, v in ARGS.items()), 'Exact request controls')
        for key, expected in [('model', plan['Model']), ('tokens-file', plan['InputFile']['Path']), ('model-identity', plan['ModelIdentity']['Path'])]:
            check(canonical(report['Arguments'][key]) == canonical(expected), 'Request argument path: ' + key)
        check(report['EffectiveContextTokens'] == 1024 and report['EffectivePrefillChunkTokens'] == 32
            and report['EffectiveCpuThreads'] == 8, 'Effective shapes/threads')
        scheduler = report['Scheduler']
        check(all(type(scheduler[k]) is type(v) and scheduler[k] == v for k, v in dict(PrefillChunkTokenLimit=32, MaxPrefillChunkSize=32,
            SoloPrefillChunkSize=32, MaxNumBatchedTokens=32, MaxNumRunningSequences=1,
            EnablePrefixCaching=False, StopRepetition=False, SharedAdmission=True, Enabled=False).items()), 'Scheduler uses effective cap/shared admission/no spec')
        rounds = report['Rounds']; check(len(rounds) == 1, 'Single round array')
        if len(rounds) == 1:
            check(rounds[0]['Round'] == 1 and rounds[0]['RequestId'] == report['RequestId'], 'Exact round/request identity')
            check(all(rounds[0][k] == report[k] for k in ('Completion', 'OutputTokens', 'PublishedTokens', 'ComputedTokens',
                'RawText', 'Content', 'ToolCalls', 'ThinkingClosed', 'SemanticPassed', 'Passed', 'Status')), 'Round/top-level consistency')
        check(report['Errors'] == [] and all(report[k] is True for k in IDENTITY_FLAGS), 'No failure or changed identity')
        check(report['ModelStatsBefore'] == report['ModelStatsAfter'] and report['SourceToolsBefore'] == report['SourceToolsAfter'], 'Recomputed model metadata/tool source stability')
        check(report['ModelIdentityLinked'] is True and report['HistoricalModelIdentity']['Sha256'] == plan['ModelIdentity']['Sha256'], 'Original model full-hash linkage')
        check(report['HistoricalModelManifest'] == plan['ModelManifest']
            and same_identity(report['HistoricalModelIdentityAfter'], plan['ModelIdentity']), 'Original manifest before/after linkage')
        model_binding = model_metadata(binding, plan['ModelManifest'], report['ModelStatsBefore'], report['ModelStatsAfter'])
        check(model_binding['Matches'], 'All required original model shards size/mtime binding')
        check(same_identity(report['InputFile'], plan['InputFile']) and same_identity(report['InputFileAfter'], plan['InputFile']), 'Exact pinned original input identity')
        check(managed(report['ManagedBefore']) == plan['Managed'] and managed(report['ManagedAfter']) == plan['Managed'], 'Pinned actual managed/tool identities')
        for rows in (report['ManagedBefore'], report['ManagedAfter']):
            check(len(rows) == len(plan['ManagedIdentities']) and all(any(same_identity(a, e) for a in rows)
                for e in plan['ManagedIdentities']), 'Exact deployment managed paths/bytes')
        native = plan['Native']
        check(same_identity(report['IntendedNative'], native) and len(report['LoadedNativeAfterLoad']) == 1
            and same_identity(report['LoadedNativeAfterLoad'][0], native) and len(report['LoadedNativeAfter']) == 1
            and same_identity(report['LoadedNativeAfter'][0], native) and same_identity(report['IntendedNativeAfter'], native), 'Pinned actual loaded native')
        check(report['Arguments']['native-sha256'] == native['Sha256'], 'Native requested pin')
        check(report['EnvironmentEffective']['CUDA_VISIBLE_DEVICES'] == '0'
            and report['EnvironmentEffective']['CUDA_MODULE_LOADING'] == 'EAGER', 'CUDA environment controls')
        check(report['EngineDisposed'] is True and report['SessionDisposed'] is True
            and report['LedgerZeroAfterCleanup'] is True, 'Completed engine/session cleanup')
        pools = report['BudgetAfterSessionDisposed']['Current']
        check({p['Pool'] for p in pools} == {'unified/ram', 'unified/cuda0'} and len(pools) == 2, 'Complete single-device ledger pools')
        check(all(type(p['Reserved']) is int and type(p['Committed']) is int and p['Reserved'] == 0
            and p['Committed'] == 0 and p['Available'] == p['Capacity'] and p['Capacity'] > 0 for p in pools), 'Recomputed zero final ledger')
        check(report['FinalLogitsFinite'] is True or report['FinalLogitsFinite'] is None, 'Finite logits or explicitly unavailable')
        check(wrapper['complete'] is True and wrapper['timed_out'] is False and wrapper['exit_code'] == 1,
            'Frozen Probev3 completed unknown exit1; exit is not semantic verdict')
        check(type(wrapper['pid']) is int and wrapper['pid'] > 0 and wrapper['pid'] == report['ProcessId'], 'Same owned execution process')
        telemetry = wrapper['telemetry']
        check(wrapper['cleanup_errors'] == [] and telemetry['errors'] == [] and telemetry['passed'] is True
            and telemetry['native_verified'] is True and telemetry['managed_files_unchanged'] is True
            and telemetry['loaded_native']['sha256'] == native['Sha256']
            and canonical(telemetry['loaded_native']['path']) == canonical(native['Path']), 'Owned wrapper native/identity/cleanup receipts')
        check(telemetry['loaded_native']['pid'] == wrapper['pid'] and telemetry['loaded_native']['capture_phase'] == 'owned process alive'
            and telemetry['expected_native_sha256'] == native['Sha256'] and not wrapper.get('error'), 'Actual live native observation belongs to this execution')
        check(len(wrapper['command']) == len(plan['Command']) and all(canonical(a) == canonical(b)
            for a, b in zip(wrapper['command'], plan['Command'])), 'Exact owned wrapper command')
        check(canonical(report['Arguments']['output']) == canonical(plan['ReportOutput']), 'Report output belongs to planned execution')
        for field in ('managed_files_before', 'managed_files_after'):
            check({canonical(p): value for p, value in telemetry[field].items()} == plan['WrapperFiles'], 'Wrapper deployment hashes: ' + field)
        engine = report['EngineCountersBeforeDispose']
        check(engine['TotalSubmitted'] == 1 and engine['TotalCompleted'] == 1
            and all(engine[k] == 0 for k in ('RunningCount', 'WaitingCount', 'PendingRequestCount')), 'One completed quiescent engine request')
        # Serialized null means no tool calls in this production parser. Missing
        # keys fail above; do not turn an actual call or broken field into [].
        calls = report['ToolCalls']; calls = [] if calls is None else calls
        closed = fixed.thinking_closed(binding['RenderedPrompt'], report['RawText'])
        check(report['ThinkingClosed'] is True and closed is True, 'Raw thinking closes')
        semantic = fixed.parsed_answer(case, binding['Prompt'], report['Content'], closed, calls)
        return dict(SchemaVersion=1, Passed=not errors and semantic['passed'], IntegrityPassed=not errors,
            IndependentSemantic=semantic, Errors=errors, OriginalProbeStatus=report['Status'],
            OriginalProbePassed=report['Passed'], OriginalWrapperExit=wrapper['exit_code'],
            Tokens=output, RawText=report['RawText'], Content=report['Content'], ToolCalls=report['ToolCalls'],
            Timing=report['Timing'], BudgetAfterCompletion=report['BudgetAfterCompletion'],
            BudgetAfterCleanup=report['BudgetAfterSessionDisposed'], FinalLogitsFinite=report['FinalLogitsFinite'],
            ModelMetadataBinding=model_binding,
            Limits=['Only independent fixed task, no reference-answer equality or output repair.',
                'Code oracle compares safe AST; no generated code execution. Literal tool JSON is not tool execution/agent acceptance.',
                'Logits null means unavailable, not numerical acceptance. Parsed output is supplied by pinned production parser.',
                'Payload ceilings and sampled memory are not complete physical hard caps; no fair Maya speed ratio.',
                'Original unknown report remains unchanged/unassessed; this independent verdict is separate.'])
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        return dict(SchemaVersion=1, Passed=False, IntegrityPassed=False, IndependentSemantic=None,
            Errors=errors + [type(error).__name__ + ': ' + str(error)])


def prepare(args):
    receipt = load(args.deployment_receipt)
    app = pathlib.Path(args.app).resolve()
    files, native, managed_expected, wrapper_expected = deployment(receipt, str(app))
    for name in ('GgmlOps.dll', 'GlmUnifiedSessionProbe.dll'):
        if (pathlib.Path(args.app) / name).stat().st_size != files[name]['Bytes']:
            raise ValueError('Frozen app file size differs; no binary reread in preparation')
    out = pathlib.Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    plans = []
    for case in CASES:
        source = pathlib.Path(args.source_directory) / ('glm-e-budget-c2f-v1.json' if case == 'extraction' else 'glm-extended-quality-autoplan-v1.json')
        maya = pathlib.Path(args.source_directory) / ('maya-e-extraction-batch32-v4' if case == 'extraction' else 'maya-e-' + case + '-batch32-v6') / 'report.json'
        binding = bind_direct(load(source), case); bind_maya(load(maya), binding)
        token_file = out / (case + '.ids'); token_file.write_text(','.join(map(str, binding['PromptTokens'])) + '\n', encoding='utf-8')
        target = out / (case + '-run')
        plan = dict(SchemaVersion=1, Case=case, Binding=binding, Source=identity(source), MayaSource=identity(maya),
            DeploymentReceipt=identity(args.deployment_receipt), ModelIdentity=identity(args.model_identity), ModelManifest=load(args.model_identity),
            Oracle=identity(fixed.__file__), Checker=identity(__file__), InputFile=identity(token_file),
            Model=args.model, EosIds=[154820, 154827, 154829], Native=native, Managed=managed(managed_expected),
            ManagedIdentities=managed_expected, WrapperFiles=wrapper_expected, AppDirectory=str(app),
            WrapperOutput=str(out / (case + '-wrapper')), ReportOutput=str(target),
            Environment={'CUDA_VISIBLE_DEVICES': '0', 'CUDA_MODULE_LOADING': 'EAGER', 'TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS': '1'},
            DeadlineSeconds=900, RequestedOptions=OPTIONS,
            Scope='Preparation only; source input IDs, not generated answers. Same historical Maya conditioning, not equivalent physical resource or current Maya performance acceptance.',
            ExpectedProbeUnknownExit=1, ExecutionStatus='not-run')
        plan['Command'] = command(plan)
        verify_plan_files(plan)
        (out / (case + '-plan.json')).write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); plans.append(case)
    (out / 'preparation.json').write_text(json.dumps(dict(Prepared=plans, ExecutionStatus='not-run',
        NativeIdentityScope='Recorded frozen receipt + current file size, no DLL/model content reread. Runtime probe/wrapper must verify actual identities.'), indent=2) + '\n', encoding='utf-8')


def revalidate_plan(args):
    origin = load(args.plan)
    snapshot = identity(args.origin_checker)
    if snapshot['Bytes'] != origin['Checker']['Bytes'] or snapshot['Sha256'] != origin['Checker']['Sha256']:
        raise ValueError('Require preserved exact original checker before revalidation')
    plan = copy.deepcopy(origin)
    plan.update(Checker=identity(__file__), OriginPlan=identity(args.plan), OriginCheckerSnapshot=snapshot,
        ExecutionStatus='revalidation-only-existing-execution',
        RevalidationScope='No new process execution; only regrades original capture. Original Command/input/runtime/output targets remain exact.')
    verify_plan_files(plan)
    target = pathlib.Path(args.output)
    with target.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--output', required=True)
    prep.add_argument('--source-directory', default=str(BASE / 'default-readiness-20261010'))
    prep.add_argument('--app', default=str(BASE / 'master-merge-20261013/glm-selected-comparison-v3-apps/native16'))
    prep.add_argument('--deployment-receipt', default=str(BASE / 'master-merge-20261013/glm-selected-comparison-v3-apps/deployment-receipt.json'))
    prep.add_argument('--model', default='E:/models/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf')
    prep.add_argument('--model-identity', default=str(BASE / 'default-readiness-20261010/glm-e-identity-v1.json'))
    reval = sub.add_parser('revalidate-plan'); reval.add_argument('--plan', required=True)
    reval.add_argument('--origin-checker', required=True); reval.add_argument('--output', required=True)
    check = sub.add_parser('check'); check.add_argument('--plan', required=True); check.add_argument('--report', required=True)
    check.add_argument('--wrapper', required=True); check.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args); return 0
    if args.mode == 'revalidate-plan':
        revalidate_plan(args); return 0
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=False)
    try:
        plan = load(args.plan); verify_plan_files(plan)
        result = qualify(plan, load(args.report), load(args.wrapper))
        result['Sources'] = dict(Plan=identity(args.plan), Report=identity(args.report), Wrapper=identity(args.wrapper),
            Checker=identity(__file__), Oracle=identity(fixed.__file__))
        if 'OriginPlan' in plan:
            result['Revalidation'] = dict(Scope=plan['RevalidationScope'], OriginPlan=plan['OriginPlan'],
                OriginCheckerSnapshot=plan['OriginCheckerSnapshot'])
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        result = dict(Passed=False, IntegrityPassed=False, IndependentSemantic=None, Errors=[str(error)])
    (out / 'qualification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(Output=str(out), Passed=result['Passed'])))
    return 0 if result['Passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
