import copy
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / 'glm-unified-task-qualification.py'
SPEC = importlib.util.spec_from_file_location('glm_unified_qualification', TOOL)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def fixture(case='extraction'):
    answer = {'extraction': 'cedar, maple', 'code': 'def square(x):\n    return x*x',
        'tool_json': '{"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}'}[case]
    native = dict(Path='/app/GgmlOps.dll', Bytes=32, Sha256='a' * 64)
    input_file = dict(Path='/input.ids', Bytes=6, Sha256='d' * 64)
    manifest = dict(Path='/model.json', Bytes=2, Sha256='e' * 64)
    model_manifest = {'files': [dict(path='model.gguf', bytes=10, mtime_ns=100, sha256='f' * 64)]}
    names = {'glmunifiedsessionprobe.dll': 'b' * 64, 'tensorsharp.models.dll': 'c' * 64}
    plan = dict(SchemaVersion=1, Case=case, Binding=dict(Case=case, Prompt=MOD.fixed.PROMPTS[case],
        RenderedPrompt=MOD.PREFIX + MOD.fixed.PROMPTS[case] + MOD.SUFFIX, PromptTokens=[1, 2, 3], VocabSize=100,
        SourceModel='model.gguf', SourceModelFiles=[dict(Path='model.gguf', Bytes=10, MtimeNs=100)]),
        Native=native, Managed=names, EosIds=[99], InputFile=input_file, Model='model.gguf', ModelIdentity=manifest,
        ModelManifest=model_manifest, AppDirectory='/app', ReportOutput='/output-run')
    completion = dict(Status='FinishedStopped', FinishReason='eos', PromptTokenCount=3, OutputTokenCount=3, PrefixCacheReusedTokens=0)
    identities = [dict(Path='/app/GlmUnifiedSessionProbe.dll', Bytes=10, Sha256='b' * 64),
        dict(Path='/app/TensorSharp.Models.dll', Bytes=10, Sha256='c' * 64)]
    ledger = dict(Current=[dict(Pool=p, Capacity=1000, Available=1000, Reserved=0, Committed=0)
        for p in ['unified/ram', 'unified/cuda0']])
    report = dict(SchemaVersion=2, ProcessId=41, PromptTokens=[1, 2, 3], DecodedPrompt=plan['Binding']['RenderedPrompt'],
        OutputTokens=[5, 6, 99], PublishedTokens=[5, 6], ComputedTokens=6, CheckpointEosIds=[99],
        OutputTokensAfterCleanup=[5, 6, 99], ComputedTokensAfterCleanup=6, SequenceStatusAfterCleanup='FinishedStopped',
        FinishReasonAfterCleanup='eos', SequenceError=None, Completion=completion, StructuralPassed=True,
        StatusBeforeDispose='FinishedStopped', RequestId='unit-request',
        AllRoundsStructuralPassed=True, NaturalEos=True, NaturalEosAfterCleanup=True, Status='completed-quality-unknown',
        Passed=False, SemanticPassed=None, AllRoundsSemanticPassed=None, CompletedRepetitions=1, RequestedRepetitions=1,
        EngineCompletionObserved=True, SessionOptions=copy.deepcopy(MOD.OPTIONS),
        Arguments=MOD.ARGS | {'model': 'model.gguf', 'tokens-file': '/input.ids', 'model-identity': '/model.json', 'native-sha256': 'a' * 64, 'output': '/output-run'},
        EffectiveContextTokens=1024, EffectivePrefillChunkTokens=32, EffectiveCpuThreads=8,
        Scheduler=dict(PrefillChunkTokenLimit=32, MaxPrefillChunkSize=32, SoloPrefillChunkSize=32,
            MaxNumBatchedTokens=32, MaxNumRunningSequences=1, EnablePrefixCaching=False,
            StopRepetition=False, SharedAdmission=True, Enabled=False), Errors=[], ModelIdentityLinked=True,
        HistoricalModelIdentity=copy.deepcopy(manifest), HistoricalModelIdentityAfter=copy.deepcopy(manifest), HistoricalModelManifest=model_manifest,
        ModelStatsBefore={'model.gguf': dict(Path='model.gguf', Bytes=10, LastWriteTimeUtcTicks=621355968000000001)},
        ModelStatsAfter={'model.gguf': dict(Path='model.gguf', Bytes=10, LastWriteTimeUtcTicks=621355968000000001)},
        SourceToolsBefore=[], SourceToolsAfter=[],
        InputFile=copy.deepcopy(input_file), InputFileAfter=copy.deepcopy(input_file), ManagedBefore=copy.deepcopy(identities),
        ManagedAfter=copy.deepcopy(identities), IntendedNative=copy.deepcopy(native), IntendedNativeAfter=copy.deepcopy(native),
        LoadedNativeAfterLoad=[copy.deepcopy(native)], LoadedNativeAfter=[copy.deepcopy(native)],
        EnvironmentEffective={'CUDA_VISIBLE_DEVICES': '0', 'CUDA_MODULE_LOADING': 'EAGER'},
        EngineDisposed=True, SessionDisposed=True, LedgerZeroAfterCleanup=True, BudgetAfterSessionDisposed=ledger,
        BudgetAfterCompletion=copy.deepcopy(ledger), FinalLogitsFinite=None, RawText='</think>' + answer + '<|user|>',
        Content=answer, ToolCalls=None, ThinkingClosed=True, Timing={})
    report['EngineCountersBeforeDispose'] = dict(TotalSubmitted=1, TotalCompleted=1, RunningCount=0, WaitingCount=0, PendingRequestCount=0)
    plan['ManagedIdentities'] = copy.deepcopy(identities)
    plan['WrapperFiles'] = {row['Path']: row['Sha256'] for row in identities + [native]}
    plan['Command'] = MOD.command(plan)
    report['RawTextAfterCleanup'] = report['RawText']
    for key in MOD.IDENTITY_FLAGS: report[key] = True
    round_keys = ('Completion', 'OutputTokens', 'PublishedTokens', 'ComputedTokens', 'RawText', 'Content',
        'ToolCalls', 'ThinkingClosed', 'SemanticPassed', 'Passed', 'Status')
    report['Rounds'] = [{key: copy.deepcopy(report[key]) for key in round_keys}]
    report['Rounds'][0].update(Round=1, RequestId='unit-request')
    wrapper = dict(complete=True, timed_out=False, exit_code=1, pid=41, command=copy.deepcopy(plan['Command']), cleanup_errors=[], telemetry=dict(errors=[],
        passed=True, native_verified=True, managed_files_unchanged=True,
        expected_native_sha256='a' * 64, managed_files_before=copy.deepcopy(plan['WrapperFiles']),
        managed_files_after=copy.deepcopy(plan['WrapperFiles']),
        loaded_native={'path': '/app/GgmlOps.dll', 'sha256': 'a' * 64, 'pid': 41, 'capture_phase': 'owned process alive'}))
    return plan, report, wrapper


def relocated_fixture():
    plan, report, wrapper = fixture()
    names = ['glm-00001-of-00003.gguf', 'glm-00002-of-00003.gguf',
             'glm-00003-of-00003.gguf', 'mmproj.gguf']
    source, destination = 'E:/models/glm', 'C:/models/glm'
    original_rows = [dict(path=source + '/' + name, bytes=(index + 1) * 10,
        mtime_ns=(index + 1) * 100, sha256=str(index + 1) * 64) for index, name in enumerate(names)]
    runtime_rows = [dict(row, path=destination + '/' + name, mtime_ns=row['mtime_ns'] + 1000)
                    for row, name in zip(original_rows, names)]
    original_manifest, runtime_manifest = dict(files=original_rows), dict(files=runtime_rows)
    binding = plan['Binding']
    binding['SourceModel'] = original_rows[0]['path']
    binding['SourceModelFiles'] = [dict(Path=row['path'], Bytes=row['bytes'], MtimeNs=row['mtime_ns'])
                                   for row in original_rows[:3]]
    snapshot = lambda row: dict(size=row['bytes'], mtime_ns=row['mtime_ns'], ctime_ns=100,
                               device=1, inode=2, mode=0o100666)
    receipt = dict(schema='tensorsharp.verified-relocation.v1', status='copy_complete',
        config=dict(kind='directory', source=source, destination=destination,
                    source_root='E:/models', destination_root='C:/models',
                    include_files=names.copy(), retire_source=False, junction=False),
        logical_source_bytes=100,
        files={name: dict(source_snapshot=snapshot(old), destination_snapshot=snapshot(new),
                         sha256=old['sha256'], state='verified')
               for name, old, new in zip(names, original_rows, runtime_rows)})
    plan['ModelManifest'] = runtime_manifest
    plan['ModelIdentity'] = dict(Path='/runtime-model.json', Bytes=100, Sha256='9' * 64)
    plan['Relocation'] = dict(Receipt=dict(Path='/relocation.json', Bytes=100, Sha256='8' * 64),
        ReceiptData=receipt, SourceManifest=original_manifest,
        SourceManifestIdentity=dict(Path='/original-model.json', Bytes=100, Sha256='7' * 64),
        RuntimeBinding=MOD.relocation_runtime_binding(binding, original_manifest, runtime_manifest, receipt))
    plan['Model'] = runtime_rows[0]['path']
    report['Arguments'].update(model=plan['Model'], **{'model-identity': plan['ModelIdentity']['Path']})
    report['HistoricalModelIdentity'] = copy.deepcopy(plan['ModelIdentity'])
    report['HistoricalModelIdentityAfter'] = copy.deepcopy(plan['ModelIdentity'])
    report['HistoricalModelManifest'] = runtime_manifest
    stats = {row['path']: dict(Path=row['path'], Bytes=row['bytes'],
        LastWriteTimeUtcTicks=621355968000000000 + row['mtime_ns'] // 100) for row in runtime_rows[:3]}
    report['ModelStatsBefore'] = stats
    report['ModelStatsAfter'] = copy.deepcopy(stats)
    plan['Command'] = MOD.command(plan)
    wrapper['command'] = copy.deepcopy(plan['Command'])
    return plan, report, wrapper


class GlmUnifiedTaskQualificationTests(unittest.TestCase):
    def test_prepare_relocated_plans_pins_small_provenance_without_opening_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            p, _, _ = relocated_fixture()
            relocation = p['Relocation']
            source_dir, app = root / 'source', root / 'app'
            source_dir.mkdir(); app.mkdir()
            for name in ('GgmlOps.dll', 'GlmUnifiedSessionProbe.dll'):
                (app / name).write_bytes(b'tiny fake fixture, never executed')
            native_sha = MOD.sha(app / 'GgmlOps.dll')
            deployment = [dict(Variant='native16', Directory=str(app), NativeSha256=native_sha,
                Files=[dict(Name=item.name, Bytes=item.stat().st_size, Sha256=MOD.sha(item)) for item in app.iterdir()])]
            paths = {}
            for name, data in [('deployment', deployment), ('receipt', relocation['ReceiptData']),
                ('original', relocation['SourceManifest']), ('runtime', p['ModelManifest'])]:
                paths[name] = root / (name + '.json')
                paths[name].write_text(json.dumps(data), encoding='utf-8')
            quality = []
            for case in MOD.CASES:
                rendered = MOD.PREFIX + MOD.fixed.PROMPTS[case] + MOD.SUFFIX
                quality.append(dict(name=case, prompt=MOD.fixed.PROMPTS[case], rendered_prompt=rendered,
                    rendered_prompt_utf8_sha256=hashlib.sha256(rendered.encode()).hexdigest(), prompt_tokens=[1, 2, 3]))
                maya_dir = source_dir / ('maya-e-extraction-batch32-v4' if case == 'extraction' else 'maya-e-' + case + '-batch32-v6')
                maya_dir.mkdir()
                (maya_dir / 'report.json').write_text(json.dumps(dict(prompt_tokenizer_exact=True,
                    reference=dict(name=case, prompt=MOD.fixed.PROMPTS[case], rendered_prompt=rendered, prompt_tokens=[1, 2, 3]))), encoding='utf-8')
            original = dict(synthetic=False, architecture='glm5-next', model_path=p['Binding']['SourceModel'],
                dimensions={'VocabSize': 154880}, source_dimensions={'glm5-next.vocab_size': 154880}, quality=quality,
                model_files=[dict(path=row['Path'], bytes=row['Bytes'],
                    last_write_utc='1970-01-01T00:00:00.' + str(row['MtimeNs']).zfill(9) + 'Z') for row in p['Binding']['SourceModelFiles']])
            for name in ('glm-e-budget-c2f-v1.json', 'glm-extended-quality-autoplan-v1.json'):
                (source_dir / name).write_text(json.dumps(original), encoding='utf-8')
            args = types.SimpleNamespace(deployment_receipt=paths['deployment'], app=app, variant='native16',
                output=root / 'plans', source_directory=source_dir, model=p['Model'], model_identity=paths['runtime'],
                relocation_receipt=paths['receipt'], relocation_source_manifest=paths['original'])
            MOD.prepare(args)
            for case in MOD.CASES:
                plan = MOD.load(args.output / (case + '-plan.json'))
                MOD.verify_plan_files(plan)
                self.assertEqual(original['model_path'], plan['Binding']['SourceModel'])
                self.assertEqual(p['Model'], plan['Model'])
                self.assertEqual([1, 2, 3], plan['Binding']['PromptTokens'])
                self.assertEqual(plan['Binding']['PromptTokens'], plan['Relocation']['RuntimeBinding']['PromptTokens'])
            paths['receipt'].write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): MOD.verify_plan_files(plan)

    def test_completed_relocation_preserves_original_binding_and_checks_runtime_shards(self):
        p, r, w = relocated_fixture()
        original_binding, original_report = copy.deepcopy(p['Binding']), copy.deepcopy(r)
        result = MOD.qualify(p, r, w)
        self.assertTrue(result['Passed'], result)
        self.assertEqual(original_binding, p['Binding'])
        self.assertEqual(original_report, r)
        self.assertTrue(all(path.startswith('e:/') for path in result['ModelMetadataBinding']['OriginalRequiredModelPaths']))
        self.assertTrue(all(path.startswith('c:/') for path in result['ModelMetadataBinding']['RequiredModelPaths']))
        self.assertEqual(['c:/models/glm/mmproj.gguf'], result['ModelMetadataBinding']['AdditionalManifestPaths'])
        self.assertNotIn('C:/models/glm/mmproj.gguf', r['ModelStatsBefore'])

    def test_changing_only_model_path_never_enables_relocation(self):
        p, r, w = relocated_fixture()
        del p['Relocation']
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture()
        p['Model'] = 'C:/models/guess.gguf'
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_unfinished_or_incompatible_relocation_states_fail(self):
        for state in ('copying', 'part_verified', 'publishing', 'verified', 'failed', 'complete', None, True):
            p, r, w = relocated_fixture()
            p['Relocation']['ReceiptData']['status'] = state
            self.assertFalse(MOD.qualify(p, r, w)['Passed'], state)
        for state in ('pending', 'part_verified', 'publishing', 'retired', 'failed', True):
            p, r, w = relocated_fixture()
            next(iter(p['Relocation']['ReceiptData']['files'].values()))['state'] = state
            self.assertFalse(MOD.qualify(p, r, w)['Passed'], state)

    def test_complete_retirement_requires_full_scope_not_a_selected_subset(self):
        p, r, w = relocated_fixture()
        raw = p['Relocation']['ReceiptData']
        raw.update(status='complete')
        raw['config'].update(retire_source=True, junction=True)
        for item in raw['files'].values(): item['state'] = 'retired'
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        raw['config']['include_files'] = []
        self.assertTrue(MOD.qualify(p, r, w)['Passed'])

    def test_receipt_hashes_sizes_timestamps_and_runtime_binding_are_not_trusted_flags(self):
        changes = [
            lambda p: p['Relocation']['ReceiptData']['files']['glm-00002-of-00003.gguf'].update(sha256='f' * 64),
            lambda p: p['ModelManifest']['files'][1].update(sha256='f' * 64),
            lambda p: p['Relocation']['SourceManifest']['files'][1].update(sha256='f' * 64),
            lambda p: p['Relocation']['ReceiptData']['files']['glm-00002-of-00003.gguf']['source_snapshot'].update(size=21),
            lambda p: p['Relocation']['ReceiptData']['files']['glm-00002-of-00003.gguf']['destination_snapshot'].update(mtime_ns=1201),
            lambda p: p['Relocation']['SourceManifest']['files'][1].update(mtime_ns=201),
            lambda p: p['Relocation']['RuntimeBinding'].update(PromptTokens=[1, 2, 4]),
            lambda p: p['Relocation']['RuntimeBinding'].update(SourceModel='C:/models/other.gguf'),
            lambda p: p['Relocation']['RuntimeBinding']['SourceModelFiles'][0].update(MtimeNs=999),
            lambda p: p['Relocation']['ReceiptData'].update(logical_source_bytes=99),
            lambda p: p['Relocation']['ReceiptData']['config'].update(retire_source=0),
            lambda p: p['Relocation']['ReceiptData']['config'].update(junction=True),
        ]
        for index, change in enumerate(changes):
            p, r, w = relocated_fixture(); change(p)
            with self.subTest(index=index):
                self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_missing_extra_duplicate_or_guessed_relocation_mappings_fail(self):
        def rename_mapping(p, name):
            rows = p['Relocation']['ReceiptData']['files']
            rows[name] = rows.pop('glm-00002-of-00003.gguf')
        changes = [
            lambda p: p['Relocation']['ReceiptData']['files'].pop('glm-00002-of-00003.gguf'),
            lambda p: p['Relocation']['ReceiptData']['files'].update({'GLM-00002-OF-00003.GGUF': copy.deepcopy(p['Relocation']['ReceiptData']['files']['glm-00002-of-00003.gguf'])}),
            lambda p: p['ModelManifest']['files'].append(copy.deepcopy(p['ModelManifest']['files'][1])),
            lambda p: p['Relocation']['SourceManifest']['files'].pop(1),
            lambda p: rename_mapping(p, '../glm-00002-of-00003.gguf'),
            lambda p: rename_mapping(p, '/glm-00002-of-00003.gguf'),
            lambda p: rename_mapping(p, 'sub/../glm-00002-of-00003.gguf'),
            lambda p: p['Relocation']['ReceiptData']['config'].update(destination='C:/models/other'),
            lambda p: p['Relocation']['ReceiptData']['config'].update(source='E:/models/glm/../glm'),
            lambda p: p['Relocation']['ReceiptData']['config']['include_files'].pop(),
            lambda p: p['Relocation']['ReceiptData']['config']['include_files'].append('GLM-00001-OF-00003.GGUF'),
        ]
        for index, change in enumerate(changes):
            p, r, w = relocated_fixture(); change(p)
            with self.subTest(index=index):
                self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_relocated_runtime_requires_exact_new_metadata_for_every_required_shard(self):
        for field, value in [('Bytes', 11), ('LastWriteTimeUtcTicks', 621355968000000012)]:
            p, r, w = relocated_fixture()
            r['ModelStatsBefore']['C:/models/glm/glm-00001-of-00003.gguf'][field] = value
            r['ModelStatsAfter'] = copy.deepcopy(r['ModelStatsBefore'])
            self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = relocated_fixture()
        r['ModelStatsBefore'].pop('C:/models/glm/glm-00003-of-00003.gguf')
        r['ModelStatsAfter'] = copy.deepcopy(r['ModelStatsBefore'])
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_relocation_source_and_receipt_small_file_identities_are_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            p, _, _ = relocated_fixture()
            root = pathlib.Path(directory)
            receipt = root / 'receipt.json'; source = root / 'source.json'
            receipt.write_text(json.dumps(p['Relocation']['ReceiptData']), encoding='utf-8')
            source.write_text(json.dumps(p['Relocation']['SourceManifest']), encoding='utf-8')
            p['Relocation']['Receipt'] = MOD.identity(receipt)
            p['Relocation']['SourceManifestIdentity'] = MOD.identity(source)
            MOD.verify_relocation_files(p)
            receipt.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): MOD.verify_relocation_files(p)
            receipt.write_text(json.dumps(p['Relocation']['ReceiptData']), encoding='utf-8')
            source.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): MOD.verify_relocation_files(p)

    def test_deployment_requires_exact_variant_and_application_path(self):
        files = [dict(Name='GgmlOps.dll', Bytes=32, Sha256='a' * 64),
            dict(Name='GlmUnifiedSessionProbe.dll', Bytes=10, Sha256='b' * 64)]
        receipt = [dict(Variant='native17', Directory='/app17', Files=files, NativeSha256='a' * 64)]
        _, native, managed, _ = MOD.deployment(receipt, '/app17', 'native17')
        self.assertEqual('/app17/GgmlOps.dll', native['Path'].replace('\\', '/'))
        self.assertEqual('b' * 64, managed[0]['Sha256'])
        for app, variant in [('/app17', 'native16'), ('/app16', 'native17'),
            ('/app17', 'native0'), ('/app17', 'native017'), ('/app17', True), ('/app17', 'candidate')]:
            with self.subTest(app=app, variant=variant), self.assertRaises(ValueError):
                MOD.deployment(receipt, app, variant)
        with self.assertRaises(ValueError):
            MOD.deployment(receipt + receipt, '/app17', 'native17')
        bad = copy.deepcopy(receipt)
        bad[0]['NativeSha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            MOD.deployment(bad, '/app17', 'native17')

    def test_default_deployment_keeps_native16_and_rejects_duplicate_files(self):
        files = [dict(Name='GgmlOps.dll', Bytes=32, Sha256='a' * 64),
            dict(Name='GlmUnifiedSessionProbe.dll', Bytes=10, Sha256='b' * 64)]
        receipt = [dict(Variant='native16', Directory='/app', Files=files, NativeSha256='a' * 64)]
        self.assertEqual('a' * 64, MOD.deployment(receipt, '/app')[1]['Sha256'])
        bad = copy.deepcopy(receipt)
        bad[0]['Files'].append(copy.deepcopy(files[0]))
        with self.assertRaises(ValueError):
            MOD.deployment(bad, '/app')

    def test_unknown_probe_is_graded_independently_for_all_tasks_without_mutation(self):
        for case in MOD.CASES:
            with self.subTest(case=case):
                p, r, w = fixture(case); original = copy.deepcopy(r)
                result = MOD.qualify(p, r, w)
                self.assertTrue(result['Passed'], result)
                self.assertFalse(result['OriginalProbePassed'])
                self.assertEqual(1, result['OriginalWrapperExit'])
                self.assertEqual(original, r)
                self.assertIsNone(result['FinalLogitsFinite'])

    def test_counterfeit_success_timeout_or_wrong_exit_never_passes(self):
        for key, value in [('exit_code', 0), ('exit_code', 2), ('exit_code', 134), ('timed_out', True), ('complete', False), ('pid', 42)]:
            with self.subTest(key=key, value=value):
                p, r, w = fixture(); w[key] = value
                self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_early_eos_missing_eos_and_bool_tokens_fail_even_if_receipt_says_passed(self):
        for value in [[99, 6, 99], [5, 6], [True, 6, 99]]:
            p, r, w = fixture(); r['OutputTokens'] = value
            self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_stale_round_extra_commit_and_cleanup_leak_fail(self):
        for field, value in [('ComputedTokens', 7), ('OutputTokensAfterCleanup', [5, 7, 99]),
            ('PromptTokens', [1, 2, 4]), ('SessionDisposed', False), ('StructuralPassed', False),
            ('Passed', True), ('SemanticPassed', True), ('FinalLogitsFinite', False)]:
            with self.subTest(field=field):
                p, r, w = fixture(); r[field] = value
                self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); r['BudgetAfterSessionDisposed']['Current'][0]['Committed'] = 1
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_metadata_changes_native_drift_and_missing_fields_fail_closed(self):
        p, r, w = fixture(); r['ModelStatsAfter'] = {'changed': True}
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); r['LoadedNativeAfter'][0]['Sha256'] = 'f' * 64
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        for field in ['Content', 'ToolCalls', 'Completion', 'SourceToolsBefore', 'InputFileAfter']:
            p, r, w = fixture(); del r[field]
            self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_optional_projector_manifest_does_not_invent_a_loaded_attachment(self):
        p, r, w = fixture()
        p['ModelManifest']['files'].append(dict(path='mmproj-F16.gguf', bytes=20, mtime_ns=200, sha256='1' * 64))
        result = MOD.qualify(p, r, w)
        self.assertTrue(result['Passed'], result)
        self.assertEqual(['model.gguf'], result['ModelMetadataBinding']['RequiredModelPaths'])
        self.assertEqual(['mmproj-F16.gguf'], result['ModelMetadataBinding']['AdditionalManifestPaths'])
        self.assertNotIn('mmproj-F16.gguf', r['ModelStatsBefore'])

    def test_timestamp_precision_is_exact_at_real_seven_digit_windows_resolution(self):
        p, r, w = fixture()
        ticks = 639272497709848349
        ns = MOD.utc_ns('2026-10-10T17:22:50.9848349Z')
        self.assertEqual(1791652970984834900, ns)
        self.assertEqual(ns, (ticks - 621355968000000000) * 100)
        p['Binding']['SourceModelFiles'][0]['MtimeNs'] = ns
        p['ModelManifest']['files'][0]['mtime_ns'] = ns
        for phase in ('ModelStatsBefore', 'ModelStatsAfter'):
            r[phase]['model.gguf']['LastWriteTimeUtcTicks'] = ticks
        self.assertTrue(MOD.qualify(p, r, w)['Passed'])
        for drift in (1, 100, -100):
            bad = copy.deepcopy(p)
            bad['Binding']['SourceModelFiles'][0]['MtimeNs'] += drift
            self.assertFalse(MOD.qualify(bad, r, w)['Passed'])
        self.assertEqual(ns + 1, MOD.utc_ns('2026-10-10T17:22:50.984834901Z'))
        with self.assertRaises(ValueError): MOD.utc_ns('2026-10-10T17:22:50.9848349001Z')

    def test_model_schema_duplicate_missing_extra_runtime_or_noninteger_are_rejected(self):
        mutations = [
            lambda p, r: p['Binding']['SourceModelFiles'].append(copy.deepcopy(p['Binding']['SourceModelFiles'][0])),
            lambda p, r: p['ModelManifest']['files'].append(copy.deepcopy(p['ModelManifest']['files'][0])),
            lambda p, r: p['ModelManifest']['files'].clear(),
            lambda p, r: r['ModelStatsBefore'].update(other=dict(Path='other', Bytes=20, LastWriteTimeUtcTicks=621355968000000002)),
            lambda p, r: r['ModelStatsBefore'].update(wrong=r['ModelStatsBefore'].pop('model.gguf')),
            lambda p, r: r['ModelStatsBefore']['model.gguf'].update(LastWriteTimeUtcTicks=621355968000000001.0),
            lambda p, r: p['Binding']['SourceModelFiles'][0].update(MtimeNs=100.0),
            lambda p, r: p['ModelManifest']['files'][0].update(bytes=True),
            lambda p, r: r['ModelStatsAfter']['model.gguf'].update(LastWriteTimeUtcTicks=621355968000000002)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                p, r, w = fixture(); mutation(p, r)
                self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_revalidation_origin_preserves_command_inputs_and_original_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            old_checker = root / 'checker-v1.py'; old_checker.write_text('original checker\n', encoding='utf-8')
            origin, _, _ = fixture(); origin.update(Checker=MOD.identity(old_checker), ExecutionStatus='not-run')
            original_path = root / 'origin-plan.json'; original_path.write_text(json.dumps(origin), encoding='utf-8')
            revised = copy.deepcopy(origin)
            revised.update(Checker=MOD.identity(TOOL), OriginPlan=MOD.identity(original_path),
                OriginCheckerSnapshot=MOD.identity(old_checker), ExecutionStatus='revalidation-only-existing-execution',
                RevalidationScope='Only regrade existing capture')
            MOD.verify_revalidation_origin(revised)
            for mutation in (lambda p: p['Command'].append('--different-request'),
                lambda p: p['InputFile'].update(Sha256='0' * 64),
                lambda p: p.update(ExecutionStatus='not-run')):
                bad = copy.deepcopy(revised); mutation(bad)
                with self.assertRaises(ValueError): MOD.verify_revalidation_origin(bad)
            original_path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): MOD.verify_revalidation_origin(revised)

    def test_known_controls_cannot_be_silently_changed(self):
        for field, value in [('EnableGlmSelectedExperts', False), ('EnableAsyncExpertPrefill', True),
            ('EnableExpertCacheCapacityGrowth', True), ('RetainedConversationCount', 2), ('PrefillChunkTokens', 64)]:
            p, r, w = fixture(); r['SessionOptions'][field] = value
            self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); r['Scheduler']['Enabled'] = 0
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_thinking_reopening_or_real_tool_call_fails(self):
        p, r, w = fixture(); r['RawText'] += '<think>'; r['RawTextAfterCleanup'] = r['RawText']
        r['Rounds'][0]['RawText'] = r['RawText']
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); r['ToolCalls'] = [{'name': 'get_weather'}]; r['Rounds'][0]['ToolCalls'] = r['ToolCalls']
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_semantic_oracle_does_not_repair_fences_or_duplicate_json(self):
        for case, answer in [('extraction', 'pine, maple'), ('code', '```python\ndef square(x):\n    return x*x\n```'),
            ('code', 'import os\ndef square(x):\n    return x*x'),
            ('tool_json', '{"name":"bad","name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}')]:
            p, r, w = fixture(case); r['Content'] = answer; r['Rounds'][0]['Content'] = answer
            result = MOD.qualify(p, r, w)
            self.assertTrue(result['IntegrityPassed'], result)
            self.assertFalse(result['IndependentSemantic']['passed'])
            self.assertFalse(result['Passed'])

    def test_direct_input_binding_ignores_generated_answer_and_top_level_benchmark_ids(self):
        case = 'code'; rendered = MOD.PREFIX + MOD.fixed.PROMPTS[case] + MOD.SUFFIX
        source = dict(synthetic=False, architecture='glm5-next', model_path='model.gguf', prompt_tokens=[99],
            dimensions={'VocabSize': 100}, source_dimensions={'glm5-next.vocab_size': 100}, model_files=[], quality=[dict(name=case,
                prompt=MOD.fixed.PROMPTS[case], rendered_prompt=rendered,
                rendered_prompt_utf8_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                prompt_tokens=[1, 2, 3], generated_tokens=[99], parsed_text='WRONG HISTORICAL ANSWER', passed=False)])
        bound = MOD.bind_direct(source, case, 100)
        self.assertEqual([1, 2, 3], bound['PromptTokens'])
        source['quality'][0]['prompt'] += ' extra'
        with self.assertRaises(ValueError): MOD.bind_direct(source, case, 100)

    def test_wrong_maya_input_and_duplicate_json_keys_rejected(self):
        p, _, _ = fixture(); row = dict(name=p['Case'], prompt=p['Binding']['Prompt'],
            rendered_prompt=p['Binding']['RenderedPrompt'], prompt_tokens=[1, 2, 4])
        with self.assertRaises(ValueError): MOD.bind_maya(dict(prompt_tokenizer_exact=True, reference=row), p['Binding'])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'bad.json'; path.write_text('{"Passed":false,"Passed":true}', encoding='utf-8')
            with self.assertRaises(ValueError): MOD.load(path)

    def test_wrong_model_shards_and_owned_observation_cannot_be_mixed(self):
        p, r, w = fixture(); p['Model'] = 'other.gguf'
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); r['ModelStatsBefore']['model.gguf']['Bytes'] = 11; r['ModelStatsAfter'] = copy.deepcopy(r['ModelStatsBefore'])
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); w['telemetry']['loaded_native']['pid'] = 42
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); w['command'][-1] = 'different-task'
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])

    def test_boolean_input_or_bad_identity_type_never_qualifies(self):
        for field in ['PromptTokens', 'PublishedTokens', 'OutputTokensAfterCleanup']:
            p, r, w = fixture(); r[field][0] = True
            self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        for value in [0, 1, 'true', {}]:
            p, r, w = fixture(); r['FinalLogitsFinite'] = value
            self.assertFalse(MOD.qualify(p, r, w)['Passed'])
        p, r, w = fixture(); r['IntendedNative']['Sha256'] = None
        self.assertFalse(MOD.qualify(p, r, w)['Passed'])


if __name__ == '__main__':
    unittest.main()
