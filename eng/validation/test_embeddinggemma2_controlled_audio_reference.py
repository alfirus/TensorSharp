"""Offline input-policy gates for the separate controlled-clamp diagnostic."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('controlled_audio', Path(__file__).with_name('embeddinggemma2-controlled-audio-reference.py'))
controlled = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controlled)


class ControlledAudioInputTests(unittest.TestCase):
    def setUp(self):
        parts = [dict(kind='audio', value='/fixture/tone.wav')]
        self.pair = dict(completed=True, model=dict(sha256='model'), projector=dict(sha256='projector'),
                         inputs=[dict(name='audio', parts=parts)], media=[dict(path='/fixture/tone.wav', sha256='wave')],
                         engines={name: dict(completed=True, server_exit_code=0,
                                  command=['env', 'TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp'],
                                  benchmarks=dict(audio=dict(parts=copy.deepcopy(parts)))) for name in ('tensorsharp', 'llama')})
        self.audio = dict(audio_log_policy='llama_clamp', past_distance_inclusive=11,
                          projector_sha256='projector', waveform_sha256='wave')

    def test_explicit_controlled_input_is_accepted(self):
        controlled.validate_inputs(self.pair, self.audio, 'model')

    def test_default_hf_or_previous_mask_cannot_be_called_controlled_clamp(self):
        for key, value in [('audio_log_policy', 'huggingface_add'), ('past_distance_inclusive', 12)]:
            changed = copy.deepcopy(self.audio)
            changed[key] = value
            with self.assertRaises(ValueError):
                controlled.validate_inputs(self.pair, changed, 'model')

    def test_model_projector_waveform_and_actual_sequence_must_match(self):
        changes = [lambda p: p['model'].update(sha256='different'),
                   lambda p: p['projector'].update(sha256='different'),
                   lambda p: p['media'][0].update(sha256='different'),
                   lambda p: p['engines']['llama']['benchmarks']['audio'].update(parts=[dict(kind='text', value='prefix')])]
        for change in changes:
            changed = copy.deepcopy(self.pair)
            change(changed)
            with self.assertRaises(ValueError):
                controlled.validate_inputs(changed, self.audio, 'model')

    def test_completed_outputs_do_not_hide_a_process_cleanup_failure(self):
        for name in ('tensorsharp', 'llama'):
            changed = copy.deepcopy(self.pair)
            changed['engines'][name]['server_exit_code'] = 134
            with self.assertRaises(ValueError):
                controlled.validate_inputs(changed, self.audio, 'model')

    def test_candidate_policy_must_be_explicit_and_unprefixed(self):
        changed = copy.deepcopy(self.pair)
        changed['engines']['tensorsharp']['command'] = ['dotnet', 'server']
        with self.assertRaises(ValueError):
            controlled.validate_inputs(changed, self.audio, 'model')
        changed = copy.deepcopy(self.pair)
        changed['inputs'][0]['parts'].insert(0, dict(kind='text', value='task: search'))
        with self.assertRaises(ValueError):
            controlled.validate_inputs(changed, self.audio, 'model')


if __name__ == '__main__':
    unittest.main()
