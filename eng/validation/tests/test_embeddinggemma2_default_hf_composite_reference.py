import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "embeddinggemma2-default-hf-composite-reference.py"
spec = importlib.util.spec_from_file_location("default_hf_reference_test", PATH)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


class DefaultHfEvidenceTests(unittest.TestCase):
    def report(self):
        flags = {name: None for name in tool.environment.OPTIMIZATION_FLAGS}
        return dict(completed=True, passed=True, server_exit_code=0, require_embedding_default=True,
            audio_log_policy="HuggingFaceAddFloor", model=dict(sha256="m"), projector=dict(sha256="p"),
            binary_artifacts=[dict(path="Models/TensorSharp.Models.dll",sha256="models"),
                dict(path="Runtime/TensorSharp.Runtime.dll",sha256="runtime")],
            environment_audit=dict(require_embedding_default=True,resolved_server_environment=True,
                scope="direct host child environment",audio_log_policy="HuggingFaceAddFloor",
                optimization_flags=flags,optimization_flag_sources=dict(flags)),media=[],
            cases=[dict(name=name,parts=[dict(kind="text",value=name)]) for name in tool.CASE_NAMES])

    def proof(self):
        defaults=dict(video_prefix="",include_video_timestamps=False,image_soft_tokens=280,
            video_soft_tokens=140,video_frames_per_second=1,max_video_frames=32,audio_log_policy="HuggingFaceAddFloor",
            type="TensorSharp.Models.Embeddings.EmbeddingGemma2MultimodalOptions")
        return dict(completed=True,source_report_kind="default_hf_http",controlled_video_prefix=False,
            video_prefix="",include_video_timestamps=False,image_soft_tokens=280,image_reference_sizing=True,
            video_soft_tokens=140,video_frames_per_second=1,max_video_frames=32,
            models_assembly_sha256="models",runtime_assembly_sha256="runtime",
            audio_log_policy="HuggingFaceAddFloor",default_options_observed=defaults)

    def test_complete_default_evidence(self):
        tool.validate_proof(self.proof())
        tool.validate_http(self.report(),self.proof(),"m","p")

    def test_default_guard_rejects_instrumentation_and_missing_selectors(self):
        for value in ("trace.json", "", "0"):
            report=self.report();report["environment_audit"]["optimization_flags"]["TSG_CUPTI_ACTIVITY_OUTPUT"]=value
            with self.assertRaises(ValueError):tool.validate_http(report,self.proof(),"m","p")
        report=self.report();del report["environment_audit"]["optimization_flags"]["TSG_CUPTI_ACTIVITY_OUTPUT"]
        with self.assertRaises(ValueError):tool.validate_http(report,self.proof(),"m","p")

    def test_rejects_failed_shutdown_and_stale_assembly(self):
        for change in (dict(server_exit_code=1),dict(completed=False),dict(passed=False),dict(binary_artifacts_unchanged=False)):
            report=self.report();report.update(change)
            with self.assertRaises(ValueError):tool.validate_http(report,self.proof(),"m","p")
        proof=self.proof();proof["models_assembly_sha256"]="stale"
        with self.assertRaises(ValueError):tool.validate_http(self.report(),proof,"m","p")

    def test_rejects_inconsistent_audit_and_observed_defaults(self):
        for change in (dict(scope="unresolved wrapper"),dict(audio_log_policy="LlamaClampFloor")):
            report=self.report();report["environment_audit"].update(change)
            with self.assertRaises(ValueError):tool.validate_http(report,self.proof(),"m","p")
        proof=self.proof();proof["default_options_observed"]["include_video_timestamps"]=True
        with self.assertRaises(ValueError):tool.validate_proof(proof)

    def test_rejects_controlled_and_changed_sampling_proof(self):
        for change in (dict(controlled_video_prefix=True),dict(video_prefix="Video:"),
                dict(include_video_timestamps=True),dict(video_frames_per_second=2),dict(max_video_frames=31)):
            proof=self.proof();proof.update(change)
            with self.assertRaises(ValueError):tool.validate_proof(proof)

    def test_rejects_missing_case_and_media_identity(self):
        report=self.report();report["cases"].pop()
        with self.assertRaises(ValueError):tool.validate_http(report,self.proof(),"m","p")
        report=self.report();report["cases"][0]["parts"]=[dict(kind="image",value="unbound.png")]
        with self.assertRaises(ValueError):tool.validate_http(report,self.proof(),"m","p")

    def test_both_protocols_and_exact_accounting_gate(self):
        expected=np.zeros(768);expected[0]=1
        row=dict(tokens=4,embedding=expected.tolist(),ollama_embedding=expected.tolist())
        self.assertTrue(all(x["passed"] for x in tool.compare(expected,row,4).values()))
        row["ollama_embedding"]=np.roll(expected,1).tolist()
        self.assertFalse(tool.compare(expected,row,4)["ollama"]["passed"])
        with self.assertRaises(ValueError):tool.compare(expected,row,5)

    def test_nonidentity_resize_and_pixel_evidence(self):
        hf=tool.module("hf_resize_reference_test",tool.ROOT/"eng/diffusiongemma-vision-oracle.py")
        with tempfile.TemporaryDirectory() as temporary:
            parent=Path(temporary)
            rgba=np.full((2,3,4),255,dtype=np.uint8);rgba[:,0,:3]=[255,0,0];rgba[:,1,:3]=[0,255,0]
            pixels=hf._resize_chw(rgba[:,:,:3].transpose(2,0,1),48,48,mode="float")*np.float32(1/255)
            rgba.tofile(parent/"input.rgba8");pixels.astype("<f4").tofile(parent/"input.f32")
            record=dict(source_width=3,source_height=2,processed_width=48,processed_height=48,soft_tokens=1,
                rgba="input.rgba8",pixels="input.f32",rgba_sha256=tool.sha(parent/"input.rgba8"),pixels_sha256=tool.sha(parent/"input.f32"))
            _, observation=tool.validate_pixels(record,parent,rgba,hf)
            self.assertEqual(0,observation["max_absolute_error"])
            changed=rgba.copy();changed[0,0,0]-=1
            with self.assertRaises(ValueError):tool.validate_pixels(record,parent,changed,hf)
            pixels[0,0,0]+=.01;pixels.astype("<f4").tofile(parent/"input.f32")
            record["pixels_sha256"]=tool.sha(parent/"input.f32")
            with self.assertRaises(ValueError):tool.validate_pixels(record,parent,rgba,hf)

    def test_vision_cache_binds_spatial_geometry(self):
        square=np.zeros((3,96,96),dtype=np.float32)
        rectangle=np.zeros((3,48,192),dtype=np.float32)
        self.assertEqual(square.tobytes(),rectangle.tobytes())
        self.assertNotEqual(tool.vision_key(square,"same-bytes"),tool.vision_key(rectangle,"same-bytes"))

    def test_runtime_guard_binds_one_complete_unchanged_snapshot(self):
        report=self.report();before={r["path"]:r["sha256"] for r in report["binary_artifacts"]}
        guard=dict(artifacts_unchanged=True,before=before,after=dict(before))
        tool.validate_runtime_guards(report,[guard])
        bad=copy.deepcopy(guard);bad["after"][next(iter(before))]="changed"
        with self.assertRaises(ValueError):tool.validate_runtime_guards(report,[bad])
        bad=copy.deepcopy(guard);bad["before"]={"unrelated":"hash"};bad["after"]=dict(bad["before"])
        with self.assertRaises(ValueError):tool.validate_runtime_guards(report,[bad])


if __name__ == "__main__":
    unittest.main()
