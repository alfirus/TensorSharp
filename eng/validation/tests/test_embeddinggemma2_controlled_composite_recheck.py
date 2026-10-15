import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("composite_recheck", Path(__file__).parents[1] / "embeddinggemma2-controlled-composite-recheck.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CompositeRecheckTests(unittest.TestCase):
    def ablation(self):
        command = ["env", "TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp", "dotnet", "/frozen/TensorSharp.Server.Host.dll"]
        engines = {name: dict(command=command, completed=True, server_exit_code=0) for name in ("tensorsharp", "llama")}
        artifacts = {name: [dict(path="/frozen/TensorSharp.Models.dll", sha256="models"),
            dict(path="/frozen/TensorSharp.Runtime.dll", sha256="runtime")] for name in engines}
        pair = dict(completed=True, binary_artifacts_unchanged=True, dependencies_unchanged_and_clean=True,
            reference_controlled_video_prefix=True, controlled_video_prefix=True,
            model=dict(sha256="model"), projector=dict(sha256="projector"), engines=engines,
            binary_artifacts=artifacts, binary_artifacts_after=copy.deepcopy(artifacts))
        proof = dict(completed=True, model_sha256="model", models_assembly_sha256="models",
            runtime_assembly_sha256="runtime", video_prefix="Video:", video_soft_tokens=140,
            include_video_timestamps=False, video_frames_per_second=1, max_video_frames=32)
        return pair, proof

    def test_explicit_tensorsharp_reference_requires_actual_original_assemblies(self):
        pair, proof = self.ablation()
        tool.bind_tensorsharp_reference(pair, proof, "model", "projector")
        pair["binary_artifacts"]["llama"][0]["sha256"] = "newer-reference"
        pair["binary_artifacts_after"] = copy.deepcopy(pair["binary_artifacts"])
        with self.assertRaises(ValueError): tool.bind_tensorsharp_reference(pair, proof, "model", "projector")

    def test_ablation_rejects_llama_wrapper_missing_prefix_or_changed_binary(self):
        pair, proof = self.ablation()
        for field in ("actual-llama", "wrapper", "missing-prefix", "changed-binary"):
            bad = copy.deepcopy(pair)
            if field == "actual-llama": bad["engines"]["llama"]["command"] = ["/llama-server"]
            if field == "wrapper": bad["engines"]["llama"]["command"] = ["bash", "-lc", "launch"]
            if field == "missing-prefix": bad["reference_controlled_video_prefix"] = False
            if field == "changed-binary": bad["binary_artifacts_after"]["llama"][0]["sha256"] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.bind_tensorsharp_reference(bad, proof, "model", "projector")

    def test_reference_prefix_must_match_the_observed_timed_parts(self):
        parts = [dict(kind="video", value="clip")]
        expected = [1.] + [0.] * 767
        controlled = tool.tool.controlled_parts(parts, "Video:")
        row = dict(parts=controlled, prompt_tokens=4, embedding=expected)
        self.assertTrue(tool.tool.compare(expected, row, controlled, 4)["passed"])
        row["parts"] = parts
        with self.assertRaises(ValueError): tool.tool.compare(expected, row, controlled, 4)

    def test_rejects_ambiguous_or_ineffective_controlled_audio_environment(self):
        def pair(command):return dict(engines=dict(tensorsharp=dict(command=command)))
        key="TS_EMBEDDING_AUDIO_LOG_POLICY="
        tool.controlled_audio_command(pair(["env","TS_GGML_CPU_THREADS=8",key+"llama_clamp","dotnet","host.dll"]))
        for command in (["env",key+"llama_clamp",key+"huggingface_add","dotnet"],
                ["env",key+"llama_clamp",key+"llama_clamp","dotnet"],
                ["dotnet","host.dll",key+"llama_clamp"],
                ["env","dotnet","host.dll",key+"llama_clamp"],
                ["env",key+"llama_clamp"],["env",key+"llama_clamp","--unset=X","dotnet"]):
            with self.subTest(command=command),self.assertRaises(ValueError):tool.controlled_audio_command(pair(command))

    def test_rebuilt_assemblies_require_identical_observed_inputs_and_codecs(self):
        keys = ("completed", "model_sha256", "tokenizer_type", "token_ids", "texts", "case_text_runs",
            "controlled_video_prefix", "video_prefix", "video_prefix_source", "include_video_timestamps",
            "video_soft_tokens", "video_frames_per_second", "max_video_frames", "videos",
            "image_soft_tokens", "image_reference_sizing", "images", "audios", "video_decoder",
            "video_codec_sha256", "dotnet_runtime", "os", "architecture", "execution_scope")
        original = {key:None for key in keys}
        original.update(models_assembly_sha256="old", texts=[dict(text="Video:",ids=[1,2])],
            videos=[dict(frames=[dict(pixels_sha256="pixels")])],
            codec_assemblies=[dict(path="/old/OpenCv.dll",sha256="managed")],
            loaded_native_codecs=[dict(path="/old/codec.so",sha256="native")])
        rebuilt = copy.deepcopy(original)
        rebuilt.update(models_assembly_sha256="new")
        rebuilt["codec_assemblies"][0]["path"]="/new/OpenCv.dll"
        tool.equivalent_proof(original, rebuilt)
        for field in ("token", "pixel", "codec", "sampling"):
            changed = copy.deepcopy(rebuilt)
            if field == "token":changed["texts"][0]["ids"][1]=3
            if field == "pixel":changed["videos"][0]["frames"][0]["pixels_sha256"]="different"
            if field == "codec":changed["loaded_native_codecs"][0]["sha256"]="different"
            if field == "sampling":changed["max_video_frames"]=16
            with self.subTest(field=field), self.assertRaises(ValueError):tool.equivalent_proof(original,changed)

    def test_rejects_changed_order_or_media_identity(self):
        original = dict(inputs=[dict(name="video", parts=[dict(kind="text", value="x"),dict(kind="video", value="v")])],
            media=[dict(path="v", sha256="clip")])
        tool.matching_inputs(original, copy.deepcopy(original))
        changed = copy.deepcopy(original)
        changed["inputs"][0]["parts"].reverse()
        with self.assertRaises(ValueError):tool.matching_inputs(original, changed)
        changed = copy.deepcopy(original)
        changed["media"][0]["sha256"] = "other"
        with self.assertRaises(ValueError):tool.matching_inputs(original, changed)

    def test_rejects_partial_weaker_or_nonfinite_oracle(self):
        valid = dict(completed=True, fresh_unique_video_frames=4, fresh_complete_text_cases=3,
            source_hashes={name:"sha" for name in tool.SOURCE_FILES},
            thresholds=dict(min_cosine=.9999, max_absolute_error=.002),
            cases=[dict(name=name, total_input_rows=4, embedding=[1.] + [0.] * 767)
                for name in ("video", "mixed", "all-modalities")])
        tool.validate_oracle(valid)
        for key, value in (("completed",False),("fresh_unique_video_frames",3),("fresh_complete_text_cases",2),
                ("thresholds",dict(min_cosine=.999, max_absolute_error=.002))):
            changed = dict(valid, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):tool.validate_oracle(changed)
        changed = copy.deepcopy(valid)
        changed["cases"][0]["embedding"][1] = float("nan")
        with self.assertRaises(ValueError):tool.validate_oracle(changed)
        for sources in ({}, {name:"sha" for name in tool.SOURCE_FILES[:-1]}):
            with self.assertRaises(ValueError):tool.validate_oracle(dict(valid,source_hashes=sources))


if __name__ == "__main__":
    unittest.main()
