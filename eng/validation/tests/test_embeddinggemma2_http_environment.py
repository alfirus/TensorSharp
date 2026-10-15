import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

VALIDATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATION))
import embeddinggemma2_http_environment as environment

SPEC = importlib.util.spec_from_file_location("embeddinggemma2_media_http_guard_test", VALIDATION / "embeddinggemma2-media-http.py")
MEDIA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEDIA)
HOST = ["dotnet", "/frozen/TensorSharp.Server.Host.dll", "--backend", "ggml_metal"]


class HttpEnvironmentAuditTests(unittest.TestCase):
    def test_clean_default_direct_host_is_accepted(self):
        audit = environment.audit_child_environment(HOST, {"PATH": "/bin"}, True)
        self.assertEqual(HOST, audit.argv)
        self.assertEqual({"PATH": "/bin"}, audit.environment)
        self.assertTrue(audit.resolved_server_environment)
        self.assertTrue(audit.metadata()["require_embedding_default"])
        self.assertTrue(all(value is None for value in audit.metadata()["optimization_flags"].values()))

    def test_explicit_thread_policy_is_recorded_without_exposing_other_environment(self):
        audit = environment.audit_child_environment(["env", "TS_GGML_CPU_THREADS=8", "OPENBLAS_NUM_THREADS=1",
            "GOMP_SPINCOUNT=0", *HOST], {"PRIVATE_TOKEN": "private-thread-policy-credential",
            "OMP_NUM_THREADS": "8", "OMP_WAIT_POLICY": "PASSIVE"}, True)
        self.assertEqual({"TS_GGML_CPU_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "GOTO_NUM_THREADS": None,
            "OMP_NUM_THREADS": "8", "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"},
            audit.metadata()["thread_policy"])
        self.assertNotIn("private-thread-policy-credential", json.dumps(audit.metadata()))

    def test_every_known_inherited_override_is_rejected_including_zero_and_empty(self):
        for name in environment.OPTIMIZATION_FLAGS:
            for value in ("1", "0", ""):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError) as failure:
                    environment.audit_child_environment(HOST, {name: value}, True)
                self.assertIn(name, str(failure.exception))
        for name in ("TS_EMBEDDING_CUDA_NORMALIZED_HALF", "TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION_TF32"):
            self.assertIn(name, environment.OPTIMIZATION_FLAGS)
        for name in ("TSG_EMBEDDING_LINUX_BLAS", "TSG_EMBEDDING_BLAS_LIBRARY", "TSG_EMBEDDING_CPU_ATTENTION", "TSG_EMBEDDING_CPU_PROFILE", "TS_VISION_PROF", "TSG_CUPTI_ACTIVITY_OUTPUT"):
            self.assertIn(name, environment.OPTIMIZATION_FLAGS)

    def test_known_command_override_is_rejected_before_launch(self):
        command = ["/usr/bin/env", "TS_GEMMA4_AUDIO_METAL_FUSED_OPS=1", *HOST]
        with self.assertRaises(ValueError) as failure:
            environment.audit_child_environment(command, {}, True)
        self.assertIn("TS_GEMMA4_AUDIO_METAL_FUSED_OPS", str(failure.exception))

    def test_guarded_clamp_command_is_rejected_and_ordinary_clamp_keeps_canonical_oracle_metadata(self):
        command = ["env", "TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp", *HOST]
        with self.assertRaises(ValueError):
            environment.audit_child_environment(command, {}, True)
        audit = environment.audit_child_environment(command, {})
        self.assertEqual("LlamaClampFloor", audit.audio_log_policy())
        self.assertEqual("llama_clamp", audit.metadata()["optimization_flags"]["TS_EMBEDDING_AUDIO_LOG_POLICY"])
        self.assertEqual(HOST, audit.reported_command)
        # The independent canonical oracle consumes this top-level field even
        # though env assignments have been removed from reported_command.
        self.assertEqual("LlamaClampFloor", {"audio_log_policy": audit.audio_log_policy()}["audio_log_policy"])

    def test_direct_host_audio_policy_mapping_records_default_explicit_and_invalid_values(self):
        for value, expected in ((None, "HuggingFaceAddFloor"), ("", "HuggingFaceAddFloor"),
                                ("huggingface_add", "HuggingFaceAddFloor"), ("llama_clamp", "LlamaClampFloor"), ("invalid", "Invalid")):
            with self.subTest(value=value):
                inherited = {} if value is None else {"TS_EMBEDDING_AUDIO_LOG_POLICY": value}
                self.assertEqual(expected, environment.audit_child_environment(HOST, inherited).audio_log_policy())
        self.assertIsNone(environment.audit_child_environment(["bash", "-lc", "launch"], {}).audio_log_policy())

    def test_leading_assignments_are_literal_and_exact_child_environment_is_resolved(self):
        private = "private-value-$HOME-$(not-executed)=tail"
        inherited = {"PATH": "/bin", "PRIVATE_TOKEN": "old"}
        command = ["env", "PRIVATE_TOKEN=" + private, "TENSORSHARP_LOG_LEVEL=Warning", *HOST]
        audit = environment.audit_child_environment(command, inherited, True)
        self.assertEqual(HOST, audit.argv)
        self.assertEqual(private, audit.environment["PRIVATE_TOKEN"])
        self.assertEqual("Warning", audit.environment["TENSORSHARP_LOG_LEVEL"])
        self.assertEqual("old", inherited["PRIVATE_TOKEN"])
        serialized = json.dumps({"command": audit.reported_command, "environment_audit": audit.metadata()})
        self.assertNotIn(private, serialized)
        self.assertNotIn("PRIVATE_TOKEN", serialized)
        self.assertNotIn("TENSORSHARP_LOG_LEVEL", serialized)

    def test_ordinary_explicit_opt_in_records_actual_last_assignment_and_source(self):
        name = "TS_EMBEDDING_CUDA_NORMALIZED_HALF"
        command = ["env", name + "=0", name + "=1", *HOST]
        audit = environment.audit_child_environment(command, {name: "0"})
        self.assertEqual("1", audit.environment[name])
        self.assertEqual("1", audit.metadata()["optimization_flags"][name])
        self.assertEqual("command", audit.metadata()["optimization_flag_sources"][name])
        self.assertFalse(audit.metadata()["require_embedding_default"])

    def test_nested_literal_env_prefixes_resolve_without_wrappers(self):
        audit = environment.audit_child_environment(["env", "PRIVATE_A=one", "/usr/bin/env", "PRIVATE_B=two", *HOST], {}, True)
        self.assertEqual(HOST, audit.argv)
        self.assertEqual("one", audit.environment["PRIVATE_A"])
        self.assertEqual("two", audit.environment["PRIVATE_B"])
        self.assertNotIn("PRIVATE_", json.dumps(audit.metadata()))

    def test_ambiguous_env_options_shells_scripts_and_other_interpreters_are_rejected(self):
        secret = "do-not-serialize-this-private-value"
        commands = [
            ["env", "-u", "TS_GEMMA4_AUDIO_METAL_FUSED_OPS", *HOST],
            ["env", "-i", *HOST], ["env", "--", *HOST],
            ["env", "-S", "PRIVATE=" + secret + " dotnet /frozen/TensorSharp.Server.Host.dll"],
            ["bash", "-lc", "PRIVATE=" + secret + " dotnet /frozen/TensorSharp.Server.Host.dll"],
            ["/bin/sh", "-c", secret], ["python3", "launch.py"], ["./launch.sh"],
            ["dotnet", "run", "--project", "project"], ["dotnet", "/frozen/Other.dll"],
        ]
        for command in commands:
            with self.subTest(command=command), self.assertRaises(ValueError) as failure:
                environment.audit_child_environment(command, {}, True)
            self.assertNotIn(secret, str(failure.exception))

    def test_unguarded_unsupported_env_options_preserve_original_launcher_semantics(self):
        command = ["env", "-u", "PRIVATE_TOKEN", *HOST]
        inherited = {"PRIVATE_TOKEN": "secret", "TS_GEMMA4_AUDIO_METAL_FUSED_OPS": "1"}
        audit = environment.audit_child_environment(command, inherited)
        self.assertEqual(command, audit.argv)
        self.assertEqual(inherited, audit.environment)
        self.assertFalse(audit.resolved_server_environment)
        self.assertEqual("1", audit.metadata()["optimization_flags"]["TS_GEMMA4_AUDIO_METAL_FUSED_OPS"])
        self.assertNotIn("PRIVATE_TOKEN", json.dumps(audit.metadata()))

    def test_unguarded_shell_keeps_execution_but_omits_secret_wrapper_arguments(self):
        secret = "private-user-environment-value"
        command = ["bash", "-lc", "PRIVATE=" + secret + " dotnet host.dll"]
        audit = environment.audit_child_environment(command, {"PRIVATE": secret})
        self.assertEqual(command, audit.argv)
        self.assertFalse(audit.resolved_server_environment)
        serialized = json.dumps({"command": audit.reported_command, "environment_audit": audit.metadata()})
        self.assertNotIn(secret, serialized)
        self.assertNotIn("PRIVATE", serialized)
        self.assertIn("unresolved", serialized)

    def test_direct_exec_and_self_contained_frozen_host_are_supported(self):
        for command in (["dotnet", "exec", "/frozen/TensorSharp.Server.Host.dll"],
                        ["/frozen/TensorSharp.Server.Host"], ["/frozen/TensorSharp.Server.Host.exe"]):
            with self.subTest(command=command):
                self.assertTrue(environment.audit_child_environment(command, {}, True).resolved_server_environment)

    def test_invalid_command_shapes_never_expose_command_values(self):
        for command in (None, [], "secret", ["dotnet", 12], ["dotnet\0secret"]):
            with self.subTest(command=command), self.assertRaises(ValueError) as failure:
                environment.audit_child_environment(command, {}, True)
            self.assertNotIn("secret", str(failure.exception))

    def test_default_guard_rejects_before_input_reads_artifacts_and_server_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            argv = ["probe", "--command", json.dumps(HOST), "--model", "test", "--model-file", "missing.gguf",
                    "--mmproj", "missing-mmproj.gguf", "--cases", "missing-cases.json", "--output", str(output),
                    "--expected-backend", "ggml_metal", "--require-embedding-default"]
            with patch.dict(os.environ, {"TS_GEMMA4_AUDIO_METAL_FUSED_OPS": "1"}, clear=True), patch.object(sys, "argv", argv), \
                    patch.object(MEDIA.subprocess, "Popen") as process:
                with self.assertRaises(ValueError):
                    MEDIA.main()
                process.assert_not_called()
            self.assertFalse(output.exists())

    def test_popen_receives_exact_audited_environment_with_no_server_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root / "model.gguf"; model.write_bytes(b"model")
            projector = root / "projector.gguf"; projector.write_bytes(b"projector")
            manifest = root / "cases.json"; manifest.write_text(json.dumps({"cases": [{"name": "text", "parts": [{"kind": "text", "value": "test"}]}]}))
            host = root / "TensorSharp.Server.Host.dll"; host.write_bytes(b"fake-assembly-not-executed")
            private = "private-credential-not-for-report"
            command = ["env", "PRIVATE_TOKEN=" + private, "TENSORSHARP_LOG_LEVEL=Warning", "dotnet", str(host)]
            output = root / "output"
            argv = ["probe", "--command", json.dumps(command), "--model", "test", "--model-file", str(model),
                    "--mmproj", str(projector), "--cases", str(manifest), "--output", str(output),
                    "--expected-backend", "ggml_metal", "--require-embedding-default"]
            with patch.dict(os.environ, {"PRIVATE_TOKEN": "inherited", "PATH": "/bin"}, clear=True), patch.object(sys, "argv", argv), \
                    patch.object(MEDIA.subprocess, "Popen", side_effect=RuntimeError("test-stop-before-any-process")) as process:
                with self.assertRaisesRegex(RuntimeError, "test-stop"):
                    MEDIA.main()
                self.assertEqual(["dotnet", str(host)], process.call_args.args[0])
                child = process.call_args.kwargs["env"]
                self.assertEqual(private, child["PRIVATE_TOKEN"])
                self.assertEqual("Warning", child["TENSORSHARP_LOG_LEVEL"])
                self.assertEqual(str(output.resolve() / "requests-temp") + os.sep, child["TMPDIR"])
                self.assertTrue(all(name not in child for name in environment.OPTIMIZATION_FLAGS))
            self.assertEqual("", (output / "server.log").read_text())


if __name__ == "__main__":
    unittest.main()
