"""Auditable child environments for the EmbeddingGemma 2 HTTP probe.

Never serialize the complete environment or unknown env assignments. The
default guard accepts a direct frozen host launcher; wrappers require an
ordinary unguarded experiment and are explicitly marked as unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


# Exact, nonsecret route/precision/diagnostic selectors. No prefix scan: that
# could expose credentials or unrelated user environment values. Register new
# experimental route flags here before qualifying a fresh default run.
OPTIMIZATION_FLAGS = (
    "NVIDIA_TF32_OVERRIDE", "TS_CPU_DISABLE_AVX512", "TSG_EMBEDDING_F32_LINEAR",
    "TSG_EMBEDDING_LINUX_BLAS", "TSG_EMBEDDING_BLAS_LIBRARY",
    "TSG_EMBEDDING_CPU_ATTENTION", "TSG_EMBEDDING_CPU_PROFILE",
    "TSG_EMBEDDING_CPU_ATTENTION_PROFILE",
    "TSG_EMBEDDING_CPU_ATTENTION_SIMD",
    "TSG_CUPTI_ACTIVITY_OUTPUT",
    "TENSORSHARP_EMBEDDING_CPU_PROFILE", "TENSORSHARP_EMBEDDING_FUSED_GATE_GELU",
    "TENSORSHARP_EMBEDDING_NATIVE_ATTENTION", "TENSORSHARP_EMBEDDING_NATIVE_Q16",
    "TENSORSHARP_EMBEDDING_Q16_AVX512", "TENSORSHARP_EMBEDDING_Q8_AVX",
    "TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION", "TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION_TF32",
    "TS_EMBEDDING_CUDA_NORMALIZED_HALF", "TS_EMBEDDING_CUDA_PHASES",
    "TS_EMBEDDING_AUDIO_LOG_POLICY",
    "TS_GEMMA4_AUDIO_SSCP_GRAPH", "TS_GEMMA4_AUDIO_BLOCK_GRAPH", "TS_GEMMA4_AUDIO_PRECISE_ROWS",
    "TS_GEMMA4_AUDIO_METAL_F32_MM", "TS_GEMMA4_AUDIO_METAL_MPS_MM", "TS_GEMMA4_AUDIO_METAL_FUSED_OPS",
    "TS_GEMMA4_AUDIO_METAL_SIMD_MM", "TS_GEMMA4_AUDIO_METAL_PROFILE", "TS_GEMMA4_AUDIO_CUDA_PRECISE_SSCP",
    "TS_GEMMA4_AUDIO_CUDA_PRECISE_LINEAR", "TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC",
    "TS_GEMMA4_VISION_DEVICE", "TS_GEMMA4_VISION_CUDA_DEVICE", "TS_GEMMA4_VISION_CPU_Q16",
    "TS_GEMMA4_VISION_METAL_PRECISE", "TS_GEMMA4_VISION_METAL_MPS_MM", "TS_GEMMA4_VISION_METAL_SPLIT_MM",
    "TS_GEMMA4_VISION_METAL_MIXED_MPS", "TS_GEMMA4_VISION_SPATIAL_DEVICE",
    "TS_GEMMA4_VISION_SPLIT_PROJECTION", "TS_GEMMA4_VISION_SIMD_PROJECTION",
    "TS_GEMMA4_VISION_SPLIT_ATTENTION", "TS_VISION_STAGE", "TS_VISION_TRACE", "TS_VISION_PROF",
)
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
THREAD_POLICY_KEYS = (
    "TS_GGML_CPU_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS",
    "OMP_WAIT_POLICY", "GOMP_SPINCOUNT",
)


@dataclass
class ChildEnvironmentAudit:
    # Only argv/environment are passed to Popen. They are private launch data,
    # not report fields; metadata() deliberately does not serialize them all.
    argv: list[str]
    environment: dict[str, str]
    reported_command: list[str]
    resolved_server_environment: bool
    flag_sources: dict[str, str | None]
    require_default: bool

    def audio_log_policy(self) -> str | None:
        if not self.resolved_server_environment:
            return None
        value = self.environment.get("TS_EMBEDDING_AUDIO_LOG_POLICY")
        if value in (None, "", "huggingface_add"):
            return "HuggingFaceAddFloor"
        if value == "llama_clamp":
            return "LlamaClampFloor"
        return "Invalid"  # Raw nonsecret selector remains explicit below.

    def metadata(self) -> dict:
        return {
            "require_embedding_default": self.require_default,
            "resolved_server_environment": self.resolved_server_environment,
            "scope": "direct host child environment" if self.resolved_server_environment else
                "immediate launcher only; wrapper/server environment unresolved",
            "optimization_flags": {key: self.environment.get(key) for key in OPTIMIZATION_FLAGS},
            "optimization_flag_sources": self.flag_sources,
            "audio_log_policy": self.audio_log_policy(),
            "thread_policy": {key: self.environment.get(key) for key in THREAD_POLICY_KEYS},
            "limitations": "Only known nonsecret selectors are recorded. Unknown environment names/values and wrapper arguments are omitted.",
        }


def _direct_host(argv: list[str]) -> bool:
    name = Path(argv[0]).name.lower()
    if name in ("tensorsharp.server.host", "tensorsharp.server.host.exe"):
        return True
    if name not in ("dotnet", "dotnet.exe"):
        return False
    first = 2 if len(argv) > 1 and argv[1] == "exec" else 1
    return len(argv) > first and Path(argv[first]).name.lower() == "tensorsharp.server.host.dll"


def audit_child_environment(command, inherited, require_default=False) -> ChildEnvironmentAudit:
    """Resolve literal leading env NAME=value prefixes without a shell.

    Unsupported env options/wrappers retain their original unguarded execution
    semantics; metadata then describes only Popen's immediate child. A guarded
    run rejects them before launch rather than guessing a grandchild's env.
    Any present known selector, including =0 or an empty value, is an override.
    Errors contain no environment values or potentially secret command text.
    """
    if not isinstance(command, list) or not command or any(not isinstance(part, str) or "\0" in part for part in command):
        raise ValueError("The server command must be a nonempty argv array without NUL characters.")
    base = dict(inherited)
    environment = dict(base)
    sources = {key: "inherited" if key in environment else None for key in OPTIMIZATION_FLAGS}
    argv = list(command)
    unsupported_env = False
    while argv and Path(argv[0]).name.lower() in ("env", "env.exe"):
        argv.pop(0)
        while argv and _ASSIGNMENT.match(argv[0]):
            name, value = argv.pop(0).split("=", 1)
            environment[name] = value
            if name in sources:
                sources[name] = "command"
        if not argv or argv[0].startswith("-") or "=" in argv[0]:
            unsupported_env = True
            break
    if not argv and not unsupported_env:
        raise ValueError("The server command has no executable.")
    if unsupported_env:
        if require_default:
            raise ValueError("The embedding-default guard cannot resolve env options or an incomplete env launcher.")
        argv = list(command)
        environment = base
        sources = {key: "inherited" if key in base else None for key in OPTIMIZATION_FLAGS}
    resolved = not unsupported_env and _direct_host(argv)
    if require_default and not resolved:
        raise ValueError("The embedding-default guard requires a direct dotnet [exec] TensorSharp.Server.Host.dll or self-contained host; wrappers are ambiguous.")
    if require_default:
        present = [name for name in OPTIMIZATION_FLAGS if name in environment]
        if present:
            raise ValueError("The embedding-default guard requires optimization selectors unset: " + ", ".join(present))
    # Resolved argv has no env assignments. Unresolved wrapper arguments can
    # contain shell assignments or secrets, so omit them from report JSON.
    reported = list(argv) if resolved else ["<unresolved launcher command omitted>"]
    return ChildEnvironmentAudit(argv, environment, reported, resolved, sources, bool(require_default))
