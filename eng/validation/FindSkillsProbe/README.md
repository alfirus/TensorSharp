# Live find-skills validation

Run from the TensorSharp repository root with .NET 10, Python 3, network access,
and the OS sandbox required by `SkillScriptRunner`:

```sh
dotnet run --project eng/validation/FindSkillsProbe/FindSkillsProbe.csproj -c Release -- artifacts/find-skills/fresh-run
```

The output directory must be fresh. Generated evidence and installed packages stay
under ignored `artifacts/`. The probe uses the production `SkillTools.Execute`
entry point, the real skills.sh search API, immutable GitHub revision archives,
the production registry, and the production sandboxed script runner. It verifies:

- Search, inspect, pinned install, and immediate same-request reading for Vercel's
  `web-design-guidelines` and Anthropic's `skill-creator`.
- All installed file hashes and source commit revisions are recorded.
- The installed `skill-creator` reference and actual Python helper can be read.
- The upstream `aggregate_benchmark.py` executes against six deterministic input
  runs, writes JSON and Markdown artifacts, and computes the expected primary
  mean, baseline mean, and difference. It uses Python's standard library and has
  network access disabled while running.
- The same-request catalog includes newly installed skills, and 20 repeated
  searches return the correct cached results. The report includes individual
  cold operation timings and warm median/p95 timings.

The external skills are kept unchanged. A failed network operation, unsupported
sandbox, missing helper, or wrong result fails the probe and is recorded in
`report.json`; none is counted as passing.

## Genuine local-model exercise

Build `TensorSharp.Server.Host`, then run a copy of its output under an ignored
directory so marketplace installs do not affect another host's library. Point it
at only the repository's `TensorAgent/skills/find-skills` root, select a real
tool-capable model, and enable:

```text
--host 127.0.0.1 --port 5049
--skills-dir /absolute/repo/TensorAgent/skills/find-skills
--skills-no-discovery --skills-allow-exec --skills-allow-network
--skills-max-rounds 16 --max-tokens 2200
```

For existing native binaries, the managed build can use
`-p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true`. Record the
actual upstream ggml revision, cleanliness, native binary hash, loaded model,
backend, and launch command alongside the generated report. Do not patch ggml.

Once the host is ready:

```sh
python3 eng/validation/FindSkillsProbe/model_probe.py \
  --model /absolute/path/model.gguf --backend ggml_metal \
  --out artifacts/find-skills/model-report.json
```

This requests real model-driven search, inspection, installation, and reading.
The probe explicitly requests temperature `0` and seed `42`, both supported by
the Web UI chat request parser, and records them in the report. Use
`--temperature 0.8 --seed -1` for an unseeded sample matching the host's ordinary
chat defaults. A fixed sampler does not establish compatibility across models
or hardware, and failed samples must remain recorded.
The first case applies `frontend-design` to a small HTML page and verifies
specified semantic requirements; it is not a subjective visual quality score.
The second asks the model to run the actual `skill-creator` helper with `--help`
and report its interface. The managed probe above separately proves the helper's
artifact-producing path and arithmetic. The model probe checks successful tool
events and the final response and preserves all SSE frames. It shares the chat
host used by TensorAgent, but does not launch or claim validation of the native
GUI or untested devices. Use `--case instructions` or `--case script` to isolate a
case, always starting with a fresh library when testing first-time installation.

To exercise TensorAgent's actual `AgentAppHost` instead of the standalone server,
use the existing launcher with a fresh data root:

```sh
dotnet build eng/validation/TensorAgentHost/TensorAgentHost.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet eng/validation/TensorAgentHost/bin/Release/net10.0/TensorAgentHost.dll \
  --root artifacts/find-skills/tensoragent-live \
  --skills /absolute/repo/TensorAgent/skills/find-skills \
  --weights /absolute/path/model.gguf --backend ggml_metal \
  --network --port 5050 --context 32768 --max-tokens 2200 \
  --stop-file artifacts/find-skills/tensoragent-stop
```

Ensure the launcher's output directory can resolve the same validated native
backend library as the desktop server. In another terminal run:

```sh
python3 eng/validation/FindSkillsProbe/model_probe.py \
  --connection-file artifacts/find-skills/tensoragent-live/connection.json \
  --model /absolute/path/model.gguf --backend ggml_metal \
  --out artifacts/find-skills/tensoragent-model-report.json
```

The connection file supplies the local loopback authentication cookie; the probe
does not copy it into its report. Create the specified stop file to shut down the
launcher after validation. This covers the real desktop app host and shell
backend; it still does not exercise the native graphical UI or mobile runtime.

The scripts deliberately make no universal compatibility or performance claim:
marketplace skills may require external CLIs, credentials, browser tools, or
other capabilities the selected host does not provide. Cache timings are
single-process measurements; live network and model latencies are individual
observations, not a throughput/load benchmark.
