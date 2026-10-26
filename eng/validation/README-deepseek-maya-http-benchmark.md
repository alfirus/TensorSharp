# TensorSharp / Maya HTTP measurements

`deepseek-maya-http-benchmark.py` uses Python's standard library to drive an
already running OpenAI-compatible server. It records full request bodies,
answer/reasoning text, usage, finish reasons, every SSE line/event with its
arrival time, TTFT, request wall time and `(completion_tokens - 1) /
(last_output_frame - first_output_frame)`. It never substitutes SSE frame counts
for tokens. A single batched frame has no measurable decode window.

Run it from a TensorSharp checkout on each VM. Generated evidence must stay
under that checkout's ignored `artifacts/` or `docs/validation/` directory:

```sh
python3 eng/validation/deepseek-maya-http-benchmark.py \
  --url http://127.0.0.1:8000 --model SERVED_ALIAS --engine tensorsharp-vm1 \
  --suite all --warmups 1 --repeats 3 --max-tokens 128 \
  --extra-body-file artifacts/deepseek-maya/request-options.json \
  --provenance artifacts/deepseek-maya/tensorsharp-vm1-provenance.json \
  --output artifacts/deepseek-maya/tensorsharp-vm1-http.json
```

Set explicit, equivalent thinking and sampling options for each engine. The
extra body is preserved verbatim; the tool cannot prove that an engine honored
an option. The provenance JSON should record checkpoint revision and all shard
SHA-256 values, engine/dependency revisions, launch command/environment,
container RAM/CPU allowance, GPU topology, participating nodes, and speculation
configuration. Server logs and resource telemetry must independently demonstrate
GPU/node engagement.

For plain text on the experimental DeepSeek two-node pipeline, start the host
with `--no-prefix-cache --no-continuous-batching --no-webui --no-skills
--no-multi-agent`. Disable code execution and omit client tools. Skills and
delegation have independent switches; a tool catalog changes the prompt and
invalidates comparison to a raw text-only case. Set `"multi_agent": false` in
the request options and omit `tool_choice` to exercise unconstrained compact
greedy transport. Explicit `tool_choice: "none"` installs a no-tools grammar
and uses the full-logit fallback. Verify prompt-token counts and server-side
transport diagnostics rather than assuming the request took the fast path.

`--suite smoke` checks arithmetic, exact JSON and extraction through a clean
`stop` finish; `--suite throughput` uses short and longer prompts without a
semantic oracle. `--suite all` runs both. These are bounded probes, not a broad
quality qualification. Warmups remain in the report but are excluded from
measured summaries. Repeated requests have identical bodies, allowing prompt
reuse; cached-token counters are retained when available. Compare the same
request order and cache policy. `--concurrency 2` starts two requests together
per warmup/measured batch.

For another workload, supply `--cases path.json` with an array like:

```json
[
  {
    "name": "arithmetic",
    "messages": [{"role": "user", "content": "What is 123+456? Reply with only the integer."}],
    "max_tokens": 32,
    "expected_text": "579"
  },
  {
    "name": "json",
    "messages": [{"role": "user", "content": "Return exactly {\"ready\":true} without markdown."}],
    "max_tokens": 32,
    "expected_json": {"ready": true}
  }
]
```

Each case can also contain `extra_body`. It cannot override the model,
messages, stream settings or number of choices. Cases without an expectation
are `not_evaluated`, never quality passes. Any transport, protocol, usage or
correctness failure returns exit code 1. The report is checkpointed after each
batch, retains failures and cannot overwrite an existing evidence file.
`--timeout` is a whole-request deadline.

Add `--compare-with artifacts/deepseek-maya/maya-vm1-http.json` to save ratios
against an existing report. Comparisons flag mismatched case/request identities,
request order/count, prompt/output token counts, finish reasons or failed checks.
Model aliases are excluded from request identity. Engine-specific option
spellings may legitimately differ; review those differences explicitly. Ratios
never set performance or quality qualification. Client SSE timing includes
network/buffering effects; inspect the retained server timings too.

Tests use real loopback delivery, including a flushed short frame and a
trickling timeout, plus token accounting and comparison failure cases:

```sh
python3 -m unittest discover -s eng/validation/tests \
  -p test_deepseek_maya_http_benchmark.py -v
```
