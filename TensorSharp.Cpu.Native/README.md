# TensorSharp-owned CPU math prototype

This experimental library has no ggml dependency. Its Q8-weight/Q16-activation
entry point performs one bounded CPU tile under the caller's worker pool. It
allocates no memory and owns no model cache, worker threads, or device state.
The Q16 projection and Float32 attention entries have explicit opt-in
EmbeddingGemma 2 CPU graph bridges; actual model/HTTP gates decide whether
either can be adopted. Portable managed builds require no native compiler.

Build it separately:

```sh
cmake -S TensorSharp.Cpu.Native -B artifacts/cpu-native/linux-x64 -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/cpu-native/linux-x64 -j 2
```

The library exports ABI version 1, a runtime availability check, and bounded
packing, tile and group functions. GCC/Clang x86 builds check AVX512F/BW/VNNI,
AVX2, FMA, SSSE3, and operating
system register support before executing the specialized function. Other
architectures compile an unavailable stub. No caller should treat unavailable
math as passing validation.

The availability entry is compiled without global AVX switches. It uses the
[GCC runtime feature builtins](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html);
GCC's [CPU detector](https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-12/gcc/common/config/i386/cpuinfo.h)
checks OSXSAVE and the required XGETBV register-state mask before advertising
AVX512 features. Specialized instructions are confined to target-attributed
math functions.

Weights retain their signed int8 values, packed as 16 columns with four
dimensions per VNNI lane. For each signed int16 activation `q`, the exact
integer block dot uses `low = q & 255`, `high = (q >> 8) + 128`, then
`dpbusd(low, w) + 256 * dpbusd(high, w) - 32768 * sum(w)`. Non-saturating
integer accumulation stays within int32. Float32 block scales and explicit FMA
preserve the existing managed kernel's accumulation order. The caller keeps
the original Float32 inputs for subnormal-scale fallback.

Packed inputs arrange four low bytes followed by four biased high bytes.
One AVX2 pass creates that representation per activation row. A group entry
handles at most 24 rows by 64 columns per call, using eight-row VNNI tiles
under the existing managed worker pool.

The isolated `eng/validation/EmbeddingGemma2VnniProbe` compares this library
with the existing managed kernel at identical shapes, activation quantization,
and worker counts. It checks scalar integer references, signed endpoints,
tails, output stride canaries, argument bounds, and input/call layout variants.
Kernel timing gains alone do not establish model quality or HTTP performance.
Generated evidence belongs in ignored `docs/validation/` or `artifacts/`.

The current prototype has passed its independent arithmetic microgate on the
Xeon Gold 6342. Its measured gains at representative model shapes were modest;
it has not met the projection adoption requirement. The optional projection
bridge retains the same quantization and subnormal Float32 row correction.
Actual combined projection/attention validation passed 14 production bridge
cases, 10 independent Q16 scalar/double cases and all 9 NumPy model goldens
at the unchanged `.99999/.0005` bounds. CPU HTTP performance still met the 1.05
latency bound in only 1/7 cases, so the projection bridge remains opt-in.
Keep the portable managed route and all existing default choices.

To inspect the actual managed model before further kernel work, set
`TENSORSHARP_EMBEDDING_CPU_PROFILE=1` before loading an EmbeddingGemma 2 CPU
encoder. It emits projection names, original weight types, effective math,
dimensions, call counts and elapsed times once on disposal. Summarize its log
with `eng/validation/embeddinggemma2-profile.py`. These diagnostic times include
warmup and worker scheduling, and the unmeasured remainder cannot be assigned
to a particular operation without further profiling.

The profile also includes disjoint nonprojection phases and separate input
preparation/token-row-export records outside compute. Derived parent totals
explicitly overlap their leaf records; do not add them to the disjoint sum.
Actual Float32/Q16 model diagnostics identified score/softmax/value scheduling
and layout copying as larger targets than PLE.

`embedding_attention_avx512.cpp` is a separate experimental Float32 score/value
prototype. Its eight-query tiles reuse key/value loads while retaining each
output lane's original FMA order. It owns no buffers, threads or model state.
The linked-source `EmbeddingGemma2AttentionProbe` checks independent scalar bit
oracles, double-product error bounds, tail and stride canaries, rejected argument
bounds and matching-shape kernel timings. The opt-in graph bridge leaves
softmax and sliding-window masking managed. Model quality and HTTP performance
require separate validation; kernel gains establish neither.

Set `TENSORSHARP_EMBEDDING_NATIVE_ATTENTION=1` to request the attention bridge.
Use `TENSORSHARP_CPU_NATIVE_LIBRARY=/absolute/path/libTensorSharpCpu.so` for an
explicit independently built library, or put the library beside the app. The
bridge checks ABI 1 and both native/.NET AVX512 capability guards before use.
An unavailable or incompatible library falls back to existing managed math;
that fallback cannot count as a native validation pass. `TS_CPU_DISABLE_AVX512=1`
also prevents the native route. A single buffered `EMBEDDING_CPU_MATH` disposal
record includes the actual math, successful tile/row counts, path, hash and
fallback reason. The CPU backend still owns a managed graph and worker pool;
its opted-in native arithmetic is explicitly distinguished from pure managed
execution and from GGML inference.

The attention-only actual model gate passed all eight exact independent NumPy
cases and the 8192-token case at the original `.99999/.0005` thresholds, with
zero fallback. Actual CPU HTTP performance met the 1.05 latency bound in only
one of seven cases, so attention remains opt-in. The unchanged llama reference's
strict batch/duplicate checks also failed; all timings were retained using an
explicit reference-only continuation mode and the overall strict result stayed
failed. Generated proof and limitations are in
`docs/validation/embeddinggemma2/owned-attention-experiment/`.

The combined VNNI/attention proof is separate in
`docs/validation/embeddinggemma2/owned-q16-attention-experiment/`.
The 9 model cases recorded zero fallback; maximum absolute error was 2.10e-5.
Actual Host disposal recorded 337,440 attention tiles and 37,920 projection
calls/5,122,560 bounded groups, using the same pinned native library. Sensitive
Float32 projections are intentional and separately named. The candidate's
strict HTTP correctness passed; the reference's strict consistency failures
remained failed, with all 7 diagnostic timings retained. These are measured
limitations, not a default adoption or CPU parity claim.

Set both `TENSORSHARP_EMBEDDING_Q16_AVX512=1` and
`TENSORSHARP_EMBEDDING_NATIVE_Q16=1` to request the VNNI projection bridge.
It checks ABI1, native VNNI/OS support and the managed AVX512/disable guard
before preparing packed int8 weights. The encoder owns the library lease and
frees it after worker jobs finish; matrices only borrow the lease. Missing,
incompatible or unsupported native math preserves the existing managed Q16
fallback. PLE, raw input gates and output projections intentionally retain
Float32 arithmetic. Actual projection calls, packed rows, bounded groups,
library hash, managed Q16 calls and subnormal Float32 fallback rows are recorded
separately. Native flags do not select a GGML inference backend. Neither loader
policy checks nor arithmetic microgates establish a whole-model or HTTP pass.

To copy/publish/package a built library, pass
`-p:TensorSharpCpuNativeLibrary=/absolute/path/libTensorSharpCpu.so` and
`-p:TensorSharpCpuNativeRuntimeIdentifier=linux-x64` to the consuming .NET build.
The Models project imports `Packaging.targets`: it verifies the explicit path,
copies the library beside the app and packages it under the specified runtime's
native assets. No library is built/downloaded or enabled automatically.
