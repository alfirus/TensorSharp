# iOS PDF OCR runtime probe

This standalone UIKit app executes `ApplePdfOcr` from the actual
`TensorSharp.Models` iOS assembly. It also exercises the production PDF OCR bridge
and knowledge store, including provider selection with unusable desktop executable
paths. The final shared portable-provider/options sources are compiled into the
probe too, while iOS must select Apple Vision and reject desktop capability. It
loads no language model, uses no remote service, and never
reads TensorAgent's data. Original PDF inputs are linked from the test fixtures.

The Python runner creates a disposable simulator from an installed runtime,
installs only this probe, copies its JSON report and logs to the requested evidence
directory, then terminates the app and deletes the simulator. Requires macOS,
Xcode, a .NET iOS workload, and the selected installed simulator runtime:

```sh
python3 eng/validation/ApplePdfOcrProbe/run_simulator.py \
  --runtime com.apple.CoreSimulator.SimRuntime.iOS-27-0 \
  --output artifacts/knowledge/ios-pdf-ocr-27 --samples 3

python3 eng/validation/ApplePdfOcrProbe/run_simulator.py --no-build \
  --runtime com.apple.CoreSimulator.SimRuntime.iOS-26-5 \
  --output artifacts/knowledge/ios-pdf-ocr-26 --samples 3
```

The default compiler is `~/.dotnet/dotnet`; override `--dotnet` if needed. Its
workload and Xcode SDK must match. The runtime must exist (`xcrun simctl list
runtimes`); unavailable platforms are not passing results. The bundle ID is
`ai.tensorsharp.validation.pdfocr`. The runner does not select or alter a physical
device, signing identity, or provisioning profile.

The report includes scanned English codes/totals, Chinese phrases, both raster
blocks on a mixed page, offset crop boxes at four rotations, up/down raster scale,
cancellation and bounds, durable import and scoped search with page provenance,
and byte-preserved originals. Every failed case remains in the result. Benchmarks
measure complete page rasterization plus Vision OCR after correctness cases have
warmed the frameworks; samples, median, minimum, and maximum are retained. These
small synthetic fixtures do not establish accuracy for noisy handwriting, complex
layouts, or arbitrary languages.

`environment.json` records the bundled product assembly and fixture hashes and selected runtime.
`benchmark-summary.json` aggregates every measured page sample (18 with the default
three samples per page), using the median for p50 and nearest-rank p95. The pages
have different content, so this aggregate is not a repeated single-page workload.
`runner-metrics.json` records the largest sampled host app RSS at one-second
intervals; this excludes separate simulator OS services and can miss shorter peaks.

Simulator execution verifies the real .NET iOS/PInvoke path and the simulator's
CoreGraphics/Vision services. Simulator timings measure the host, not phone
performance. A successful managed or app build alone does not prove physical
device OCR, AOT PInvoke behavior, or memory limits. Device validation requires a
separate signed installation and actual run; do not label it passed from the
simulator report.

The repository's TensorAgent app targets are iOS, Mac Catalyst and Windows; it has
no Android or browser-WASM app head. The browser UI uploads to an HTTP host, where
the selected host provider performs OCR. This probe establishes no Android,
browser-local, Windows, Linux, FreeBSD, or physical-phone OCR capability.

An unsigned physical-device AOT/link check can be built separately, without using a
signing identity or changing provisioning (this cannot be installed as-is):

```sh
DOTNET_ROOT="$HOME/.dotnet" "$HOME/.dotnet/dotnet" build \
  eng/validation/ApplePdfOcrProbe/ApplePdfOcrProbe.csproj \
  -c Release -r ios-arm64 -p:TensorSharpAppleTargets=true \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  -p:EnableCodeSigning=false -p:BuildIpa=false -m:1 -nologo
```
