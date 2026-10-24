# Read-only mapped-file CUDA registration probe

This isolated Windows tool tests whether CUDA can register pages from a
`PAGE_READONLY` / `FILE_MAP_READ` generated file mapping. It changes no model,
TensorSharp loader, native cache, or upstream source. Build and CPU self-test do
not call CUDA; schedule the actual probe on an idle GPU.

```powershell
./eng/validation/build-cuda-readonly-register-probe.ps1
$env:PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64;" + $env:PATH
$probe = "artifacts/perf/glm-memory-tier-20261015/cuda-readonly-register-probe/cuda-readonly-register-probe.exe"
& $probe --self-test --fixture artifacts/probe-cpu-fixture.bin --output artifacts/probe-cpu.json
# Run only after obtaining the shared GPU window:
& $probe --fixture artifacts/probe-cuda-fixture.bin --output artifacts/probe-cuda.json --bytes 16777216 --rounds 4
```

Fixture and output paths must not already exist. The default fixture is less
than 17 MiB; payloads are bounded to 4 KiB–64 MiB and rounds to 1–8. Fixture
creation writes only the new fixture, then closes its writable handle. All
tested views use a read-only file handle and read-only mapping. `VirtualQuery`
checks the complete view protection. CPU tests check mapped bytes against a
buffered `ReadFile` reference and validate page envelopes with both zero and
nonzero registration bases. The CUDA run tests aligned and unaligned payloads,
with 4 KiB aligned registration address/length fully inside the mapping.

The tool queries host-registration support, read-only registration support,
host page table access, pageable access, unified addressing and copy engines.
Absent required capabilities, enumeration/query failure, or a registration
refusal returns **77 / `unavailable`**, which is not a passing validation. Other
transfer, byte-verification or cleanup failures return **1 / `failed`**. Stop
registered-file-page tier design if the required capability is unavailable.
Only full successful CUDA execution returns **0 / `complete`**; CPU self-test
has the separate status `cpu_self_test_complete`.

Trials alternate the order of buffered reference, pageable mapping, and
registered mapping uploads. Every payload byte is compared with the buffered
reference after H2D and D2H. Register/unregister wall latency, H2D submit/wall
time and CUDA event time are recorded separately. These event times are
descriptive for pageable transfers because host staging can delay submission.
Fixture creation, buffered reads, mapping validation and verification warm file
data; this probe establishes no cold SSD or Windows file-cache behavior. It
does not test zero-copy GPU loads or infer full-model performance.

Every registered H2D completes via `cudaStreamSynchronize` before unregister.
Final cleanup fences before unregister, and unregisters before unmapping. A
failed final fence deliberately skips explicit unregister/unmap and records a
failure; process teardown owns remaining resources. A failed unregister keeps
the mapping alive. Cleanup failures can never produce a passing report.

The API contract is documented in the installed CUDA runtime headers around
`cudaHostRegister`, especially `cudaHostRegisterReadOnly` and
`cudaDevAttrHostRegisterReadOnlySupported`. The build receipt records source,
compiler, linked CUDA import library and executable hashes. No `nvcc`, kernels,
or product build are needed.
