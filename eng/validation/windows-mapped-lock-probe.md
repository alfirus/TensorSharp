# Read-only mapped-page Windows lock probe

This CPU-only Python 3.10+ tool qualifies bounded `VirtualLock` ownership of a
generated read-only file mapping. It loads no CUDA or TensorSharp code and
accesses no checkpoint. No build is required.

```powershell
python eng/validation/windows-mapped-lock-probe.py --self-test
python eng/validation/windows-mapped-lock-probe.py `
  --fixture artifacts/probe-lock-fixture.bin `
  --output artifacts/probe-lock.json
# Optional separate read-only view / model-trim isolation follow-up:
python eng/validation/windows-mapped-lock-probe.py --alias-trim `
  --fixture artifacts/probe-lock-alias-fixture.bin `
  --output artifacts/probe-lock-alias.json
```

Use new fixture/report paths in an existing ignored artifact directory. The
default generated file is exactly 16MiB; accepted sizes are 8KiB–64MiB in 4KiB
steps. The parent closes its writable fixture handle before spawning one child.
Every tested view uses `GENERIC_READ`, `PAGE_READONLY`, and `FILE_MAP_READ`.
`VirtualQuery` checks the complete view's committed, mapped, read-only protection.
The tool compares every mapped byte and its SHA256 against buffered `ReadFile`
using bounded transient buffers. It never writes through a mapped view.

The child tests unaligned, page-crossing, tail and full-file page envelopes
under its original working-set limits. If a lock refuses, it records the actual
error and any observed partial locks. Only this child may increase its own
working-set minimum and maximum, using the current-process pseudo handle.
Requests are capped at 512MiB; a 16MiB fixture normally requests a 32MiB minimum
and 48MiB maximum. Refusal remains `unavailable` rather than passing validation.
The original limits are restored exactly. The parent's limits are queried
before and after and must remain unchanged. No other process is queried or
modified, and no cache trim or memory-pressure scan runs.

[`QueryWorkingSetEx`](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-queryworkingsetex)
observes every mapped page. Only valid entries are interpreted. Successful
locks must produce exactly the requested rounded page intervals with their
[`Locked` bit](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-psapi_working_set_ex_block)
set; after unlock no locked pages may remain. An injected exception after a
successful lock verifies exceptional cleanup. Every successful lock is
unlocked before unmapping, then mapping/file handles close and child quotas
restore. Unknown partial-lock state or cleanup failure cannot produce a
passing result; remaining ownership is left to child process termination.
The parent waits for child exit before deleting its generated fixture, unless
`--keep-fixture` was requested.

The optional `--alias-trim` follow-up creates two distinct read-only virtual
views of the same fixture in this child. It locks owned-cache-like view A and
leaves model-like view B unlocked. Calling `VirtualUnlock` on B must return the
documented `ERROR_NOT_LOCKED` (158), which the report records explicitly.
A must still show its entire locked envelope and exact read bytes. The probe
then unmaps B and repeats A's locked-page, read-only-protection and byte checks
before finally unlocking and unmapping A. This tests whether a separate view's
trim-like operation interferes with this owned view's virtual lock. It does
not prove physical page identity between a mapping and the buffered file cache,
cross-process behavior, or any production GGUF loader lifetime contract.

Exit **0 / `complete`** means bounded capability and lifetime checks passed.
Exit **77 / `unavailable`** means the platform, quota or lock capability was
unavailable; it is not passing validation. Exit **1 / `failed`** means a byte,
protection, observation, protocol or cleanup check failed. The JSON report
records individual default-quota refusals even when an adjusted-quota trial
later succeeds. The pure `--self-test` checks envelope bounds, valid/locked bit
decoding and quota arithmetic without invoking Windows APIs.

The Windows [`VirtualLock` contract](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtuallock)
permits committed accessible pages and limits lockable pages by the process's
minimum working set, with overhead. A successful
[`SetProcessWorkingSetSize`](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsize)
call alone does not establish residency; this probe separately observes the
actual lock. The fixed probe headroom is not a production admission forecast.

Fixture creation and verification warm file pages. API wall latencies are
descriptive and establish no inference speedup or cold-storage behavior.
`Shared` means a page is shareable; it does not prove the mapping and buffered
file reads use identical physical Cache Manager pages. Exact bytes establish
coherence, not attributable SSD reads or avoided reads. This probe neither
changes the existing buffered-read → pinned-staging → H2D path nor establishes
that retaining mapped pages benefits that path. No GiB-scale retention,
multi-process alias ownership, route policy, physical-page sharing or model
quality/performance qualification is claimed.
