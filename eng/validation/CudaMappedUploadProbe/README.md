# CUDA mapped-weight upload diagnostic

This standalone probe compares exact-byte transfers of one selected model-file
range. It changes no TensorSharp loader and executes no model. Run only on an
otherwise idle CUDA device; its context synchronization would disturb concurrent
inference or benchmarks.

```sh
dotnet build eng/validation/CudaMappedUploadProbe/CudaMappedUploadProbe.csproj -c Release -m:1
dotnet eng/validation/CudaMappedUploadProbe/bin/Release/net10.0/CudaMappedUploadProbe.dll \
  --file /workspace/models/ltx25/LTX-2.5-Distilled-Q4_K_M.gguf \
  --offset 1048576 --bytes 1073741824 --chunk-bytes 8388608 --rounds 3 \
  --output docs/validation/ltx25/references/cuda-mapped-upload.json
```

The three variants use a new read-only mapping for each trial:

- `pageable_mmap`: the current direct loader's synchronous mapped H2D upload.
- `pinned_mmap`: copy mapped bytes into two bounded pinned chunks, then async H2D.
- `pinned_pread`: positional file reads into the same pinned chunks, then async H2D.

Orders alternate across rounds. A positional SHA256 read verifies the selected
file range before CUDA setup; a disposable mapping then warms and independently
verifies those bytes, including on filesystems whose positional reads bypass the
mmap page cache. That warmup view is closed before trials; all fresh views use
one held file mapping so filesystems that invalidate cache on file open are not
reopened between trials. Linux `mincore` verifies at least99.9% file-cache
residency before every trial; it does not populate the fresh mapping's PTEs.
Buffers/events/device memory are reused, with exactly two pinned chunks. Events
prevent DMA source reuse until copies complete. Every timed transfer finishes
on the device, then a full D2H SHA256 verifies every byte outside the timing.

Reports include file/range/chunk sizes, pinned bytes, actual ordering, total
transfer time, CPU read/copy time, event waits, submit time and exact-byte hashes.
Setup, cache warming and verification are reported separately and excluded from
transfer throughput. These measurements cannot establish full-model startup or
inference throughput; any product loader change needs matching full-process
model comparisons and numerical validation.

To investigate fresh path mappings whose file pages are absent according to
Linux `mincore`, add `--residency fresh_nonresident`. This lane omits mapped
warmup, opens a new MMF from the path for every trial, and requires at most0.1%
residency before **every** variant. It still computes a positional source SHA256
before CUDA setup and verifies the entire GPU result after each transfer. An
unmatched starting state writes an `unavailable` report with attempted order and
observed residency, then aborts. The tool never evicts caches or changes the model
file. Fresh mmap/file-cache residency cannot establish cold disk or NFS server
cache state; report it as the measured fresh-mapping condition only.

The adjacent [`nfs_mmap_open_probe.py`](../nfs_mmap_open_probe.py) is a CPU-only,
read-only Linux diagnostic for extra descriptor opens/closes and fresh mmap
views. It uses no CUDA context and makes no cache-eviction requests:

```sh
python3 eng/validation/nfs_mmap_open_probe.py /path/to/model.gguf \
  --bytes 33554432 --rounds 3 --output docs/validation/nfs-mmap-open.json
```

The API distinction follows the [NVIDIA synchronization documentation](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html):
pageable H2D can synchronize and stage internally, while pinned async H2D permits
explicit bounded staging. The existing TensorSharp-owned `CudaWeightArena`
already uses positional reads and pinned-buffer events for whole-model loading.
