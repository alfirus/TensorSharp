# App-managed PDF OCR packages

TensorAgent prepares its private Poppler/Tesseract installation when a PDF needs
OCR. End users do not run these maintenance tools or install a system OCR
package manager. The product downloads only dependency files; documents remain
on the device. This directory provides reproducible dependency locks, cache
seeding, source/license audits, and actual-platform validation.

## Locked runtime

[`manifest.json`](../../TensorSharp.Chat/Knowledge/OcrAssets/manifest.json) pins
the direct static micromamba bootstrap, every native package in the conda-forge
dependency closure, and five official `tessdata_best` packs. Runtime identifiers
are `win-x64`, `linux-x64`, and `linux-arm64`. Windows 11 ARM64 may use the x64
tools through OS emulation; this is not a native ARM64 OCR build or an assertion
of tested Windows coverage. Linux packages require glibc, not musl/Alpine.

The versioned packages include fonts, Poppler CMaps, codecs, native libraries,
and Windows runtime DLLs. Current native-package downloads are approximately
235 MB on Windows and 243–245 MB on Linux, plus the bootstrap and roughly 68 MB
of best-profile language files. The private package cache and installed runtime
require more disk space than the download. Inspect the exact byte counts in the
manifest rather than relying on these rounded sizes.

The default OCR languages are English and simplified Chinese (`eng+chi_sim`).
Traditional Chinese and both Chinese vertical companion models are also
installed automatically. Set `TS_KNOWLEDGE_OCR_LANGUAGES=eng+chi_tra` to choose
traditional Chinese; no additional package installation is required. Combining
both Chinese recognition models can change the written script of output.

The installer verifies SHA-256 and size before executing the bootstrap or
installing a package. It supplies only local package URLs to an offline explicit
installation, disables ambient configuration and package link scripts, and does
not perform a runtime dependency solve. The unique installation prefix is
permanent: moving it after installation would invalidate paths relocated by
conda. Child processes receive private environment paths; host environment and
shell settings remain unchanged.

On actual Linux ARM64, the locked binaries run directly without activation or
`LD_LIBRARY_PATH`; ELF runtime paths resolve non-glibc libraries within the
prefix. The manifest specifies private fontconfig paths. Windows requires
`Library/bin` in the child process's DLL search path; its runtime execution must
still be verified on Windows before claiming platform validation.

## Seed an offline package cache

```sh
python3 eng/ocr/seed_cache.py \
  --output artifacts/ocr-offline-packages --rid linux-arm64
```

Repeat `--rid` or omit it for every runtime. The output layout is
`<sha256>/<fileName>`. Point the provisioner's optional
`bundledPackagesDirectory` at that directory, or place its contents under an
installation's `downloads/` folder. The default provisioner also detects an
`ocr-assets` directory beside the application (`AppContext.BaseDirectory`);
place the same `<sha256>/<fileName>` contents there for an offline app bundle.
Seeding verifies every file and neither
installs nor executes it. Product startup does not invoke this Python tool.

## Refresh a lock deliberately

Only maintainers use a solver. Start with a checksum-verified micromamba release
for the maintainer's host; the runtime bootstrap pins appear in the manifest.
For the current macOS ARM64 maintenance executable, the
[2.9.0-0 release](https://github.com/mamba-org/micromamba-releases/releases/tag/2.9.0-0)
asset `micromamba-osx-arm64` has SHA256
`ec2a072f028e1a7cf20f3e2e74d5a8127cf5a5f27636375b5359811565f4e5be`.

```sh
python3 eng/ocr/generate_manifest.py \
  --micromamba /absolute/path/to/verified/micromamba \
  --evidence artifacts/ocr-refresh --id ocr-YYYY-MM-DD.N
python3 eng/ocr/seed_cache.py --output artifacts/ocr-refresh/downloads
```

The generator pins the requested Poppler/Tesseract versions, resolves only
conda-forge packages for each target, and writes all resulting URLs, checksums,
sizes, builds and licenses. The Linux x64 solve targets glibc 2.17; this is a
dependency constraint, not runtime validation on that oldest OS. Review the
entire manifest diff, package licenses and dependency changes. A refresh is not
approved merely because resolution succeeds.

Download the release `.tar.bz2` metadata archives for the three micromamba
platforms and verify them against the SHA-256 in the saved official release JSON.
They contain the static executable's third-party license notices. Then run:

```sh
python3 eng/ocr/audit_packages.py \
  --downloads artifacts/ocr-refresh/downloads \
  --bootstrap-release-json artifacts/ocr-refresh/micromamba-release.json \
  --bootstrap-archives artifacts/ocr-refresh/bootstrap-archives \
  --output TensorSharp.Chat/Knowledge/OcrAssets/package-notices.json
```

The audit requires `bsdtar` with zstd support and reads metadata from verified
archives without executing package payloads. It records exact source recipes,
feedstock commits, source homes, and available license texts, including the
bootstrap's statically linked dependencies. Preserve
[`NOTICE.md`](../../TensorSharp.Chat/Knowledge/OcrAssets/NOTICE.md), the package
records, and applicable original licenses/source materials when producing an
offline distribution. These records do not relicense dependencies.

## Validate actual first use

On the actual target OS, with .NET 10 available and no system OCR tools:

```sh
python3 eng/ocr/validate_managed.py \
  --dotnet /absolute/path/to/dotnet \
  --output artifacts/ocr-first-use \
  --cache artifacts/ocr-empty-cache \
  --require-no-system-tools --samples 3
```

This requires an empty private cache and exercises actual automatic downloads,
installation, page OCR, crop/rotation, durable indexing, original/citation
integrity and lifecycle. It then starts a new process with a refused HTTP proxy
to validate warm offline reuse. A separate empty cache with the same proxy must
fail and never publish a ready marker. Missing or skipped native scenarios are
not counted as passing. `TENSORSHARP_GGML_NATIVE_SKIP=true` means this validates
CPU OCR and knowledge indexing, not ggml/GPU or model inference.
The validator builds Release by default; `--configuration Debug` explicitly
selects a diagnostic build and is recorded in the validation report.

The reusable `eng/validation/knowledge/pdf_store_probe.py` separately exercises
TensorAgent's actual HTTP upload and reindex flow. Start the
`eng/validation/TensorAgentHost` launcher with a fresh root, no model and no OCR
dependency environment overrides. The observer below runs that HTTP probe and
also requires actual downloading, installing and ready state transitions:

```sh
python3 eng/ocr/validate_managed_http.py \
  --connection artifacts/fresh-host/connection.json \
  --output artifacts/ocr-first-upload
```

Generated reports, raw OCR, screenshots and logs belong under ignored
`artifacts/` or `docs/validation/`. Windows workflows or cross-published binaries
alone do not establish Windows execution coverage.

Primary references: [micromamba installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
[explicit package files](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#explicit-spec-files),
[Poppler package](https://anaconda.org/conda-forge/poppler),
[Tesseract package](https://anaconda.org/conda-forge/tesseract), and
[official OCR data profiles](https://tesseract-ocr.github.io/tessdoc/Data-Files.html).
