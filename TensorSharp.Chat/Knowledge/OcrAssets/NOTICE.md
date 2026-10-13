# App-managed PDF OCR dependencies

TensorSharp downloads the original upstream/community package archives listed in
`manifest.json` into an application-private cache. It verifies each byte count and
SHA-256 before running the pinned installer. Installation uses an explicit local
package list without dependency solving, shell activation, link scripts, or a
system package manager. No PDF content is sent to these download sources.

The dependencies retain their original licenses; TensorSharp does not relicense
them. `package-notices.json` records each exact archive, its declared license,
available upstream license texts, source recipe, upstream source home, and
conda-forge feedstock revision. Some metapackages or runtime packages do not
include an `info/licenses` text; their original metadata and source recipe are
retained instead. The original package cache and installed package files preserve
upstream notices as supplied.

- **Poppler 26.09.0:** GPL-2.0-or-later, as declared by the
  [conda-forge package](https://anaconda.org/conda-forge/poppler).
  [Upstream source](https://poppler.freedesktop.org/) and the exact build recipe
  URLs/revisions appear in the package records.
- **Tesseract 5.5.3:** Apache-2.0.
  [Source and license](https://github.com/tesseract-ocr/tesseract/tree/5.5.3).
- **Micromamba 2.9.0-0:** BSD-3-Clause project license.
  [Source](https://github.com/mamba-org/mamba/tree/2.9.0) and
  [release binaries](https://github.com/mamba-org/micromamba-releases/releases/tag/2.9.0-0).
  The self-contained binary also contains third-party dependencies under their
  original licenses; its upstream package metadata is retained in the notices.
- **English and Chinese OCR data:** Apache-2.0, from official
  [`tessdata_best` commit e12c65a915945e4c28e237a9b52bc4a8f39a0cec](https://github.com/tesseract-ocr/tessdata_best/tree/e12c65a915945e4c28e237a9b52bc4a8f39a0cec).
- **Native libraries, fonts, and Windows runtime libraries:** individually
  identified in `package-notices.json`, including GPL/LGPL/MPL/BSD/MIT, font, and
  Microsoft runtime license declarations where applicable.

This manifest describes downloads from the original package URLs; it does not
publish a new combined binary distribution. Repackaged/offline distributions
must retain the original notices and applicable source/redistribution materials.
The maintenance instructions in `eng/ocr/README.md` explain how to regenerate and
audit the lock without replacing pinned packages silently.
