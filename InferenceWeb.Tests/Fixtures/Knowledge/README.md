# Original PDF OCR fixtures

These files contain original synthetic content created for TensorSharp tests, with
no real personal/customer data or externally sourced document content. They can be
redistributed under the repository license. The DejaVu/Songti fonts used during
creation are rendered to pixels; font files are not bundled or embedded in the
scanned PDFs. The mixed document's selectable page uses built-in PDF Helvetica.

- `scanned-four-pages.pdf`: four upright raster-only pages, one distinct code and
  small table per page. Page 4 also has a clearly labeled Chinese sentence.
- `mixed-two-pages.pdf`: selectable text on page 1; page 2 contains a selectable
  heading plus two separate raster blocks. The heading must not suppress OCR of
  the image-based body. Reading only the largest image loses part of the page (the code
  and table are in different blocks).
- `manifest.json`: exact file hashes, expected page codes/totals, text-layer and
  raster-block counts, supplementary Chinese phrases, font hashes, and generation
  dependency versions. Do not treat manifest text as OCR output.

Regenerate from the repository root with a Python environment containing Pillow,
ReportLab, and pypdf:

```sh
python3 eng/validation/knowledge/create_pdf_fixtures.py \
  --latin-font /path/to/DejaVuSans.ttf --cjk-font /path/to/Songti.ttc
```

Fonts are explicit inputs. Different fonts or package versions may change generated
bytes; tests identify the committed fixtures by hash. The generator uses invariant
PDF metadata and no random content. Generated preview images go under ignored
`artifacts/knowledge/pdf-fixture-preview/`; do not commit them.

`KnowledgePdfFixtureTests` independently checks both PDFs with PdfPig: page count,
manifest hashes, absence/presence of selectable text, and all embedded image
blocks. Render every page with Poppler and visually inspect it after a fixture
change. These clean synthetic pages do not measure skew/noise/handwriting quality,
complex document layout, or general Chinese OCR. Chinese results must be reported
separately from English codes.

`crop-rotation.pdf` is a separate original vector fixture for native rendering
regressions. Its four pages use an offset crop box, rotations 0/90/180/270, four
colored corner regions, upright counter-rotated text, and content outside the crop.
It is not part of the factual OCR manifest. Regenerate byte-for-byte with the
standard library alone:

```sh
python3 eng/validation/knowledge/create_pdf_transform_fixture.py
```

`KnowledgePdfNativeTransformTests` checks exact corner colors and raster dimensions
at 200 DPI and under a smaller pixel budget, and uses actual Apple Vision to read
the rotated visible text while excluding text outside the crop. These native tests
are explicitly skipped on platforms without Apple CoreGraphics/Vision.
