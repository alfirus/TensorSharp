# Mac platform video decode observation

This bounded validation tool invokes `AVAssetReaderTrackOutput` directly, with
no OpenCV or TensorSharp imports. It requests packed32BGRA, decodes every frame
sequentially from the beginning, and copies opaque pixels to RGBA by channel
reordering only. It performs no color adjustment, resize, rotation, interpolation
or learned forward. Track transforms must be identity; dimensions are bounded
to1024 and frame count to64. Source row padding is excluded from the saved pixels.

The comparison target is a **fresh**, assembly-bound
`EmbeddingGemma2CompositeInputProbe --http` observation. Compare complete RGBA
bytes at the observed sample indices, dimensions and timestamps. Bind the video,
source, executable, actual compiler/SDK interfaces, OS build, native OpenCV codec,
input proof and generated files before/after in the execution wrapper. Existing
files are never replaced; failed/partial decode outputs must remain preserved.

```sh
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-deprecated-declarations \
  eng/validation/embeddinggemma2-avfoundation-decode-probe.mm \
  -framework AVFoundation -framework CoreMedia -framework CoreVideo \
  -framework Foundation -o /absolute/ignored/avfoundation-decode
/absolute/ignored/avfoundation-decode --host-controls
/absolute/ignored/avfoundation-decode \
  --video /absolute/fixture.mp4 --output /absolute/ignored/fresh-decode
```

The deprecated synchronous asset metadata APIs keep this one-file harness
simple; they select no different decoder/color policy. Six host controls check
padded channel copying and shape/span/alpha refusals; they decode no video.
Runtime failure returns1 with `completed=false` and `passed=false` in a preserved
report when an output directory has been reserved. CLI/output setup failures
return2. The tool supports macOS only; unavailable platforms are untested.

A byte match would verify a separate application path through the **same Apple
codec**, not an independent codec implementation. It cannot restore the
historical failed FFmpeg equality gate, prove cross-platform pixel identity or
qualify HF/video model math by itself. Subsequent Mac reference validation must
keep those limits explicit and retain the original numerical gates. SDK stubs
bind link interfaces, not implementation bytes in the Apple dyld shared cache.
