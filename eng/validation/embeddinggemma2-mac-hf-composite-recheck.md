# Saved Mac default-HF regression

This separate saved-report tool derives from unchanged default-HF recheck
d675695a and uses the reviewed Mac reference e1cdfe47. It executes no encoder,
native graph, video decode, or service timing. The original FFmpeg reference and
recheck are unchanged.

The trusted original oracle must be complete and successful: six fresh vision
inputs, seven full text cases, 768 finite unit-normalized components per case,
both API comparisons passed, unchanged input guards, and the original 0.9999
cosine / 0.002 component limits plus 2e-6 resized-pixel limit. All **six exact**
arithmetic/reference source identities are required, including both the original
default-HF and Mac-specific reference sources.

Mac video scope is mandatory and cannot be relabeled:

- codec_reference_kind is mac_avfoundation_same_codec_separate_application.
- ffmpeg_equivalence_qualified and independent_codec_implementation are false.
- The original decoder_proof is present in the original oracle's before guard,
  and its exact sampled-RGBA/same-platform observations passed.

The checker calls the unchanged Mac validate_decoder with the original proof.
Every decoder metadata JSON read requires its digest from the original oracle's
before_hashes; every returned input/generated-file digest must equal that
original binding before it is added to the new guard. It never replaces a
decoder digest with a fresh current hash. Original decoder inputs, aliases,
manifest bytes, complete frame bytes, frame counts/PTS/sample indices, assemblies,
codec, and generated artifacts are revalidated. No decoding is repeated.

Exactly **one** current HTTP report is accepted per invocation. Keep separate
CPU7/CPU8/Metal reports in separate rechecks, preserving the original
unique-backend rule rather than counting repeated backends as coverage.
The complete seven ordered inputs, token counts, media and manifest, model and
projector hashes, preprocessing assemblies, every full vector, both APIs and
unchanged numerical limits remain required.

Native-only changes can reuse the original preprocessing observations.
If Models/Runtime change, --input-proof must provide fresh identical tokenizer,
resize, sampling, codec and runtime observations bound to the new HTTP report.
The same complete runtime guard covering the current HTTP binary list must also
contain the observed managed and native codec files in the current host. Those
files are bound to the original proof's digests, even though HTTP's short binary
list omits codecs. A changed codec cannot reuse the old proof.

If the original HTTP includes a base64 MRL summary, the candidate must retain
the same dimension, finite cosine and original 0.99999 gate. This is a rechecked
saved HTTP summary, not fresh independent prefix math; complete saved full-vector
comparison remains separate. No MRL coverage is claimed when both reports omit
that summary.

    python3 eng/validation/embeddinggemma2-mac-hf-composite-recheck.py \
      --oracle /absolute/fresh-math-v1/default-hf-mac-complete.json \
      --oracle-sha256 TRUSTED_ORIGINAL_SHA256 \
      --model /absolute/embeddinggemma-2-Q8_0.gguf \
      --projector /absolute/embeddinggemma-2-mmproj-Q8_0.gguf \
      --http /absolute/one-current/http-media.json \
      --runtime-guard /absolute/current/complete-runtime-guard.json \
      --output /absolute/fresh/ignored/mac-recheck.json

The current reference oracle is SHA
15940e37fa2cc4e63ecc78106695197a37c4bc2e45392d1298396bc0a86c0402.
Pass the independently verified digest of the actual retained report; no path
or digest in this README supplies trust by itself.

Outputs are fresh and prior evidence is preserved. Same Apple codec through a
separate application is not a second codec implementation, restored FFmpeg
equality, or cross-platform pixel equivalence. Original observed-tokenizer,
float-bicubic, reused HF audio rows, opaque-media and OS shared-cache limits
remain. Passing these saved synthetic seven cases establishes no retrieval
leaderboard result, new learned-reference computation, service performance or
production promotion.
