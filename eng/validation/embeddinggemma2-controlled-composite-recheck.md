# Saved controlled composite checks

The rechecker compares complete video, mixed and all-modality embeddings with
the explicitly hash-bound independent F32 oracle. It executes no fresh model
math. Weights, arithmetic sources, ordered inputs, media, original observations,
and frozen preprocessing identities must match. Rebuilt candidate assemblies
require `--input-proof` containing fresh identical observations. Original
failed cross-engine reports remain unchanged.

Ordinary `--pair` reports compare TensorSharp and llama.cpp as before. For a
TensorSharp candidate-versus-baseline experiment, explicitly add
`--tensorsharp-reference`. Such reports must come from the pair driver's
`--controlled-video-prefix --reference-controlled-video-prefix` mode so both
TensorSharp roles receive identical ordered `Video:` parts. The reference must
be a resolved direct TensorSharp host with stable before/after binary identities,
the canonical clamp environment, and matching frozen preprocessing observations.
Use `--reference-input-proof` if its assemblies differ from the original oracle's
proof; the new observations must still be identical to the original. The
candidate can separately use `--input-proof`. Native-only candidates may share
the same proof on both sides.

The output labels legacy `tensorsharp` as candidate and legacy `llama` as the
frozen TensorSharp baseline. It makes no llama.cpp execution or performance
claim. Passing both ablation outputs does not convert older failed llama.cpp
comparisons into passes. Unexpected reference prefix metadata is rejected
without the explicit ablation option.
