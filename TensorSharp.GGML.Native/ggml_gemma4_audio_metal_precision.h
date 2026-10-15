// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

// TensorSharp-owned F32 projection kernels interleaved with unchanged ggml
// Metal operations in one command buffer. No input/result host round trips.
// Returns false only when this backend/graph has no applicable owned operation.
bool tsg_gemma4_audio_metal_graph_compute(
    ggml_backend_t backend, ggml_cgraph * graph, ggml_status * status, bool prefer_mps = false);

// These TensorSharp-owned graph nodes are executable only by the helper above.
// All inputs are declared as sources so ggml's allocator tracks their lifetime.
// Unsupported bounded geometry returns nullptr for the ordinary graph fallback.
ggml_tensor * tsg_gemma4_audio_metal_attention(ggml_context * ctx,
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * relative,
    int sequence, int heads, int head, int past, int future, float cap);
ggml_tensor * tsg_gemma4_audio_metal_depthwise(ggml_context * ctx,
    ggml_tensor * input, ggml_tensor * weight, int sequence, int channels, int kernel);
// Small bidirectional vision blocks avoid upstream flash attention's Half Q
// staging. Bounded full-head F32 arithmetic; larger blocks use ordinary ggml.
ggml_tensor * tsg_gemma4_vision_metal_attention(ggml_context * ctx,
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int sequence, int heads, int head);

