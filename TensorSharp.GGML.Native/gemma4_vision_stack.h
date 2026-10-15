// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml_ops_internal.h"

namespace tsg {
// Synchronous, borrowed immutable F32 descriptors. No descriptor, activation or
// graph pointer survives the call. Immutable matrix host addresses remain cache
// keys; owners must invalidate their device bindings before freeing/reusing them.
struct Gemma4VisionStackBlockDesc {
    TensorView2DDesc matrices[7]; // Q,K,V,out,gate,up,down (row-major [output,input])
    TensorView2DDesc norms[6];    // ln1,Qnorm,Knorm,attention post,ln2,FFN post
    float clamps[28];            // seven {input min,max,output min,max} records
};
struct Gemma4VisionStackArgs {
    int struct_bytes, version, block_count, block_struct_bytes;
    TensorView2DDesc input, result;
    const Gemma4VisionStackBlockDesc* blocks;
    const std::int32_t* pos_x;
    const std::int32_t* pos_y;
    int position_count;
    float eps, rope_theta;
    int heads, head_dim;
};
static_assert(sizeof(Gemma4VisionStackBlockDesc)==528, "Vision block ABI mismatch");
static_assert(sizeof(Gemma4VisionStackArgs)==128, "Vision stack ABI mismatch");
// 1=success; 0=pre-compute admission refusal, output untouched; -1=failure.
// NotApplicable never means that a failed compute may be retried silently.
constexpr int GEMMA4_VISION_STACK_SUCCESS=1, GEMMA4_VISION_STACK_NOT_APPLICABLE=0,
    GEMMA4_VISION_STACK_FAILURE=-1;
}
