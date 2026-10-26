// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Half-open layer range. A stage requires self-contained shared-cache sources.
// Positive explicit ubatch; no draft or tensor parallelism. Routed-expert CPU
// offload automatically spills only the local layers needed by the VRAM plan.
void * TSGgml_Dsv41LoadStage(const char * gguf_path, int n_gpu, int n_ctx, int n_ubatch,
    int n_threads, const char * backend_name, int first_layer, int end_layer);
int TSGgml_Dsv41StageInfo(void * handle, int * first, int * end, int * hidden, int * hc);

// One microbatch (1..ubatch tokens), at the current cache head. F32 hidden is
// [tokens,hc,hidden], pre is [tokens,hc]. Prefix exports all rows; suffix imports
// them and optionally exports logits [vocab] or [tokens,vocab] when all_logits=1.
// Return 0 on success, negative on error. Reset every stage after any failure;
// existing Dsv4ResetChecked/Truncate/NPast/Free APIs manage local state.
int TSGgml_Dsv41StageForward(void * handle, const int32_t * tokens, int n_tokens,
    const float * hidden_in, const float * pre_in, float * hidden_out, float * pre_out,
    float * logits_out, int all_logits);

#ifdef __cplusplus
}
#endif
