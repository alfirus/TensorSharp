// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_internal.h"
#include "glm_causal_route_neighbor_owner.h"

namespace tsg {
// C++-only additive wrapper. The existing ubiquitous descriptor/header and all
// CUDA translation units retain their original interface and compile commands.
int host_moe_cached_experts_with_neighbor(const HostMoeSegment&, const float*,
    const std::int32_t*, const float*, float*, const char*, ggml_backend_t,
    const HostMoePrefetchHooks*, tsg_glm_neighbor::Owner*);
bool host_moe_causal_configure(tsg_glm_neighbor::Owner&, const void*, std::uint64_t, int, std::size_t,
    const tsg_glm_neighbor::Config&, const tsg_glm_neighbor::Source*, ggml_backend_t) noexcept;
bool host_moe_causal_close(tsg_glm_neighbor::Owner&) noexcept;
bool host_moe_causal_begin(tsg_glm_neighbor::Owner&, bool scalar, std::int64_t head = 0,
    int rows = 1, std::int64_t sequence = 0) noexcept;
bool host_moe_causal_commit(tsg_glm_neighbor::Owner&, int layer) noexcept;
bool host_moe_causal_finish(tsg_glm_neighbor::Owner&) noexcept;
void host_moe_causal_abort(tsg_glm_neighbor::Owner&) noexcept;
bool host_moe_causal_reset(tsg_glm_neighbor::Owner&, bool recover) noexcept;
bool host_moe_causal_info(tsg_glm_neighbor::Owner&, std::int64_t*, int) noexcept;
bool host_moe_causal_model_release_ready() noexcept;
#if defined(TSG_GGML_TEST_HOOKS)
bool host_moe_causal_test_configure(tsg_glm_neighbor::Owner&, const void*, std::uint64_t, std::size_t,
    const tsg_glm_neighbor::Config&, const tsg_glm_neighbor::Source*, ggml_backend_t) noexcept;
#endif
}
