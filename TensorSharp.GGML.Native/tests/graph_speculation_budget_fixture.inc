// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Uses the strict physical-owner fixture from graph_budget_test.cpp. These
// CPU checks qualify accounting and snapshot byte copies, not CUDA/MTP logits.
static void speculation_context_budget() {
    Ledger ledger;
    ledger.expected_kind = 7;
    CpuType type(ledger);
    type.max_size = 512;
    for (int mode = 0; mode < 8; ++mode) {
        ledger.capacity = mode == 3 ? 0 : 4096;
        ledger.fail_commit = mode == 5;
        type.actual_extra = mode == 6 ? 1 : 0;
        type.fail_at = mode == 4 ? type.calls + 1 : mode == 7 ? type.calls + 2 : -1;
        if (mode != 0) require(ledger.attach(mode != 1), "speculation context ledger attach failed");
        const int calls = type.calls;
        {
            Graph graph(112); // two extents exceed a single 512-byte chunk
            auto buffer = tsg::graph_budget_alloc_ctx_tensors_from_buft(graph.ctx, &type.type, 1, 7);
            require(bool(buffer) == (mode == 2), "snapshot context ignored exact attachment/quota/allocation/commit failure");
            if (mode <= 1 || mode == 3)
                require(type.calls == calls, "unadmitted snapshot allocated physical storage");
            if (buffer) {
                require(graph.input->buffer->buft == &type.type && graph.output->buffer->buft == &type.type,
                    "snapshot context changed original per-tensor buffer type");
                require(graph.input->buffer->iface.free_buffer == CpuType::free_buffer,
                    "snapshot context changed physical free dispatch");
                require(ledger.committed == ledger.physical && !ledger.detach(),
                    "snapshot context lost exact bytes or callback lifetime");
                tsg::graph_budget_free_buffer(buffer);
            }
        }
        ledger.empty();
        if (mode != 0) require(ledger.detach(), "snapshot context leaked admitted ticket");
    }
    ledger.fail_commit = false; type.actual_extra = 0; type.fail_at = -1;
    ledger.capacity = 4096;
    for (bool attached : {false, true}) {
        if (attached) require(ledger.attach(), "speculation gallocr ledger attach failed");
        const int calls = type.calls;
        auto allocator = tsg::graph_budget_gallocr_new(&type.type, 0, false, 7);
        require(allocator != nullptr, "speculation gallocr construction failed");
        {
            Graph graph(16);
            require(tsg::graph_budget_gallocr_alloc_graph(allocator, graph.graph) == attached,
                "speculation graph escaped mandatory accounting");
            if (!attached) require(type.calls == calls, "speculation gallocr allocated without envelope");
            else require(ledger.last_kind == 7 && ledger.committed > 0 && !ledger.detach(),
                "draft graph did not retain dedicated live owner");
        }
        tsg::graph_budget_gallocr_free(allocator); ledger.empty();
        if (attached) require(ledger.detach(), "speculation graph leaked owner");
    }
    auto backend = ggml_backend_cpu_init();
    require(backend != nullptr, "snapshot CPU backend unavailable");
    require(ledger.attach(), "exact snapshot ledger attach failed");
    {
        Graph snapshot(32);
        auto buffer = tsg::graph_budget_alloc_ctx_tensors(snapshot.ctx, backend, 0, 7);
        require(buffer != nullptr && ledger.last_kind == 7, "snapshot default-type adapter lost dedicated owner");
        float original[32], restored[32];
        for (int i = 0; i < 32; ++i) original[i] = (i - 13) * 0.125f;
        ggml_backend_tensor_set(snapshot.input, original, 0, sizeof(original));
        ggml_backend_tensor_copy(snapshot.input, snapshot.output);
        std::memset(restored, 0, sizeof(restored));
        ggml_backend_tensor_set(snapshot.input, restored, 0, sizeof(restored));
        ggml_backend_tensor_copy(snapshot.output, snapshot.input);
        ggml_backend_tensor_get(snapshot.input, restored, 0, sizeof(restored));
        require(std::memcmp(original, restored, sizeof(original)) == 0,
            "admitted snapshot capture/restore changed exact tensor bytes");
        tsg::graph_budget_free_buffer(buffer);
    }
    ledger.empty(); require(ledger.detach(), "exact snapshot owner did not release");
    ggml_backend_free(backend);
}
