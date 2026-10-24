// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Pure raw-row bounds/failure contract. No ggml/CUDA/model dependencies.
#include "../glm_host_token_embedding.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

namespace {
int checks = 0;
void require(bool ok, const char* message) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
template<class F> void rejects(F f, const char* message) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}
}

int main() {
    using namespace tsg_glm;
    const auto copy = [](void* d, const void* s, std::size_t n) { std::memcpy(d, s, n); return true; };
    const auto layout = checked_host_token_embedding_layout(17 * 29, 29, 256, 17, 12);
    std::vector<std::uint8_t> source(layout.table_bytes);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::uint8_t>((i * 37 + i / 29) & 255);
    std::vector<std::uint8_t> output(host_token_embedding_max_rows * layout.row_bytes + 2, 0xCD);
    const auto source_before = source;
    std::size_t copied = 999;
    for (std::size_t count : {std::size_t(1), std::size_t(9), std::size_t(512)}) {
        std::vector<std::int32_t> ids(count);
        for (std::size_t i = 0; i < count; ++i) ids[i] = static_cast<std::int32_t>((i * 7) % layout.vocabulary);
        ids.front() = 0; if (count > 1) ids[1] = 16;
        if (count > 2) ids.back() = ids[1];
        std::fill(output.begin(), output.end(), std::uint8_t{0xCD});
        require(pack_host_token_embedding_rows(layout, source.data(), source.size(), ids.data(), ids.size(),
            output.data() + 1, output.size() - 2, copy, &copied), "valid raw pack");
        require(copied == count * layout.row_bytes, "only complete bytes published");
        for (std::size_t i = 0; i < count; ++i)
            require(std::memcmp(output.data() + 1 + i * layout.row_bytes,
                source.data() + ids[i] * layout.row_bytes, layout.row_bytes) == 0, "each packed row exact including duplicate/last/first");
        require(output.front() == 0xCD && output[copied + 1] == 0xCD, "destination guard bytes");
    }
    require(source == source_before, "immutable source unchanged");
    std::vector<std::int32_t> whole(1872, 0);
    require(validate_host_token_embedding_tokens(whole.data(), whole.size(), 17), "whole layer-major submission may exceed512");
    whole.back() = 17;
    require(!validate_host_token_embedding_tokens(whole.data(), whole.size(), 17), "invalid tail of whole submission rejected");
    auto refuse = [&](const host_token_embedding_layout& l, const std::uint8_t* s, std::size_t extent,
        const std::int32_t* ids, std::size_t count, std::uint8_t* d, std::size_t capacity) {
        int calls = 0; copied = 777;
        const bool ok = pack_host_token_embedding_rows(l, s, extent, ids, count, d, capacity,
            [&](void* to, const void* from, std::size_t n) { ++calls; return copy(to, from, n); }, &copied);
        require(!ok && calls == 0 && copied == 0, "refusal precedes all source copies/publication");
    };
    std::int32_t ids[] = {0,16,0,16,0,16,0,16,0};
    auto badids = std::vector<std::int32_t>(std::begin(ids),std::end(ids));
    badids.back() = -1; refuse(layout,source.data(),source.size(),badids.data(),9,output.data(),output.size());
    badids.back() = 17; refuse(layout,source.data(),source.size(),badids.data(),9,output.data(),output.size());
    refuse(layout,source.data(),source.size(),nullptr,9,output.data(),output.size());
    refuse(layout,nullptr,source.size(),ids,9,output.data(),output.size());
    refuse(layout,source.data(),source.size(),ids,0,output.data(),output.size());
    refuse(layout,source.data(),source.size(),ids,513,output.data(),output.size());
    refuse(layout,source.data(),source.size()-1,ids,9,output.data(),output.size());
    refuse(layout,source.data(),source.size(),ids,9,output.data(),9*29-1);
    refuse(layout,source.data(),source.size(),ids,9,nullptr,output.size());
    refuse(layout,source.data(),source.size(),ids,9,source.data()+1,source.size()-1);
    refuse(layout,source.data(),source.size(),ids,9,source.data(),source.size());
    auto broken = layout; ++broken.table_bytes;
    refuse(broken,source.data(),source.size(),ids,9,output.data(),output.size());
    broken=layout; broken.row_bytes=(std::numeric_limits<std::size_t>::max)();
    refuse(broken,source.data(),source.size(),ids,9,output.data(),output.size());
    const auto invalid_address = reinterpret_cast<const std::uint8_t*>((std::numeric_limits<std::uintptr_t>::max)()-7);
    refuse(layout,invalid_address,source.size(),ids,9,output.data(),output.size());
    refuse(layout,source.data(),source.size(),ids,9,const_cast<std::uint8_t*>(invalid_address),output.size());
    require(!validate_host_token_embedding_tokens(ids,(std::numeric_limits<std::size_t>::max)(),17),"token span arithmetic overflow");
    rejects([&]{checked_host_token_embedding_layout(492,29,256,17,12);},"table extent mismatch");
    rejects([&]{checked_host_token_embedding_layout(0,0,256,17,12);},"zero row geometry");
    rejects([&]{checked_host_token_embedding_layout(493,29,0,17,12);},"zero hidden geometry");
    rejects([&]{checked_host_token_embedding_layout(493,29,256,17,-1);},"invalid type");
    rejects([&]{checked_host_token_embedding_layout(0,(std::numeric_limits<std::size_t>::max)(),256,17,12);},"table arithmetic overflow");
    for (int fail_at=0;fail_at<9;++fail_at) {
        int calls=0; copied=888; bool published=false;
        std::fill(output.begin(),output.end(),std::uint8_t{0xCD});
        const bool ok=pack_host_token_embedding_rows(layout,source.data(),source.size(),ids,9,output.data(),output.size(),
            [&](void* d,const void* s,std::size_t n) {
                if (calls++==fail_at) { std::memcpy(d,s,n/2); return false; }
                std::memcpy(d,s,n);return true;
            },&copied);
        if(ok)published=true;
        require(!published && copied==0 && calls==fail_at+1,"source fault never publishes partial destination");
        require(output[fail_at*29]==source[ids[fail_at]*29],"injected partial write is observable and must be discarded");
    }
    copied=777;
    require(!pack_host_token_embedding_rows(layout,source.data(),source.size(),ids,9,output.data(),output.size(),
        [](void*,const void*,std::size_t)->bool {throw std::runtime_error("injected source failure");},&copied)
        && copied==0,"throwing source copier refuses publication");
    const auto profile=checked_host_token_embedding_layout(host_token_embedding_profile_table_bytes,
        host_token_embedding_profile_row_bytes,host_token_embedding_profile_hidden,host_token_embedding_profile_vocabulary,host_token_embedding_profile_type);
    require(profile.table_bytes==356843520 && profile.row_bytes*512==1179648,"exact production profile without allocating table");
    std::printf("host-token-embedding-cpu passed checks=%d skips=0\n",checks);
    return 0;
}
