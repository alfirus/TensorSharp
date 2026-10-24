// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../glm_layer_major_graph_cap.h"
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

using Policy = tsg_glm::layer_major_graph_cap;
static int checks = 0;
static void require(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
int main() {
    try {
        require(Policy::valid_limit(1,512*Policy::mib) && Policy::valid_limit(1,768*Policy::mib), "Explicit supported caps refused");
        for (const auto bytes : {0ULL,1ULL,511ULL<<20,513ULL<<20,767ULL<<20,769ULL<<20,~0ULL})
            require(!Policy::valid_limit(1,bytes), "Unqualified graph cap accepted");
        require(!Policy::valid_limit(0,512*Policy::mib) && !Policy::valid_limit(2,768*Policy::mib), "Unknown profile accepted");
        Policy::shape s{4096,4,154880,45,34,11,64,64,128,4,12288,2048,2048,288,8,2048,512,3,2048,4};
        require(Policy::valid_shape(s), "Target profile shape refused");
        int Policy::shape::* fields[] = {&Policy::shape::hidden,&Policy::shape::hc,&Policy::shape::vocab,
            &Policy::shape::layers,&Policy::shape::recurrent_layers,&Policy::shape::mla_layers,&Policy::shape::heads,
            &Policy::shape::kda_heads,&Policy::shape::kda_dim,&Policy::shape::conv,&Policy::shape::dense_ff,
            &Policy::shape::shared_ff,&Policy::shape::expert_ff,&Policy::shape::experts,&Policy::shape::used,
            &Policy::shape::context_rows,&Policy::shape::ubatch,&Policy::shape::dense_lead,
            &Policy::shape::indexer_top_k,&Policy::shape::indexer_pool};
        for (auto field : fields) { auto wrong=s; ++(wrong.*field); require(!Policy::valid_shape(wrong), "Changed target geometry accepted"); }
        auto small=s; small.dense_ff=4096; small.shared_ff=1024;
        require(Policy::valid_shape(small), "Smaller bounded FFN shape refused");
        require(Policy::valid_call(1872,0,512,false,false,true) && Policy::valid_call(1024,1024,512,false,false,true), "Valid layer-major prompt refused");
        for (int tokens=513; tokens<=2048; ++tokens) {
            const int tail=tokens%512;
            require(Policy::valid_call(tokens,0,512,false,false,true) == (tail == 0 || tail >= 9), "Wide chunk/tail contract differs at boundary");
        }
        require(!Policy::valid_call(512,0,512,false,false,true) && !Policy::valid_call(1872,177,512,false,false,true)
            && !Policy::valid_call(1872,-1,512,false,false,true) && !Policy::valid_call(1872,0,256,false,false,true)
            && !Policy::valid_call(1872,0,512,true,false,true) && !Policy::valid_call(1872,0,512,false,true,true)
            && !Policy::valid_call(1872,0,512,false,false,false), "Unsupported call entered strict admission");
        require(Policy::valid_range(0,1,45,512,0,512,1,0,false,0,false)
            && Policy::valid_range(44,45,45,336,1536,2048,1,0,false,0,false)
            && Policy::valid_range(0,0,1,9,0,256,1,0,false,0,false), "Legitimate one-layer range/input/head refused");
        require(!Policy::valid_range(0,0,45,512,0,512,1,0,false,0,false)
            && !Policy::valid_range(0,2,45,512,0,512,1,0,false,0,false)
            && !Policy::valid_range(0,1,45,8,0,256,1,0,false,0,false)
            && !Policy::valid_range(44,45,45,512,1800,2048,1,0,false,0,false)
            && !Policy::valid_range(0,1,45,512,0,2304,1,0,false,0,false)
            && !Policy::valid_range(0,1,45,512,0,512,512,0,false,0,false)
            && !Policy::valid_range(0,1,45,512,0,512,1,1,false,0,false)
            && !Policy::valid_range(0,1,45,512,0,512,1,0,true,0,false)
            && !Policy::valid_range(0,1,45,512,0,512,1,0,false,1,false)
            && !Policy::valid_range(0,1,45,512,0,512,1,0,false,0,true), "Invalid strict graph range accepted");
        std::printf("PASS layer-major graph-cap profile: %d checks; exact explicit cap/shape/call/range contract; no model/device/allocator-fit claim\n",checks);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"FAIL graph-cap profile: %s\n",error.what()); return 1; }
}
