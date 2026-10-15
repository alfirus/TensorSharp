// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "gemma4_vision_stack.h"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
extern "C" {
int TSGgml_Gemma4VisionStackF32PreciseCuda(const tsg::Gemma4VisionStackArgs*);
int TSGgml_IsBackendAvailable(int);
void TSGgml_ClearHostBufferCache();
void TSGgml_SetDeviceCopyBudget(int64_t);
int64_t TSGgml_DeviceCopyCacheResidentBytes();
void TSGgml_Shutdown();
}
int main(int argc,char** argv) {
    const bool cuda=argc==2 && std::strcmp(argv[1],"--require-cuda")==0;
    if (argc>1 && !cuda) return 2;
    if (!TSGgml_IsBackendAvailable(cuda?3:2)) { std::fprintf(stderr,"Required backend unavailable\n"); return cuda?77:1; }
    constexpr float canary=9876.5f;
    std::vector<float> input(768+128,canary),result(768+128,canary),weight(3072*768,.001f),small_weight(768*768,.002f),norm(768,1.0f);
    std::array<tsg::Gemma4VisionStackBlockDesc,16> blocks{};
    auto view=[](void* data,int rows,int columns) { return tsg::TensorView2DDesc{data,rows,columns,columns,1,int64_t(rows)*columns*4}; };
    for (auto& b:blocks) {
        for(int m=0;m<7;++m) b.matrices[m]=view(m<4?small_weight.data():weight.data(),m==4 || m==5?3072:768,m==6?3072:768);
        for(int n=0;n<6;++n) b.norms[n]=view(norm.data(),1,n==1 || n==2?64:768);
        for(int c=0;c<28;c+=2) { b.clamps[c]=-std::numeric_limits<float>::max(); b.clamps[c+1]=std::numeric_limits<float>::max(); }
    }
    int32_t px=0,py=0;
    tsg::Gemma4VisionStackArgs args{sizeof(tsg::Gemma4VisionStackArgs),1,16,sizeof(tsg::Gemma4VisionStackBlockDesc),
        view(input.data()+64,1,768),view(result.data()+64,1,768),blocks.data(),&px,&py,1,1e-6f,100.0f,12,64};
    const auto original_input=input,original_result=result;
    int controls=0;
    auto rejected=[&](const tsg::Gemma4VisionStackArgs* a) {
        ++controls;
        return TSGgml_Gemma4VisionStackF32PreciseCuda(a)==-1 && input==original_input && result==original_result;
    };
    if (!rejected(nullptr)) return 1;
    auto bad=args; bad.struct_bytes--; if(!rejected(&bad)) return 1;
    bad=args; bad.version=2; if(!rejected(&bad)) return 1;
    bad=args; bad.block_struct_bytes--; if(!rejected(&bad)) return 1;
    bad=args; bad.block_count=15; if(!rejected(&bad)) return 1;
    bad=args; bad.blocks=nullptr; if(!rejected(&bad)) return 1;
    bad=args; bad.pos_x=nullptr; if(!rejected(&bad)) return 1;
    bad=args; bad.position_count=0; if(!rejected(&bad)) return 1;
    bad=args; bad.head_dim=72; if(!rejected(&bad)) return 1;
    bad=args; bad.eps=0; if(!rejected(&bad)) return 1;
    bad=args; bad.rope_theta=std::numeric_limits<float>::quiet_NaN(); if(!rejected(&bad)) return 1;
    bad=args; bad.result=bad.input; if(!rejected(&bad)) return 1;
    bad=args; bad.result.data=input.data()+65; if(!rejected(&bad)) return 1;
    bad=args; bad.input.raw_bytes-=4; if(!rejected(&bad)) return 1;
    bad=args; bad.input.stride0++; if(!rejected(&bad)) return 1;
    bad=args; bad.input.data=reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max()-4); if(!rejected(&bad)) return 1;
    bad=args; bad.result.data=small_weight.data(); if(!rejected(&bad)) return 1;
    bad=args; bad.result.data=norm.data(); if(!rejected(&bad)) return 1;
    void* saved_pointer=blocks[15].matrices[6].data;
    blocks[15].matrices[6].data=small_weight.data(); if(!rejected(&args)) return 1; blocks[15].matrices[6].data=saved_pointer;
    auto saved=blocks[15].matrices[6]; blocks[15].matrices[6].dim1=768; if(!rejected(&args)) return 1; blocks[15].matrices[6]=saved;
    saved=blocks[0].norms[1]; blocks[0].norms[1].raw_bytes=4; if(!rejected(&args)) return 1; blocks[0].norms[1]=saved;
    const float bound=blocks[8].clamps[0]; blocks[8].clamps[0]=std::numeric_limits<float>::quiet_NaN(); if(!rejected(&args)) return 1; blocks[8].clamps[0]=bound;
    px=-1; if(!rejected(&args)) return 1; px=0;
    TSGgml_ClearHostBufferCache(); TSGgml_SetDeviceCopyBudget(1);
    const int refused=TSGgml_Gemma4VisionStackF32PreciseCuda(&args);
    const bool ok=refused==0 && input==original_input && result==original_result && TSGgml_DeviceCopyCacheResidentBytes()==0;
    TSGgml_SetDeviceCopyBudget(0); TSGgml_ClearHostBufferCache(); TSGgml_Shutdown();
    if(!ok) return 1;
    std::printf("GEMMA4_VISION_STACK controls=%d unchanged_input_and_output=1 budget_refusal=%d actual_cuda=%d graph_executed=0\n",controls,cuda?1:0,cuda?1:0);
    return 0;
}
