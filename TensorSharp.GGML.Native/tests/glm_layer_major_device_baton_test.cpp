// CUDA component oracle only: no model/capture/performance qualification.
#include "glm_layer_major_device_baton.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Charge=tsg::SharedCacheCharge;
using Baton=tsg_glm::device_baton;
constexpr int hidden=32, channels=4, width=hidden*channels, tokens=19, chunk=8, layers=6;
constexpr std::size_t payload=static_cast<std::size_t>(width)*tokens*sizeof(float);
std::uint64_t checks=0, exact_f32=0, uid=UINT64_C(0x4241544f4e000000);
auto started=std::chrono::steady_clock::now();
void need(bool value,const char* message) { ++checks; if(!value)throw std::runtime_error(message); }
void stage(const char* value) {
    need(std::chrono::steady_clock::now()-started<std::chrono::seconds(90),"component deadline exceeded");
    std::fprintf(stderr,"[GLM-BATON-TEST] stage=%s\n",value); std::fflush(stderr);
}
void exact(const std::vector<float>& a,const std::vector<float>& b,const char* message) {
    need(a.size()==b.size(),"oracle vector size");
    need(std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);})
        &&std::all_of(b.begin(),b.end(),[](float v){return std::isfinite(v);}),"nonfinite F32 oracle");
    need(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,message); exact_f32+=a.size();
}
struct Ledger {
    struct Entry { std::uint64_t token=0; std::size_t bytes=0; int kind=0; bool committed=false,refunded=false; };
    std::array<Entry,64> entries{};
    std::array<std::size_t,8> pending{},committed{};
    std::uint64_t next=0,reserves=0,releases=0,last_token=0,last_released=0,refunds=0;
    int reject_reserve=-1,reject_commit=-1,ack_fault=0; // 1 pre-ledger,2 post-ledger acknowledgment
    bool failed=false; Baton* bank=nullptr;
    Entry* find(std::uint64_t token) noexcept { for(auto& e:entries)if(e.token==token)return &e;return nullptr; }
    std::size_t live() const noexcept { std::size_t n=0;for(const auto& e:entries)n+=e.token!=0;return n; }
    void disarm() noexcept { reject_reserve=reject_commit=-1;ack_fault=0; }
    static std::uint64_t reserve(void* p,int rank,int kind,std::int64_t bytes) noexcept {
        auto& l=*static_cast<Ledger*>(p);++l.reserves;
        if(rank!=0||(kind!=2&&kind!=5)||bytes<=0||bytes>(16ll<<20)){l.failed=true;return 0;}
        if(kind==l.reject_reserve){l.reject_reserve=-1;return 0;}
        for(auto& e:l.entries)if(!e.token){e={++l.next,static_cast<std::size_t>(bytes),kind,false,false};
            l.pending[kind]+=e.bytes;l.last_token=e.token;return e.token;}
        l.failed=true;return 0;
    }
    static int commit(void* p,std::uint64_t token) noexcept {
        auto& l=*static_cast<Ledger*>(p);auto* e=l.find(token);
        if(!e||e->committed||e->refunded){l.failed=true;return 0;}
        if(e->kind==l.reject_commit){l.reject_commit=-1;return 0;}
        l.pending[e->kind]-=e->bytes;l.committed[e->kind]+=e->bytes;e->committed=true;return 1;
    }
    static int release_checked(void* p,std::uint64_t token) noexcept {
        auto& l=*static_cast<Ledger*>(p);++l.releases;l.last_released=token;auto* e=l.find(token);
        if(!e){l.failed=true;return 0;}
        if(e->kind==5&&(!l.bank||l.bank->physical_bytes()!=0)){l.failed=true;return 0;}
        if(e->kind==5&&l.ack_fault==1){l.ack_fault=0;return 0;}
        if(!e->refunded){auto& amount=e->committed?l.committed[e->kind]:l.pending[e->kind];
            if(amount<e->bytes){l.failed=true;return 0;}amount-=e->bytes;e->refunded=true;++l.refunds;}
        if(e->kind==5&&l.ack_fault==2){l.ack_fault=0;return 0;}
        *e={};return 1;
    }
    static void release(void* p,std::uint64_t token) noexcept {
        // Only the fixture's independent graph kind2 uses a destructor path.
        auto& l=*static_cast<Ledger*>(p);auto* e=l.find(token);
        if(e&&e->kind==5){l.failed=true;return;}
        if(!release_checked(p,token))l.failed=true;
    }
};
struct Graph {
    ggml_backend_t backend; ggml_context* context=nullptr; ggml_gallocr_t alloc=nullptr;
    ggml_cgraph* graph=nullptr;ggml_tensor* input=nullptr;ggml_tensor* output=nullptr;
    std::shared_ptr<Charge> scratch;int rows;
    Graph(ggml_backend_t b,int n):backend(b),rows(n) {
        try {
            const auto metadata=ggml_graph_overhead_custom(64,false)+12*ggml_tensor_overhead()+4096;
            context=ggml_init({metadata,nullptr,true});need(context!=nullptr,"graph metadata allocation");
            graph=ggml_new_graph_custom(context,64,false);graph->uid=uid++;
            input=ggml_new_tensor_2d(context,GGML_TYPE_F32,width,rows);
            auto* gain=ggml_new_tensor_2d(context,GGML_TYPE_F32,width,1);
            auto* bias=ggml_new_tensor_2d(context,GGML_TYPE_F32,width,1);
            ggml_set_input(input);ggml_set_input(gain);ggml_set_input(bias);
            output=ggml_silu(context,ggml_add(context,ggml_mul(context,input,gain),bias));
            ggml_set_output(output);ggml_build_forward_expand(graph,output);
            alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));need(alloc!=nullptr,"graph allocator");
            std::size_t bytes=0;ggml_gallocr_reserve_n_size(alloc,graph,nullptr,nullptr,&bytes);
            scratch=Charge::reserve(0,2,bytes);need(scratch&&scratch->is_accounted(),"kind2 graph scratch admission");
            need(ggml_gallocr_reserve(alloc,graph)&&ggml_gallocr_get_buffer_size(alloc,0)==bytes
                &&ggml_gallocr_alloc_graph(alloc,graph),"same graph scratch allocation");
            need(scratch->commit(bytes),"graph scratch commit");
            std::vector<float> g(width),v(width);
            for(int i=0;i<width;++i){g[i]=.7f+.001f*i;v[i]=.03f*std::sin(.17f*i);}
            ggml_backend_tensor_set(gain,g.data(),0,g.size()*sizeof(float));
            ggml_backend_tensor_set(bias,v.data(),0,v.size()*sizeof(float));
        }catch(...){close();throw;}
    }
    void close() noexcept {
        if(alloc){ggml_backend_synchronize(backend);ggml_gallocr_free(alloc);alloc=nullptr;}
        scratch.reset();if(context){ggml_free(context);context=nullptr;}
    }
    ~Graph(){close();}
    void run(){need(ggml_backend_graph_compute_async(backend,graph)==GGML_STATUS_SUCCESS,"original F32 graph compute");ggml_backend_synchronize(backend);}
};
struct BankOwner {
    Ledger& ledger;ggml_backend_t backend;std::unique_ptr<Baton> bank=std::make_unique<Baton>();
    bool terminal_retention=false;
    BankOwner(Ledger& l,ggml_backend_t b):ledger(l),backend(b){ledger.bank=bank.get();}
    auto drain(){return [this](){ggml_backend_synchronize(backend);return true;};}
    bool close(){return bank->close(drain());}
    ~BankOwner(){
        ledger.disarm();
        for(int i=0;bank&&bank->owned()&&i<3;++i)(void)close();
        if(bank&&bank->owned()){
            std::fputs("FAIL: retaining exact bank/ticket/backend until process exit\n",stderr);
            terminal_retention=true;ledger.failed=true;(void)bank.release();
            // Ledger.bank remains the intentionally retained live controller.
        }else ledger.bank=nullptr;
    }
};
std::vector<float> seed(int repeat){
    std::vector<float> out(static_cast<std::size_t>(width)*tokens);
    for(int t=0;t<tokens;++t)for(int c=0;c<width;++c)
        out[static_cast<std::size_t>(t)*width+c]=.21f*std::cos(.031f*c+.19f*t+.13f*repeat);
    return out;
}
void arithmetic(){
    std::size_t bytes=99;
    need(tsg_glm::device_baton_bytes(4096,4,1872,bytes)&&bytes==122683392,"actual1872 bank arithmetic");
    need(tsg_glm::device_baton_bytes(4096,4,2048,bytes)&&bytes==134217728,"actual2048 bank arithmetic");
    for(const auto& shape:std::array<std::array<std::size_t,3>,5>{{{{0,4,19}},{{32,0,19}},{{32,4,0}},{{SIZE_MAX,4,19}},{{4096,4,SIZE_MAX}}}})
        need(!tsg_glm::device_baton_bytes(shape[0],shape[1],shape[2],bytes)&&bytes==0,"bank arithmetic refusal must clear output");
    need(tsg_glm::device_baton_bytes(hidden,channels,tokens,bytes)&&bytes==payload,"tiny exact payload arithmetic");
}
unsigned rejected_callbacks=0;
bool reject_copy(ggml_backend_buffer_t,const ggml_tensor*,ggml_tensor*){++rejected_callbacks;return false;}
void direct_copy_proof(BankOwner& owner,Graph& graph,ggml_backend_buffer_type_t type){
    stage("direct-copy-poison-and-refusal");auto& bank=*owner.bank;
    need(bank.open(type,payload),"bank open for direct copy");
    std::vector<float> original(static_cast<std::size_t>(width)*graph.rows),poison(original.size(),-1234.125f),observed(original.size());
    for(std::size_t i=0;i<original.size();++i)original[i]=.001f*static_cast<float>(i)+.25f;
    ggml_backend_tensor_set(graph.input,original.data(),0,original.size()*sizeof(float));
    need(bank.transfer(graph.input,0,true),"direct to-bank copy");
    ggml_backend_tensor_set(graph.input,poison.data(),0,poison.size()*sizeof(float));
    ggml_backend_tensor_get(graph.input,observed.data(),0,observed.size()*sizeof(float));
    need(std::memcmp(observed.data(),original.data(),observed.size()*sizeof(float))!=0,"poison must alter real destination");
    // Borrow a public buffer wrapper over its unchanged real backing context.
    // Only this TEST destination callback is replaced; never free this wrapper.
    ggml_backend_buffer wrapper=*graph.input->buffer;wrapper.iface.cpy_tensor=&reject_copy;
    ggml_tensor refused=*graph.input;refused.buffer=&wrapper;
    const auto calls=bank.input_calls;
    need(!bank.transfer(&refused,0,false)&&rejected_callbacks==1&&bank.input_calls==calls,"false direct callback must not fallback/publish");
    ggml_backend_tensor_get(graph.input,observed.data(),0,observed.size()*sizeof(float));exact(poison,observed,"refused callback modified destination");
    need(bank.transfer(graph.input,0,false),"direct bank to-input copy");
    ggml_backend_tensor_get(graph.input,observed.data(),0,observed.size()*sizeof(float));exact(original,observed,"direct callback failed actual overwrite");
    const auto before=bank.input_calls;
    need(!bank.transfer(nullptr,0,false)&&!bank.transfer(graph.input,payload,false)
        &&!bank.transfer(graph.input,SIZE_MAX,false)&&bank.input_calls==before,"invalid copy changed published counters");
    need(owner.close()&&!bank.owned(),"direct copy checked cleanup");
}
void fault_lifetimes(BankOwner& owner,Ledger& ledger,ggml_backend_buffer_type_t type){
    auto& bank=*owner.bank;stage("reserve-and-commit-refusal");
    ledger.reject_reserve=5;need(!bank.open(type,payload)&&!bank.owned()&&!ledger.pending[5]&&!ledger.committed[5],"reserve refusal owns no payload");
    ledger.reject_commit=5;need(!bank.open(type,payload)&&bank.owned()&&bank.physical_bytes()==payload
        &&bank.held_ticket_bytes()==payload&&ledger.pending[5]==payload&&!ledger.committed[5],"commit refusal must retain physical and original pending ticket");
    need(!Charge::detach(&ledger),"pending exact owner detached callback context");
    need(owner.close()&&!bank.owned()&&!ledger.pending[5],"commit refusal checked release");
    stage("drain-false-and-exception-retain");need(bank.open(type,payload),"drain fault bank open");
    const auto token=ledger.last_token;const auto release_calls=ledger.releases;
    need(!bank.close([](){return false;})&&bank.physical_bytes()==payload&&ledger.releases==release_calls
        &&ledger.last_token==token&&ledger.committed[5]==payload,"failed drain freed or refunded physical owner");
    need(!bank.close([]()->bool{throw std::runtime_error("injected prefree drain");})&&bank.physical_bytes()==payload
        &&ledger.releases==release_calls,"throwing prefree drain lost owner");
    need(owner.close()&&!bank.owned(),"drain retry checked cleanup");
    for(int mode:{1,2}){
        stage(mode==1?"pre-ledger-ack-retains-ticket":"post-ledger-ack-retains-same-ticket");
        need(bank.open(type,payload),"ACK fault bank open");const auto exact_token=ledger.last_token;
        const auto reserves=ledger.reserves;const auto refunds=ledger.refunds;ledger.ack_fault=mode;
        unsigned drains=0;
        need(!bank.close([&](){++drains;ggml_backend_synchronize(owner.backend);return true;})
            &&bank.owned()&&!bank.physical_bytes()&&bank.held_ticket_bytes()==payload
            &&ledger.last_released==exact_token&&ledger.find(exact_token)!=nullptr,"failed ACK must retain same ticket after physical free");
        need(ledger.committed[5]==(mode==1?payload:0)&&ledger.refunds==refunds+(mode==2?1:0),"pre/post ledger ACK stage is not observable");
        need(!Charge::detach(&ledger),"ticket-only quarantine detached callback context");
        need(!bank.open(type,payload)&&ledger.reserves==reserves,"ticket-only quarantine admitted another bank");
        need(bank.close([&](){++drains;return true;})&&!bank.owned()&&drains==1
            &&ledger.reserves==reserves&&ledger.last_released==exact_token&&!ledger.find(exact_token)
            &&ledger.refunds==refunds+1&&!ledger.pending[5]&&!ledger.committed[5],"same-token ACK retry doubled credit or redrained freed buffer");
        need(bank.close([](){return false;}),"already closed bank must be idempotent");
    }
}
std::vector<float> run_host(Graph& wide,Graph& tail,const std::vector<float>& initial,std::uint64_t& host_bytes){
    auto current=initial;
    for(int layer=0;layer<layers;++layer)for(int first=0;first<tokens;first+=chunk){
        auto& graph=tokens-first<chunk?tail:wide;const auto count=static_cast<std::size_t>(graph.rows)*width;
        ggml_backend_tensor_set(graph.input,current.data()+static_cast<std::size_t>(first)*width,0,count*sizeof(float));
        if(layer>0)host_bytes+=count*sizeof(float);
        graph.run();ggml_backend_tensor_get(graph.output,current.data()+static_cast<std::size_t>(first)*width,0,count*sizeof(float));
        if(layer<layers-1)host_bytes+=count*sizeof(float);
    }
    return current;
}
std::vector<float> run_bank(BankOwner& owner,Graph& wide,Graph& tail,const std::vector<float>& initial,ggml_backend_buffer_type_t type){
    auto& bank=*owner.bank;need(bank.open(type,payload),"layer-major bank open");std::vector<float> result(initial.size());
    for(int layer=0;layer<layers;++layer)for(int first=0;first<tokens;first+=chunk){
        auto& graph=tokens-first<chunk?tail:wide;const auto offset=static_cast<std::size_t>(first)*width*sizeof(float);
        const auto count=static_cast<std::size_t>(graph.rows)*width;
        if(layer==0)ggml_backend_tensor_set(graph.input,initial.data()+static_cast<std::size_t>(first)*width,0,count*sizeof(float));
        else need(bank.transfer(graph.input,offset,false),"bank-to-original-input transfer");
        graph.run();
        if(layer==layers-1)ggml_backend_tensor_get(graph.output,result.data()+static_cast<std::size_t>(first)*width,0,count*sizeof(float));
        else need(bank.transfer(graph.output,offset,true),"original-output-to-bank transfer");
    }
    const auto calls=static_cast<std::uint64_t>(layers-1)*3;
    need(bank.input_calls==calls&&bank.output_calls==calls&&bank.input_bytes==static_cast<std::uint64_t>(layers-1)*payload
        &&bank.output_bytes==static_cast<std::uint64_t>(layers-1)*payload,"actual D2D call/byte coverage differs across chunks/tail");
    need(owner.close()&&!bank.owned()&&!bank.physical_bytes()&&!bank.held_ticket_bytes(),"bank must retire before result publication");
    return result;
}
}
int main(){
    if(const auto* devices=std::getenv("GGML_CUDA_DEVICES");devices&&*devices){std::puts("UNAVAILABLE: virtual device mapping outside fixture");return 77;}
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices<1||cudaSetDevice(0)!=cudaSuccess||cudaFree(nullptr)!=cudaSuccess){
        std::puts("UNAVAILABLE: original CUDA device0 required");return 77;}
    auto backend=ggml_backend_cuda_init(0);
    if(!backend){std::fputs("FAIL: backend unavailable after positive device bootstrap\n",stderr);return 1;}
    auto ledger=std::make_unique<Ledger>();bool attached=false;int result=1;
    try{
        arithmetic();const auto type=ggml_backend_get_default_buffer_type(backend);
        need(type==ggml_backend_cuda_buffer_type(0),"exact original CUDA buft required");
        {
            Baton absent;const auto calls=ledger->reserves;
            need(!absent.open(type,payload)&&!absent.owned()&&ledger->reserves==calls,"unattached bank must refuse before reservation");
            need(Charge::attach(ledger.get(),Ledger::reserve,Ledger::commit,Ledger::release,true,false),"legacy graph ledger attach");
            need(!absent.open(type,payload)&&!absent.owned()&&ledger->reserves==calls,"legacy void-only scope cannot own bank");
            need(Charge::detach(ledger.get()),"legacy zero-owner detach");
            need(Charge::attach(ledger.get(),Ledger::reserve,Ledger::commit,Ledger::release,false,false,Ledger::release_checked),"non-graph checked ledger attach");
            need(!absent.open(type,payload)&&!absent.owned()&&ledger->reserves==calls,"unaccounted graph scope cannot own bank");
            need(Charge::detach(ledger.get()),"non-graph zero-owner detach");
        }
        need(Charge::attach(ledger.get(),Ledger::reserve,Ledger::commit,Ledger::release,true,false,Ledger::release_checked),"checked graph ledger attach");attached=true;
        {
            BankOwner owner(*ledger,backend);
            // No kind2 graph owner may mask the kind5-only detach assertions.
            fault_lifetimes(owner,*ledger,type);
            {
                Graph wide(backend,chunk),tail(backend,tokens%chunk);
                direct_copy_proof(owner,wide,type);
                for(int repeat=0;repeat<3;++repeat){
                    stage("same-graph-host-bounce-vs-device-bank");auto initial=seed(repeat);std::uint64_t host_bytes=0;
                    auto expected=run_host(wide,tail,initial,host_bytes);auto actual=run_bank(owner,wide,tail,initial,type);
                    need(host_bytes==2ull*(layers-1)*payload,"Host reference hidden-transfer coverage");
                    exact(expected,actual,"same original CUDA F32 graphs differ after bank transfer");
                    need(!ledger->pending[5]&&!ledger->committed[5],"successful call retained kind5 credit");
                }
            }
        }
        need(!ledger->failed&&ledger->live()==0,"all exact graph/bank callback owners must retire");
        for(int k=0;k<8;++k)need(!ledger->pending[k]&&!ledger->committed[k],"final ledger nonzero");
        need(Charge::detach(ledger.get()),"final zero-owner registry detach");attached=false;
        ggml_backend_synchronize(backend);ggml_backend_free(backend);backend=nullptr;
        std::printf("PASS GLM device baton checks=%llu bitexact_f32_elements=%llu layers=6 chunks=8,8,3 repeats=3 bank_bytes=%zu "
            "intermediate_host_hidden_bytes=0 direct_callback_refusal=1 checked_pending=0 committed=0; "
            "originalCUDA capture unobserved; no model/math-policy/performance/vendor-free-error claim\n",
            static_cast<unsigned long long>(checks),static_cast<unsigned long long>(exact_f32),payload);result=0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());}
    if(attached&&!Charge::detach(ledger.get())){
        std::fputs("FAIL: retain callback context/backend for unresolved exact owner\n",stderr);(void)ledger.release();return 1;
    }
    if(backend){ggml_backend_synchronize(backend);ggml_backend_free(backend);}return result;
}
