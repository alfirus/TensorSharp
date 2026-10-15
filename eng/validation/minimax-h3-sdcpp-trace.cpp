// Diagnostic wrapper around an unchanged official SD CLI translation unit.
// Build with build-minimax-h3-sdcpp-trace.py; never edit the dependency tree.
#include "ggml.h"
#include "ggml-backend.h"
#include "stable-diffusion.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#define main minimax_h3_official_cli_main
#include SD_CLI_SOURCE
#undef main

namespace {
using nlohmann::json;
struct Trace {
    std::filesystem::path output;
    std::ofstream nodes;
    std::unordered_map<const ggml_tensor*, size_t> ids;
    std::unordered_set<const ggml_tensor*> dumped;
    size_t next_id = 0;
    int last_block = -1;
    bool captured_inputs = false;
    bool completed = false;
    bool audio_phase = false;

    size_t id(const ggml_tensor* tensor) {
        auto inserted = ids.emplace(tensor, next_id);
        if (inserted.second) ++next_id;
        return inserted.first->second;
    }
    json metadata(const ggml_tensor* tensor) {
        json sources = json::array();
        for (const auto* source : tensor->src)
            if (source) sources.push_back({{"id", id(source)}, {"name", source->name},
                                          {"ne", {source->ne[0], source->ne[1], source->ne[2], source->ne[3]}}});
        return {{"id", id(tensor)}, {"name", tensor->name}, {"type", ggml_type_name(tensor->type)},
                {"op", ggml_op_name(tensor->op)}, {"ne", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}},
                {"nb", {tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]}}, {"sources", sources}};
    }
    void dump(ggml_tensor* tensor, const std::string& label) {
        if (!tensor || tensor->type != GGML_TYPE_F32 ||
            (!tensor->buffer && (!tensor->view_src || !tensor->view_src->buffer)) || !dumped.insert(tensor).second) return;
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        std::vector<float> values;
        values.reserve(ggml_nelements(tensor));
        for (int64_t i3=0; i3<tensor->ne[3]; ++i3)
            for (int64_t i2=0; i2<tensor->ne[2]; ++i2)
                for (int64_t i1=0; i1<tensor->ne[1]; ++i1)
                    for (int64_t i0=0; i0<tensor->ne[0]; ++i0) {
                        float value;
                        const size_t offset=i0*tensor->nb[0]+i1*tensor->nb[1]+i2*tensor->nb[2]+i3*tensor->nb[3];
                        std::memcpy(&value, bytes.data()+offset, sizeof(value));
                        values.push_back(value);
                    }
        const auto name=label+"-"+std::to_string(id(tensor))+".f32";
        std::ofstream file(output/name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float));
        auto entry=metadata(tensor);
        entry["file"]=name;
        std::ofstream(output/(name+".json")) << entry.dump(2) << '\n';
    }
    void input_leaves(ggml_tensor* tensor, std::unordered_set<ggml_tensor*>& visited, const std::string& label="input") {
        if (!tensor || !visited.insert(tensor).second) return;
        if (tensor->op == GGML_OP_NONE && tensor->type == GGML_TYPE_F32 &&
            ggml_nelements(tensor) <= 2000000 && std::string(tensor->name).find("weight") == std::string::npos)
            dump(tensor, label);
        for (auto* source : tensor->src) input_leaves(source, visited, label);
    }
};
bool evaluate(ggml_tensor* tensor, bool ask, void* context) {
    auto& trace=*static_cast<Trace*>(context);
    bool audio_projection=false, audio_normalization=false, condition_bias=false;
    for (const auto* source : tensor->src) if (source) {
        const std::string source_name=source->name;
        if (source_name.find("dec_in_proj.weight")!=std::string::npos &&
            (source->ne[0]==32 || source->ne[1]==32)) audio_projection=true;
        if (source_name.find("condition_proj.bias")!=std::string::npos) condition_bias=true;
        if (tensor->op==GGML_OP_MUL && ggml_nelements(source)==32 && source_name.find("latents_std")!=std::string::npos)
            audio_normalization=true;
    }
    if (trace.completed && !trace.audio_phase && !audio_projection && !audio_normalization) return ask ? false : true;
    if ((audio_projection || audio_normalization) && !trace.audio_phase) {
        trace.audio_phase=true;
        // Later stage contexts can reuse the original DiT tensor addresses.
        trace.ids.clear(); trace.dumped.clear();
    }
    if (trace.audio_phase) {
        if (ask) {
            trace.nodes << trace.metadata(tensor).dump() << '\n';
            if(audio_normalization) {
                // The graph inputs have been uploaded, but no audio compute
                // segment has run yet. Snapshot the leaf before in-place reuse.
                std::unordered_set<ggml_tensor*> visited;
                trace.input_leaves(tensor,visited,"audio-carrier-before-normalization");
            }
            return audio_projection;
        }
        trace.dump(tensor,"audio-projection");
        if (tensor->src[1]) trace.dump(tensor->src[1],"audio-normalized-input");
        std::unordered_set<ggml_tensor*> visited;
        trace.input_leaves(tensor,visited,"audio-input");
        return true;
    }
    const std::string name=tensor->name;
    const std::string prefix="ggml_runner_cut:minimax_h3.blocks.";
    const std::string suffix="|hidden_states|";
    const bool block=name.rfind(prefix, 0)==0 && name.size()>=suffix.size() &&
        name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
    bool projection=false;
    for (const auto* source : tensor->src)
        if (source && (std::string(source->name).find("audio_patch_proj.weight")!=std::string::npos ||
                       std::string(source->name).find("video_patch_proj.weight")!=std::string::npos ||
                       std::string(source->name).find("condition_proj.weight")!=std::string::npos ||
                       std::string(source->name).find("blocks.0.adaln_proj.linear.weight")!=std::string::npos)) projection=true;
    const bool packed=trace.last_block<0 && tensor->op==GGML_OP_CONCAT && tensor->ne[0]>=1024 && tensor->ne[1]<=512;
    std::string refiner;
    const auto* update=tensor->src[1];
    while (update && update->op==GGML_OP_SCALE) update=update->src[0];
    if (tensor->op==GGML_OP_ADD && update && update->op==GGML_OP_MUL_MAT && update->src[0]) {
        std::string weight=update->src[0]->name;
        const std::string refiner_prefix="token_refiner.blocks.";
        auto begin=weight.find(refiner_prefix);
        if(begin!=std::string::npos && weight.find("mlp.fc2.weight")!=std::string::npos)
            refiner=weight.substr(begin+refiner_prefix.size(),weight.find('.',begin+refiner_prefix.size())-begin-refiner_prefix.size());
    }
    if (ask) {
        trace.nodes << trace.metadata(tensor).dump() << '\n';
        if (block) trace.last_block=std::stoi(name.substr(prefix.size()));
        // Block boundaries plus the small final projections/carrier packing.
        return block || projection || condition_bias || !refiner.empty() || packed || (trace.last_block==49 && tensor->type==GGML_TYPE_F32 && ggml_nelements(tensor)<=20000);
    }
    if (block) {
        if (!trace.captured_inputs) {
            std::unordered_set<ggml_tensor*> visited;
            trace.input_leaves(tensor, visited);
            trace.captured_inputs=true;
        }
        trace.dump(tensor, "block-"+std::to_string(trace.last_block));
    } else if (condition_bias) trace.dump(tensor,"condition-proj");
    else if (!refiner.empty()) trace.dump(tensor,"refiner-"+refiner);
    else if (packed) trace.dump(tensor, "packed-in");
    else if (projection) {
        trace.dump(tensor, "projection");
        if (tensor->src[1]) trace.dump(tensor->src[1], "projection-input");
    } else {
        trace.dump(tensor, "final");
        if (tensor->op==GGML_OP_CONCAT && tensor->ne[3]>24 && tensor->src[0] && tensor->src[0]->ne[3]==24)
            trace.completed=true;
    }
    return true;
}
}

int main(int argc, const char* argv[]) {
    const char* directory=std::getenv("TS_H3_SD_TRACE_DIR");
    if (!directory || !*directory) {
        std::fprintf(stderr, "Set TS_H3_SD_TRACE_DIR to a fresh ignored evidence directory.\n");
        return 2;
    }
    Trace trace;
    trace.output=directory;
    if (std::filesystem::exists(trace.output/"nodes.jsonl")) {
        std::fprintf(stderr, "Refusing to overwrite existing diagnostic evidence.\n");
        return 2;
    }
    std::filesystem::create_directories(trace.output);
    trace.nodes.open(trace.output/"nodes.jsonl");
    sd_set_backend_eval_callback(evaluate, &trace);
    const int result=minimax_h3_official_cli_main(argc, argv);
    sd_set_backend_eval_callback(nullptr, nullptr);
    return result;
}
