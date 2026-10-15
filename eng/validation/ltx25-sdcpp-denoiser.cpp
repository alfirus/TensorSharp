// Owned diagnostic/benchmark executable; consumes unchanged SD.cpp headers/libs.
#include "model/diffusion/ltxv.hpp"
#include "model_manager.h"
#include "json.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using nlohmann::json;

static std::vector<float> values(size_t length, uint32_t seed) {
    std::vector<float> result(length);
    for (auto& value : result) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        value = ((seed >> 8) * (1.f / 16777216.f) - .5f) * .2f;
    }
    return result;
}

static void write_floats(const std::string& path, const std::vector<float>& values) {
    if (std::filesystem::exists(path)) throw std::runtime_error("Output already exists: " + path);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!file) throw std::runtime_error("Could not write output");
}

int main(int argc, char** argv) {
    std::unordered_map<std::string, std::string> options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || !options.emplace(argv[index], argv[index + 1]).second) return 2;
    }
    auto option = [&](const std::string& name, const std::string& fallback) {
        const auto found = options.find(name);
        return found == options.end() ? fallback : found->second;
    };
    const std::string model_path = option("--model", ""), output = option("--output", "");
    if (model_path.empty() || output.empty() || std::filesystem::exists(output)) return 2;
    std::filesystem::create_directories(std::filesystem::path(output).parent_path());
    json report = {{"completed", false}, {"scope", "AV denoiser with deterministic synthetic projected contexts and latents; excludes Gemma, connectors, sampler, VAEs, codec and applications"},
                   {"model", model_path}, {"source_identity", "See adjacent build.json and the verified unchanged baseline source manifest."},
                   {"limitations", {"SD.cpp reference has independent rotary/timestep conventions; this benchmark is not an official LTX numerical-quality oracle.",
                                     "Runtime configuration bypasses caption projection/connectors because contexts are already projected, and absolute keyframe embedding because the TensorSharp probe supplies no keyframe mask.",
                                     "Warm timings synchronize the returned packed audiovisual tensor; model initialization/loading is reported separately."}}};
    const auto save = [&]() { std::ofstream(output) << report.dump(2) << '\n'; };
    save();
    ggml_backend_t backend = nullptr;
    try {
        const int width = std::stoi(option("--latent-width", "2"));
        const int height = std::stoi(option("--latent-height", "2"));
        const int frames = std::stoi(option("--latent-frames", "2"));
        const int audio_tokens = std::stoi(option("--audio-tokens", "16"));
        const int context_tokens = std::stoi(option("--context-tokens", "8"));
        const int warmup = std::stoi(option("--warmup", "1")), repeats = std::stoi(option("--repeats", "5"));
        const int threads = std::stoi(option("--threads", "8"));
        const float sigma = std::stof(option("--sigma", "0.725"));
        if (width <= 0 || height <= 0 || frames <= 0 || audio_tokens <= 0 || context_tokens <= 0 || warmup < 0 || repeats <= 0) throw std::runtime_error("Invalid counts");
        const int video_tokens = width * height * frames;
        ggml_backend_load_all();
        const std::string backend_name = option("--backend", "CUDA0");
        backend = ggml_backend_init_by_name(backend_name.c_str(), nullptr);
        if (!backend) throw std::runtime_error("Requested backend unavailable: " + backend_name);
        const auto load_start = std::chrono::steady_clock::now();
        auto manager = std::make_shared<ModelManager>();
        manager->set_n_threads(threads);
        manager->set_enable_mmap(true);
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(model_path, "model.diffusion_model.")) throw std::runtime_error("Model loader initialization failed");
        auto runner = std::make_shared<LTXV::LTXAVRunner>(backend, loader.get_tensor_storage_map(), "model.diffusion_model", manager);
        const int layers = std::stoi(option("--blocks", std::to_string(runner->config.num_layers)));
        if (layers <= 0 || layers > runner->config.num_layers) throw std::runtime_error("Invalid block count");
        const int full_layers = runner->config.num_layers;
        runner->config.num_layers = runner->model.config.num_layers = layers;
        runner->config.use_connector = runner->model.config.use_connector = false;
        runner->config.use_audio_connector = runner->model.config.use_audio_connector = false;
        runner->config.use_caption_projection = runner->model.config.use_caption_projection = false;
        runner->config.use_audio_caption_projection = runner->model.config.use_audio_caption_projection = false;
        runner->config.use_keyframes_abs_pos_embedding = runner->model.config.use_keyframes_abs_pos_embedding = false;
        runner->set_flash_attention_enabled(true);
        if (!manager->set_loader(loader) || !manager->register_runner_params(ModelComponent::Diffusion, *runner, "model.diffusion_model", ModelManager::ResidencyMode::ParamBackend, backend, backend) || !manager->validate_registered_tensors()) throw std::runtime_error("Model parameter registration failed");
        if (!manager->load_all_params_eagerly()) throw std::runtime_error("Model parameter loading failed");
        report["load_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
        const auto& config = runner->config;
        const int vc = config.in_channels, ac = config.num_audio_channels * config.audio_frequency_bins;
        const int vctx = config.cross_attention_dim, actx = config.audio_cross_attention_dim;
        const auto video_input = values(static_cast<size_t>(video_tokens) * vc, 1);
        const auto audio_input = values(static_cast<size_t>(audio_tokens) * ac, 2);
        const auto video_context = values(static_cast<size_t>(context_tokens) * vctx, 3);
        const auto audio_context = values(static_cast<size_t>(context_tokens) * actx, 4);
        write_floats(output + ".input-video.f32", video_input);
        write_floats(output + ".input-audio.f32", audio_input);
        write_floats(output + ".input-video-context.f32", video_context);
        write_floats(output + ".input-audio-context.f32", audio_context);
        sd::Tensor<float> video({width, height, frames, vc});
        for (int t = 0; t < video_tokens; ++t) for (int c = 0; c < vc; ++c) video.data()[c * video_tokens + t] = video_input[t * vc + c];
        sd::Tensor<float> audio({config.audio_frequency_bins, audio_tokens, config.num_audio_channels, 1});
        for (int t = 0; t < audio_tokens; ++t) for (int c = 0; c < config.num_audio_channels; ++c) for (int f = 0; f < config.audio_frequency_bins; ++f)
            audio.data()[(c * audio_tokens + t) * config.audio_frequency_bins + f] = audio_input[t * ac + c * config.audio_frequency_bins + f];
        sd::Tensor<float> context({vctx + actx, context_tokens, 1});
        for (int t = 0; t < context_tokens; ++t) {
            std::copy_n(video_context.data() + t * vctx, vctx, context.data() + t * (vctx + actx));
            std::copy_n(audio_context.data() + t * actx, actx, context.data() + t * (vctx + actx) + vctx);
        }
        auto timestep = sd::Tensor<float>::from_vector(std::vector<float>{sigma});
        report["backend"] = backend_name;
        report["executed_blocks"] = layers; report["full_depth"] = layers == full_layers;
        report["tokens"] = {{"video", video_tokens}, {"audio", audio_tokens}, {"context", context_tokens}};
        report["latent_geometry"] = {width, height, frames}; report["sigma"] = sigma;
        report["dimensions"] = {{"video_channels", vc}, {"audio_channels", ac}, {"video_context", vctx}, {"audio_context", actx}};
        report["warmup"] = warmup; report["repeats"] = repeats;
        std::vector<double> samples;
        std::vector<float> first_video, first_audio, video_result, audio_result;
        double maximum_difference = 0;
        for (int index = 0; index < warmup + repeats; ++index) {
            const auto start = std::chrono::steady_clock::now();
            const auto packed = runner->compute(threads, video, timestep, context, audio, timestep, audio_tokens, 24.f);
            const auto split = runner->split_av_latents(packed, audio_tokens);
            if (split.first.empty() || split.second.empty()) throw std::runtime_error("Empty model output");
            video_result.resize(video_input.size()); audio_result.resize(audio_input.size());
            for (int t = 0; t < video_tokens; ++t) for (int c = 0; c < vc; ++c) video_result[t * vc + c] = split.first.data()[c * video_tokens + t];
            for (int t = 0; t < audio_tokens; ++t) for (int c = 0; c < config.num_audio_channels; ++c) for (int f = 0; f < config.audio_frequency_bins; ++f)
                audio_result[t * ac + c * config.audio_frequency_bins + f] = split.second.data()[(c * audio_tokens + t) * config.audio_frequency_bins + f];
            const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (index == 0) report["first_forward_ms_excluding_eager_weight_upload"] = milliseconds;
            if (index >= warmup) samples.push_back(milliseconds);
            if (first_video.empty()) { first_video = video_result; first_audio = audio_result; }
            for (size_t i = 0; i < video_result.size(); ++i) {
                if (!std::isfinite(video_result[i])) throw std::runtime_error("Nonfinite video output");
                maximum_difference = std::max(maximum_difference, std::abs(static_cast<double>(video_result[i]) - first_video[i]));
            }
            for (size_t i = 0; i < audio_result.size(); ++i) {
                if (!std::isfinite(audio_result[i])) throw std::runtime_error("Nonfinite audio output");
                maximum_difference = std::max(maximum_difference, std::abs(static_cast<double>(audio_result[i]) - first_audio[i]));
            }
            report["forward_samples_ms"] = samples;
            save();
            std::cerr << "Forward " << index + 1 << '/' << warmup + repeats << ": " << milliseconds << " ms\n";
        }
        write_floats(output + ".video.f32", video_result); write_floats(output + ".audio.f32", audio_result);
        std::sort(samples.begin(), samples.end());
        report["median_ms"] = (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) * .5;
        report["maximum_repeat_difference"] = maximum_difference;
        report["completed"] = true;
        // The manager stores tensor addresses owned by the runner context.
        runner->runner_end();
        if (!manager->unregister_param_tensors(ModelComponent::Diffusion)) throw std::runtime_error("Model parameter release failed");
        runner.reset(); manager.reset();
        ggml_backend_free(backend); backend = nullptr;
        save();
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        if (backend) ggml_backend_free(backend);
        save();
        return 1;
    }
}
