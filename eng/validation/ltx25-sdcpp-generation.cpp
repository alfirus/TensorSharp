// TensorSharp-owned benchmark; calls the unmodified stable-diffusion.cpp C API.
#include "stable-diffusion.h"
#include "json.hpp"
#include "common/media_io.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using nlohmann::json;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void write(const std::string& path, const void* data, size_t bytes) {
    if (std::filesystem::exists(path)) throw std::runtime_error("Output already exists: " + path);
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char*>(data), bytes);
    if (!file) throw std::runtime_error("Could not write generated output");
}
struct Media {
    sd_image_t* frames = nullptr;
    sd_audio_t* audio = nullptr;
    int count = 0, fps = 0;
    ~Media() { free_sd_images(frames, count); free_sd_audio(audio); }
};
struct ConditionImage {
    sd_image_t value{};
    ~ConditionImage() { free(value.data); }
    void load(const std::string& path, int width, int height) {
        if (!path.empty() && !load_sd_image_from_file(&value, path.c_str(), width, height, 3))
            throw std::runtime_error("Failed to decode conditioning image: " + path);
    }
};
int main(int argc, char** argv) {
    std::unordered_map<std::string, std::string> options;
    for (int i = 1; i < argc; i += 2)
        if (i + 1 >= argc || !options.emplace(argv[i], argv[i + 1]).second) return 2;
    auto option = [&](const std::string& name, const std::string& fallback) {
        auto value = options.find(name);
        return value == options.end() ? fallback : value->second;
    };
    const std::string model = option("--model", ""), encoder = option("--encoder", ""),
        video_vae = option("--video-vae", ""), audio_vae = option("--audio-vae", ""),
        output = option("--output", ""), backend = option("--backend", "CUDA0"),
        prompt = option("--prompt", "A red toy car drives slowly along a quiet road in warm afternoon light.");
    if (model.empty() || encoder.empty() || video_vae.empty() || audio_vae.empty() || output.empty() || std::filesystem::exists(output)) return 2;
    std::filesystem::create_directories(std::filesystem::path(output).parent_path());
    json report = {{"completed", false}, {"scope", "Unmodified stable-diffusion.cpp C API: full prompt conditioning, eight-step48-block AV denoising, video/audio decoding. Excludes file output and media codecs."},
        {"source_identity", "See adjacent build.json and verified clean source/build manifests."},
        {"backend", backend}, {"prompt", prompt}, {"model", model}, {"executions", json::array()}};
    auto save = [&]() { std::ofstream(output) << report.dump(2) << '\n'; };
    sd_ctx_t* context = nullptr;
    try {
        const int width = std::stoi(option("--width", "512")), height = std::stoi(option("--height", "512")),
            frames = std::stoi(option("--frames", "17")), fps = std::stoi(option("--fps", "24")),
            repeats = std::stoi(option("--repeats", "2"));
        if (width <= 0 || height <= 0 || frames <= 0 || fps <= 0 || repeats < 1) throw std::runtime_error("Invalid generation counts");
        ConditionImage first_image, last_image;
        const std::string first_path = option("--image", ""), last_path = option("--end-image", "");
        first_image.load(first_path, width, height);
        last_image.load(last_path, width, height);
        report["reference_inputs"] = json::array();
        if (!first_path.empty()) report["reference_inputs"].push_back({{"path", std::filesystem::absolute(first_path).string()}, {"frame_index", 0}, {"strength", 1}});
        if (!last_path.empty()) report["reference_inputs"].push_back({{"path", std::filesystem::absolute(last_path).string()}, {"frame_index", frames - 1}, {"strength", 1}});
        report["reference_limitations"] = "Pinned upstream LTX implementation accepts first and last images only; arbitrary interior/multiple guide images have no C API mapping and must not be benchmarked as if supported.";
        sd_ctx_params_t setup;
        sd_ctx_params_init(&setup);
        setup.diffusion_model_path = model.c_str(); setup.llm_path = encoder.c_str();
        setup.vae_path = video_vae.c_str(); setup.audio_vae_path = audio_vae.c_str();
        setup.backend = backend.c_str(); setup.params_backend = "cpu";
        setup.enable_mmap = true; setup.eager_load = false; setup.diffusion_flash_attn = true;
        setup.rng_type = CUDA_RNG; setup.sampler_rng_type = CUDA_RNG;
        setup.n_threads = std::stoi(option("--threads", "8"));
        setup.conditioning_cache_size = 4;
        auto start = Clock::now();
        context = new_sd_ctx(&setup);
        report["model_factory_ms"] = elapsed(start);
        if (!context || !sd_ctx_supports_video_generation(context)) throw std::runtime_error("Baseline failed to create a video model");
        std::vector<float> sigmas{1.f, .99375f, .9875f, .98125f, .975f, .909375f, .725f, .421875f, 0.f};
        report["settings"] = {{"width", width}, {"height", height}, {"frames", frames}, {"fps", fps},
            {"steps", 8}, {"cfg", 1}, {"seed", std::stoll(option("--seed", "42"))}, {"sigmas", sigmas},
            {"conditioning_cache_size", 4}, {"params_backend", "cpu"}};
        save();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            sd_vid_gen_params_t request;
            sd_vid_gen_params_init(&request);
            request.prompt = prompt.c_str(); request.width = width; request.height = height;
            request.video_frames = frames; request.fps = fps; request.seed = std::stoll(option("--seed", "42"));
            request.init_image = first_image.value; request.end_image = last_image.value; request.strength = 1;
            request.sample_params.sample_steps = 8; request.sample_params.guidance.txt_cfg = 1.f;
            request.sample_params.sample_method = EULER_SAMPLE_METHOD;
            request.sample_params.custom_sigmas = sigmas.data(); request.sample_params.custom_sigmas_count = static_cast<int>(sigmas.size());
            Media media;
            start = Clock::now();
            bool success = generate_video(context, &request, &media.frames, &media.count, &media.audio, &media.fps);
            double generation_ms = elapsed(start);
            if (!success || !media.frames || media.count != frames || media.fps != fps || !media.audio ||
                media.audio->sample_rate != 48000 || media.audio->channels != 2 || !media.audio->sample_count)
                throw std::runtime_error("Baseline generated unexpected AV geometry");
            std::vector<uint8_t> rgb;
            for (int frame = 0; frame < media.count; ++frame) {
                const auto& image = media.frames[frame];
                if (image.width != static_cast<unsigned>(width) || image.height != static_cast<unsigned>(height) || image.channel != 3)
                    throw std::runtime_error("Baseline returned unexpected RGB geometry");
                rgb.insert(rgb.end(), image.data, image.data + static_cast<size_t>(width) * height * 3);
            }
            const size_t samples = media.audio->sample_count * media.audio->channels;
            double square = 0, peak = 0;
            for (size_t i = 0; i < samples; ++i) {
                float value = media.audio->data[i];
                if (!std::isfinite(value)) throw std::runtime_error("Baseline generated nonfinite audio");
                square += static_cast<double>(value) * value; peak = std::max(peak, static_cast<double>(std::abs(value)));
            }
            const std::string stem = output + "." + std::to_string(repeat);
            write(stem + ".rgb.u8", rgb.data(), rgb.size());
            write(stem + ".audio.f32", media.audio->data, samples * sizeof(float));
            report["executions"].push_back({{"repeat", repeat}, {"generation_ms", generation_ms}, {"frames", media.count},
                {"fps", media.fps}, {"rgb_path", stem + ".rgb.u8"}, {"audio_path", stem + ".audio.f32"},
                {"audio_samples", media.audio->sample_count}, {"audio_rate", media.audio->sample_rate},
                {"audio_seconds", media.audio->sample_count / 48000.0}, {"audio_peak", peak}, {"audio_rms", std::sqrt(square / samples)}});
            save();
        }
        report["completed"] = true;
    } catch (const std::exception& error) {
        report["error"] = error.what(); save(); free_sd_ctx(context); return 1;
    }
    free_sd_ctx(context); save(); return 0;
}
