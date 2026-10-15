// Generates an independent fixture using the unchanged sd.cpp Philox header.
// Compile with -I <clean stable-diffusion.cpp checkout>/src.
#include "core/rng_philox.hpp"
#include <iomanip>
#include <iostream>
#include <limits>

static void array(const std::vector<float>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

int main() {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::cout << "{\"cases\":[";
    bool first = true;
    for (uint64_t seed : {uint64_t(0), uint64_t(42), uint64_t(4294967338ULL), UINT64_MAX}) {
        for (auto geometry : {std::array<int, 4>{1, 2, 2, 1}, std::array<int, 4>{3, 6, 4, 5}}) {
            const int frames = geometry[0], height = geometry[1], width = geometry[2], audio_frames = geometry[3];
            const int volume = frames * height * width, video_length = 24 * volume, audio_length = 32 * 2 * audio_frames;
            const int padded_length = volume * (24 + (audio_length + volume - 1) / volume);
            PhiloxRNG rng(seed);
            const auto packed = rng.randn(padded_length);
            std::vector<float> video(video_length), audio(audio_length), conditioned(video_length);
            for (int index = 0; index < video_length; ++index) {
                const int token = index / 96, within = index % 96, channel = within / 4, patch = within % 4;
                const int time = token / ((height / 2) * (width / 2));
                const int position = token % ((height / 2) * (width / 2));
                const int y = 2 * (position / (width / 2)) + patch / 2;
                const int x = 2 * (position % (width / 2)) + patch % 2;
                video[index] = packed[((channel * frames + time) * height + y) * width + x];
            }
            for (int index = 0; index < audio_length; ++index) {
                const int token = index / 32, dimension = index % 32;
                audio[index] = packed[video_length + (dimension * 2 + token / audio_frames) * audio_frames + token % audio_frames];
            }
            PhiloxRNG condition_rng(seed);
            const auto visual_noise = condition_rng.randn(video_length);
            for (int index = 0; index < video_length; ++index) {
                const float clean = float(index % 19 - 9) * .125f;
                const float noise = visual_noise[(index % 24) * volume + index / 24];
                conditioned[index] = clean * .999f + noise * (1.f - .999f);
            }
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"seed\":\"" << seed << "\",\"frames\":" << frames << ",\"height\":" << height
                      << ",\"width\":" << width << ",\"audioFrames\":" << audio_frames
                      << ",\"packedLength\":" << padded_length << ",\"video\":";
            array(video);
            std::cout << ",\"audio\":"; array(audio);
            std::cout << ",\"conditioned\":"; array(conditioned);
            std::cout << '}';
        }
    }
    std::cout << "]}\n";
}
