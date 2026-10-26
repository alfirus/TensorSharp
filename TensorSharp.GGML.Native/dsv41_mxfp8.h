// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// TensorSharp-owned support for published MXFP8 Engram shards. Upstream ggml
// does not define these GGUF types. The file remains untouched; ggml sees only
// a host byte tensor, while this reader retains the logical FP8 row geometry.
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace tsg_dsv41_mxfp8 {

inline float e4m3(uint8_t code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const float magnitude = exponent == 15 && mantissa == 7
        ? std::numeric_limits<float>::quiet_NaN()
        : exponent == 0 ? std::ldexp(float(mantissa), -9)
        : std::ldexp(1.0f + float(mantissa) / 8.0f, exponent - 7);
    return code & 128 ? -magnitude : magnitude;
}

// OCP MXFP8: an E8M0 power-of-two scale followed by 32 E4M3FN values.
// Arithmetic follows the published FP8 format, including subnormal values.
inline void dequantize(const void * source, float * output, size_t values) {
    if (!source || !output || !values || values % 32)
        throw std::runtime_error("MXFP8 Engram rows must contain complete 32-value blocks");
    const auto * bytes = static_cast<const uint8_t *>(source);
    for (size_t first = 0; first < values; first += 32, bytes += 33) {
        if (bytes[0] == 255) throw std::runtime_error("MXFP8 Engram has an invalid E8M0 scale");
        const float scale = std::ldexp(1.0f, int(bytes[0]) - 127);
        for (size_t i = 0; i < 32; ++i) {
            const float value = e4m3(bytes[i + 1]) * scale;
            if (!std::isfinite(value)) throw std::runtime_error("MXFP8 Engram has a nonfinite weight");
            output[first + i] = value;
        }
    }
}

struct source {
    uint64_t offset = 0, bytes = 0, row_values = 0, rows = 0;
    uint32_t type = 0;
};

class header_reader {
    std::ifstream stream;
    uint64_t position = 0;
    static constexpr uint64_t limit = 16 * 1024 * 1024;
public:
    explicit header_reader(const std::string & path) : stream(std::filesystem::u8path(path), std::ios::binary) {
        if (!stream) throw std::runtime_error("Cannot open GGUF header: " + path);
    }
    uint64_t tell() const { return position; }
    uint64_t integer(unsigned bytes) {
        if (bytes > 8 || position > limit - bytes) throw std::runtime_error("Invalid MXFP8 GGUF header bounds");
        uint64_t value = 0;
        for (unsigned i = 0; i < bytes; ++i) {
            const int byte = stream.get();
            if (byte == std::char_traits<char>::eof()) throw std::runtime_error("Truncated MXFP8 GGUF header");
            value |= uint64_t(uint8_t(byte)) << (8 * i);
        }
        position += bytes;
        return value;
    }
    void skip(uint64_t bytes) {
        if (bytes > limit || position > limit - bytes) throw std::runtime_error("MXFP8 GGUF header exceeds 16 MiB");
        stream.seekg(std::streamoff(bytes), std::ios::cur);
        if (!stream) throw std::runtime_error("Truncated MXFP8 GGUF metadata");
        position += bytes;
    }
    std::string string() {
        const uint64_t count = integer(8);
        if (count > limit || position > limit - count) throw std::runtime_error("Invalid MXFP8 GGUF string");
        std::string value(size_t(count), '\0');
        stream.read(value.data(), std::streamsize(count));
        if (!stream) throw std::runtime_error("Truncated MXFP8 GGUF string");
        position += count;
        return value;
    }
    void value(uint32_t type, unsigned depth = 0) {
        if (depth > 4) throw std::runtime_error("Nested MXFP8 GGUF metadata exceeds limit");
        switch (type) {
            case 0: case 1: case 7: skip(1); break;
            case 2: case 3: skip(2); break;
            case 4: case 5: case 6: skip(4); break;
            case 10: case 11: case 12: skip(8); break;
            case 8: skip(integer(8)); break;
            case 9: {
                const uint32_t subtype = uint32_t(integer(4));
                const uint64_t count = integer(8);
                if (count > limit) throw std::runtime_error("Invalid MXFP8 GGUF array length");
                for (uint64_t i = 0; i < count; ++i) value(subtype, depth + 1);
                break;
            }
            default: throw std::runtime_error("Unknown MXFP8 GGUF metadata type");
        }
    }
};

// Recognize only the publisher's dedicated one-tensor Engram shard. All normal
// GGUFs continue through unchanged upstream gguf_init_from_file. Unknown model
// weight formats are never relabeled as byte tensors.
inline bool read_dedicated(const std::string & path, source & out) {
    header_reader input(path);
    if (input.integer(4) != 0x46554747 || input.integer(4) != 3)
        throw std::runtime_error("Invalid GGUF magic/version");
    const uint64_t tensors = input.integer(8), metadata = input.integer(8);
    if (tensors != 1) return false;
    if (metadata > 100000) throw std::runtime_error("Invalid MXFP8 GGUF metadata count");
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < metadata; ++i) {
        const std::string key = input.string();
        const uint32_t type = uint32_t(input.integer(4));
        if (key == "general.alignment") {
            if (type != 4) throw std::runtime_error("Invalid GGUF alignment type");
            alignment = input.integer(4);
        } else input.value(type);
    }
    const std::string name = input.string();
    const uint64_t rank = input.integer(4);
    if (rank > 4 || !rank) throw std::runtime_error("Invalid MXFP8 GGUF tensor rank");
    uint64_t dimensions[4] = {1,1,1,1};
    for (uint64_t i = 0; i < rank; ++i) dimensions[i] = input.integer(8);
    const uint32_t type = uint32_t(input.integer(4));
    const uint64_t relative = input.integer(8);
    if (name != "per_layer_token_embd.weight" || (type != 43 && type != 67)) return false;
    if (rank != 2 || !dimensions[0] || dimensions[0] % 32 || dimensions[0] > 65536 ||
        !dimensions[1] || dimensions[1] > INT32_MAX || alignment < 1 || alignment > 65536 ||
        (alignment & (alignment - 1))) throw std::runtime_error("Invalid MXFP8 Engram dimensions/alignment");
    const uint64_t start = (input.tell() + alignment - 1) / alignment * alignment;
    const uint64_t row_bytes = dimensions[0] / 32 * 33;
    if (relative % alignment || relative > UINT64_MAX - start ||
        dimensions[1] > UINT64_MAX / row_bytes) throw std::runtime_error("MXFP8 Engram byte range overflows");
    const uint64_t offset = start + relative, bytes = row_bytes * dimensions[1];
    const uint64_t file_bytes = std::filesystem::file_size(std::filesystem::u8path(path));
    if (offset > file_bytes || bytes > file_bytes - offset) throw std::runtime_error("MXFP8 Engram shard is incomplete");
    out = {offset,bytes,dimensions[0],dimensions[1],type};
    return true;
}

} // namespace tsg_dsv41_mxfp8
