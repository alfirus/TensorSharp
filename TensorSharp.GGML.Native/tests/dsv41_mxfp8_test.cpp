// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../dsv41_mxfp8.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

namespace {
int checks = 0;
void check(bool value, const char * message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void refused(F action, const char * message) {
    bool failed = false;
    try { action(); } catch (const std::exception &) { failed = true; }
    check(failed, message);
}
void integer(std::vector<uint8_t> & bytes, uint64_t value, int count) {
    for (int i = 0; i < count; ++i) bytes.push_back(uint8_t(value >> (8 * i)));
}
void string(std::vector<uint8_t> & bytes, const std::string & value) {
    integer(bytes,value.size(),8); bytes.insert(bytes.end(),value.begin(),value.end());
}
void write(const std::filesystem::path & path, uint32_t type, uint64_t width = 32, int payload = 66) {
    std::vector<uint8_t> bytes;
    integer(bytes,0x46554747,4); integer(bytes,3,4); integer(bytes,1,8); integer(bytes,1,8);
    string(bytes,"general.alignment"); integer(bytes,4,4); integer(bytes,32,4);
    string(bytes,"per_layer_token_embd.weight"); integer(bytes,2,4);
    integer(bytes,width,8); integer(bytes,2,8); integer(bytes,type,4); integer(bytes,0,8);
    bytes.resize((bytes.size()+31)/32*32 + payload,0);
    std::ofstream output(path,std::ios::binary); output.write((const char *)bytes.data(),bytes.size());
}
}

int main() {
    try {
        for (int code = 0; code < 256; ++code) {
            const int sign = code & 128 ? -1 : 1, exponent = (code >> 3) & 15, mantissa = code & 7;
            const float actual = tsg_dsv41_mxfp8::e4m3(uint8_t(code));
            if (exponent == 15 && mantissa == 7) check(std::isnan(actual),"E4M3 NaN");
            else {
                const double expected = sign * (exponent == 0 ? mantissa * std::pow(2.0,-9)
                    : (8+mantissa) * std::pow(2.0,exponent-10));
                check(actual == float(expected),"E4M3 value");
                if (expected == 0) check(std::signbit(actual) == (sign < 0),"E4M3 signed zero");
            }
        }
        std::array<uint8_t,66> row{};
        std::array<float,64> values{};
        for (int exponent : {0,1,64,126,127,128,192,240}) {
            row[0] = uint8_t(exponent); row[33] = uint8_t(exponent);
            for (int i = 0; i < 32; ++i) { row[i+1] = uint8_t(i); row[i+34] = uint8_t(128+i); }
            tsg_dsv41_mxfp8::dequantize(row.data(),values.data(),64);
            for (int i = 0; i < 64; ++i) {
                const auto code = row[(i/32)*33 + i%32 + 1];
                check(values[i] == std::ldexp(tsg_dsv41_mxfp8::e4m3(code),exponent-127),"MXFP8 block scale/stride");
            }
        }
        refused([&]{ tsg_dsv41_mxfp8::dequantize(row.data(),values.data(),63); },"partial block refused");
        row[0] = 255;
        refused([&]{ tsg_dsv41_mxfp8::dequantize(row.data(),values.data(),64); },"NaN scale refused");
        row[0] = 127; row[1] = 127;
        refused([&]{ tsg_dsv41_mxfp8::dequantize(row.data(),values.data(),64); },"NaN weight refused");
        const auto path = std::filesystem::temp_directory_path() /
            ("tensorsharp-mxfp8-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
        tsg_dsv41_mxfp8::source source;
        for (uint32_t type : {43u,67u}) {
            write(path,type);
            check(tsg_dsv41_mxfp8::read_dedicated(path.u8string(),source),"dedicated shard recognized");
            check(source.type == type && source.row_values == 32 && source.rows == 2 && source.bytes == 66,
                "logical dimensions and packed byte size");
        }
        write(path,14);
        check(!tsg_dsv41_mxfp8::read_dedicated(path.u8string(),source),"known ggml type left to upstream");
        write(path,43,31);
        refused([&]{tsg_dsv41_mxfp8::read_dedicated(path.u8string(),source);},"partial block header refused");
        write(path,43,32,65);
        refused([&]{tsg_dsv41_mxfp8::read_dedicated(path.u8string(),source);},"truncated shard refused");
        std::filesystem::remove(path);
        std::cout << checks << " MXFP8 numeric/header checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
