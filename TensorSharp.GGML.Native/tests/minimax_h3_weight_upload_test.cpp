// Portable protocol test for H3's bounded automatic CUDA weight prefault.
// Backend calls are mocked; CUDA quality/performance require device tests.
// c++ -std=c++17 -pthread -I ExternalProjects/ggml/include \
//   TensorSharp.GGML.Native/tests/minimax_h3_weight_upload_test.cpp -o <test>
#include "../ggml_h3_weight_upload.h"

#include <stdexcept>
#include <utility>

struct ggml_backend { const char * name; };

namespace {
ggml_backend cuda{"CUDA0"}, cpu{"CPU"};
const std::thread::id owner_thread = std::this_thread::get_id();
std::vector<std::pair<size_t, size_t>> transfers;

void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

void environment(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

struct Fixture {
    struct Binding { ggml_tensor * t; const void * data; size_t bytes; };
    std::vector<std::vector<unsigned char>> sources;
    std::vector<Binding> bindings;
    size_t bytes = 0, reads = 0;
    unsigned int checksum = 0;

    Fixture() {
        sources.reserve(10);
        for (size_t index = 0; index < 10; ++index) {
            const size_t count = (1u << 20) + (index == 0 ? 0 : index == 1 ? 1 : index * 37);
            const size_t prefix = index + 17;
            sources.emplace_back(prefix + count);
            auto & storage = sources.back();
            for (size_t i = 0; i < storage.size(); ++i)
                storage[i] = static_cast<unsigned char>((i * 31 + index * 17) & 255);
            auto * data = storage.data() + prefix;
            bindings.push_back({nullptr, data, count});
            bytes += count;
            for (size_t offset = 0; offset < count; offset += 4096) {
                checksum ^= data[offset];
                ++reads;
            }
            if ((count - 1) % 4096 != 0) {
                checksum ^= data[count - 1];
                ++reads;
            }
        }
        std::reverse(bindings.begin(), bindings.end());
        bindings.push_back({nullptr, nullptr, 1u << 20});
        bindings.push_back({nullptr, sources[0].data(), 17});
    }
};

void verify_prefault(const Fixture & fixture, const char * mode, bool use_cuda,
        size_t capacity, bool active, const char * workers = "8", size_t expected_workers = 8) {
    const auto snapshot = fixture.sources;
    const auto original_bindings = fixture.bindings;
    environment("TS_H3_UPLOAD_PREFAULT", mode);
    environment("TS_H3_UPLOAD_WORKERS", workers);
    environment("TS_H3_UPLOAD_PROFILE", nullptr);
    transfers.clear();
    {
        tsg::H3WeightUpload upload(use_cuda ? &cuda : &cpu, "prefault-protocol-test", capacity);
        upload.prefault(fixture.bindings);
        check(upload.prefault_bytes() == (active ? fixture.bytes : 0), "prefault budget/source selection mismatch");
        check(upload.prefault_reads() == (active ? fixture.reads : 0), "prefault page or tail read mismatch");
        check(upload.prefault_checksum() == (active ? fixture.checksum : 0), "prefault sampled bytes mismatch");
        check(upload.prefault_workers() == (active ? expected_workers : 0), "prefault worker override/bound mismatch");
        check(transfers.empty(), "prefault invoked a device upload");
    }
    check(snapshot == fixture.sources, "prefault modified source bytes");
    for (size_t i = 0; i < fixture.bindings.size(); ++i)
        check(fixture.bindings[i].data == original_bindings[i].data && fixture.bindings[i].bytes == original_bindings[i].bytes,
            "prefault reordered caller bindings");
}

void verify_upload(ggml_backend_t backend, size_t bytes) {
    std::vector<char> source(bytes), destination(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) source[i] = static_cast<char>((i * 113 + i / 17) & 255);
    ggml_tensor tensor{};
    tensor.data = destination.data();
    transfers.clear();
    {
        tsg::H3WeightUpload upload(backend, "upload-protocol-test", 0);
        upload.set(&tensor, source.data(), source.size());
        upload.finish();
    }
    check(source == destination, "upload changed bytes");
    check(transfers.size() == 1 && transfers[0].first == 0 && transfers[0].second == bytes,
        "upload changed the original tensor_set range");
}

void verify_limit_parser() {
    size_t bytes = 0;
    check(tsg::h3_upload_detail::parse_memory_limit("max", bytes) && bytes == std::numeric_limits<size_t>::max(), "unlimited cgroup v2 token");
    check(tsg::h3_upload_detail::parse_memory_limit("1073741824", bytes) && bytes == (1u << 30), "finite cgroup token");
    check(tsg::h3_upload_detail::parse_memory_limit("9223372036854771712", bytes), "cgroup v1 unlimited sentinel");
    check(tsg::h3_upload_detail::parse_memory_limit("0", bytes) && bytes == 0, "zero cgroup capacity");
    for (const char * invalid : {"", "-1", "+1", "1junk", "maxjunk", "18446744073709551616"})
        check(!tsg::h3_upload_detail::parse_memory_limit(invalid, bytes), "invalid/overflow cgroup token accepted");
    size_t capacity = 1u << 30;
    tsg::h3_upload_detail::apply_memory_limit("__tensorsharp_absent_memory_limit__/memory.max", capacity);
    check(capacity == (1u << 30), "missing cgroup path discarded known capacity");
#if defined(__unix__) || defined(__APPLE__)
    // A directory cannot supply a numeric limit; its read failure is unknown.
    tsg::h3_upload_detail::apply_memory_limit("/", capacity);
    check(capacity == 0, "unreadable limit retained unrestricted host capacity");
    capacity = 1u << 30;
    // ENAMETOOLONG exercises the non-missing fopen error branch even as root,
    // unlike permissions-based EACCES tests that privileged runners bypass.
    tsg::h3_upload_detail::apply_memory_limit(std::string(8192, 'x'), capacity);
    check(capacity == 0, "non-missing limit open failure retained host capacity");
#endif
}
} // namespace

extern "C" {
const char * ggml_backend_name(ggml_backend_t backend) {
    check(std::this_thread::get_id() == owner_thread, "worker queried a backend name");
    return backend->name;
}
void ggml_backend_tensor_set(ggml_tensor * tensor, const void * source, size_t offset, size_t size) {
    check(std::this_thread::get_id() == owner_thread, "prefault worker invoked a backend upload");
    transfers.emplace_back(offset, size);
    std::memcpy(static_cast<char *>(tensor->data) + offset, source, size);
}
} // extern "C"

int main() {
    verify_limit_parser();
    const Fixture fixture;
    verify_prefault(fixture, nullptr, true, fixture.bytes * 2, true);             // exact half capacity
    verify_prefault(fixture, nullptr, true, fixture.bytes * 2 - 1, false);        // just over half
    verify_prefault(fixture, nullptr, true, 0, false);                           // unknown capacity
    verify_prefault(fixture, "0", true, fixture.bytes * 4, false);               // explicit disable
    verify_prefault(fixture, "1", true, 0, true);                                // force unknown capacity
    verify_prefault(fixture, "1", true, 1, true);                                // force insufficient capacity
    verify_prefault(fixture, "invalid", true, fixture.bytes * 2, true);          // invalid mode retains auto
    verify_prefault(fixture, nullptr, false, fixture.bytes * 4, false);           // CPU stays unchanged
    verify_prefault(fixture, "1", false, fixture.bytes * 4, false);              // force is CUDA-only
    verify_prefault(fixture, nullptr, true, fixture.bytes * 2, true, "1", 1);
    verify_prefault(fixture, nullptr, true, fixture.bytes * 2, true, "99", 8);     // invalid worker count
    verify_prefault(fixture, nullptr, true, fixture.bytes * 2, true, "0", 8);
    environment("TS_H3_UPLOAD_PREFAULT", nullptr);
    {
        std::vector<Fixture::Binding> small{{nullptr, fixture.sources[0].data(), 17}};
        tsg::H3WeightUpload upload(&cuda, "small-ranges", 0);
        upload.prefault(small);
        check(upload.prefault_reads() == 0 && std::strcmp(upload.prefault_decision(), "no-large-ranges") == 0, "small binding warmed");
    }
    {
        std::vector<Fixture::Binding> overflow{{nullptr, fixture.sources[0].data(), std::numeric_limits<size_t>::max()},
            {nullptr, fixture.sources[0].data(), 1u << 20}};
        tsg::H3WeightUpload upload(&cuda, "overflow-ranges", 0);
        upload.prefault(overflow);
        check(upload.prefault_reads() == 0 && std::strcmp(upload.prefault_decision(), "range-overflow") == 0, "aggregate overflow read source");
    }
    verify_upload(&cuda, 2 * (1u << 20) + 37);
    verify_upload(&cuda, 17);
    verify_upload(&cpu, 2 * (1u << 20) + 37);
    std::puts("H3 upload protocol passed: auto half-capacity budget, unknown/forced/disabled modes, CUDA-only warming, bounded workers, exact immutable pages/tails, original calling-thread uploads and limit parsing");
    return 0;
}
