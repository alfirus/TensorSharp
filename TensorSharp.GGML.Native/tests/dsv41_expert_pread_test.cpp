// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../dsv41_expert_pread.h"
#include "../dsv41_engram_io.h"
#include <iostream>
#include <vector>

int main() {
#if defined(_WIN32)
    std::cout << "SKIP: selected-expert pread requires POSIX\n";
    return 77;
#else
    int checks = 0, fd = -1;
    char path[] = "/tmp/tensorsharp-expert-pread-XXXXXX";
    auto check = [&](bool value,const char * message) { ++checks; if (!value) throw std::runtime_error(message); };
    try {
        fd = ::mkstemp(path);
        check(fd >= 0,"temporary file creation");
        std::vector<uint8_t> original(256*1024+37);
        for (size_t i=0;i<original.size();++i) original[i]=uint8_t((i*37+i/257)%251);
        check(::write(fd,original.data(),original.size()) == ssize_t(original.size()),"fixture write");
        auto source = tsg_dsv41_pread::file::open(path);
        tsg_dsv41_pread::file_range ranges[] = {{source,37,65533},{source,131077,65536},{source,17,117}};
        source.reset(); // the ranges keep the one persistent descriptor alive
        check(::unlink(path)==0,"unlink after open");
        tsg_dsv41::engram_io_pool pool(3);
        std::vector<std::vector<uint8_t>> values(3);
        pool.run(3,[&](size_t i) { values[i].resize(ranges[i].bytes); ranges[i].read_exact(values[i].data(),values[i].size(),0); });
        for (size_t i=0;i<3;++i)
            check(std::equal(values[i].begin(),values[i].end(),original.begin()+ranges[i].offset),"parallel pread byte identity at distinct offsets");
        uint8_t scratch[7];
        bool refused=false;
        try { ranges[0].read_exact(scratch,sizeof(scratch),ranges[0].bytes-3); } catch (const std::exception &) { refused=true; }
        check(refused,"tensor range overflow refused");
        check(::ftruncate(fd,131080)==0,"truncate after persistent open");
        refused=false;
        try { ranges[1].read_exact(values[1].data(),values[1].size(),0); } catch (const std::exception & e) {
            refused=std::string(e.what()).find("unexpected EOF")!=std::string::npos;
        }
        check(refused,"partial read followed by EOF fails explicitly");
        check(::pwrite(fd,original.data(),original.size(),0)==ssize_t(original.size()),"restore truncated fixture");
        ranges[1].read_exact(values[1].data(),values[1].size(),0);
        check(std::equal(values[1].begin(),values[1].end(),original.begin()+ranges[1].offset),"same descriptor recovers exact bytes after failure");
        ::close(fd);fd=-1;
        ranges[2].read_exact(scratch,sizeof(scratch),3);
        check(std::equal(scratch,scratch+sizeof(scratch),original.begin()+20),"reader lifetime outlives writer and pathname");
        check(!tsg_dsv41_pread::file_range{ranges[0].source,UINT64_MAX,8}.valid(8),"file offset overflow refused");
        std::cout << "PASS selected-expert pread checks=" << checks << "\n";
        return 0;
    } catch (const std::exception & error) {
        if (fd>=0) ::close(fd);
        ::unlink(path);
        std::cerr << "FAIL selected-expert pread: " << error.what() << "\n";
        return 1;
    }
#endif
}
