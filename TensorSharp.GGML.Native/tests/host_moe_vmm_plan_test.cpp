// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Standalone portable state tests: no GGML/CUDA headers, resources or model.
#include "../ggml_ops_host_moe_vmm_plan.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace tsg::host_moe_vmm;
using State = Owner<32>;
unsigned checks = 0;
void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
template<class F> void rejected(F action, const char* message) {
    bool threw = false; try { action(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
Layout layout(std::size_t capacity) {
    std::array<Projection, 3> specs{};
    const std::size_t stride[3] = {13,23,31}, granule[3] = {16,32,64}, chunk[3] = {64,64,128};
    for (unsigned p = 0; p != 3; ++p)
        specs[p] = {stride[p], capacity ? stride[p]*capacity+5+2*p : 0,
            granule[p], chunk[p], 128, 16384, 16384};
    return Layout::make(capacity, specs);
}

struct Fake final : Adapter {
    struct Ticket { std::size_t bytes; bool committed = false; };
    struct Resource { unsigned pool; std::size_t bytes, offset = 0; Id ticket;
        bool mapped = false, accessible = false; std::vector<unsigned char> data; };
    std::map<Id, Ticket> tickets;
    std::map<Id, Resource> resources;
    std::map<std::string, int> faults;
    Id next = 1;
    std::size_t budget = 65536, expected_reserves = 0, cycle_reserves = 0;
    unsigned calls = 0, creates = 0, drains = 0, refunds = 0;
    bool dirty = false, valid = true;
    bool fault(const char* name) noexcept {
        ++calls; auto it = faults.find(name);
        if (it == faults.end()) return false;
        if (it->second-- > 0) return false;
        faults.erase(it); return true;
    }
    std::size_t charged() const noexcept {
        std::size_t total = 0; for (const auto& item : tickets) total += item.second.bytes; return total;
    }
    void cycle(const Layout& before, const Layout& after) {
        cycle_reserves = expected_reserves = 0;
        for (unsigned p = 0; p != 3; ++p) expected_reserves += after.chunks(p)-before.chunks(p);
    }
    Id reserve(std::size_t bytes) noexcept override {
        if (fault("reserve") || charged() > budget || bytes > budget-charged()) return 0;
        const auto id = next++; tickets.emplace(id, Ticket{bytes}); ++cycle_reserves; return id;
    }
    bool commit(Id ticket) noexcept override {
        if (fault("commit")) return false;
        auto it = tickets.find(ticket);
        auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& r) { return r.second.ticket == ticket; });
        if (it == tickets.end() || it->second.committed || dirty || resource == resources.end()
            || !resource->second.mapped || !resource->second.accessible) { valid = false; return false; }
        it->second.committed = true; return true;
    }
    bool refund(Id ticket) noexcept override {
        if (fault("refund")) return false;
        for (const auto& r : resources) if (r.second.ticket == ticket) { valid = false; return false; }
        if (tickets.erase(ticket) != 1) { valid = false; return false; }
        ++refunds; return true;
    }
    Id create(unsigned pool, std::size_t bytes, Id ticket) noexcept override {
        ++creates;
        if (cycle_reserves != expected_reserves || !tickets.count(ticket)
            || tickets.at(ticket).bytes != bytes) { valid = false; return 0; }
        if (fault("create")) return 0;
        const auto id = next++;
        std::vector<unsigned char> data(bytes);
        for (std::size_t k=0;k!=bytes;++k) data[k]=static_cast<unsigned char>(id^(k*29)^pool);
        resources.emplace(id, Resource{pool,bytes,0,ticket,false,false,std::move(data)});
        return id;
    }
    bool map(unsigned pool, std::size_t offset, Id id, std::size_t bytes) noexcept override {
        if (fault("map")) return false;
        auto it = resources.find(id);
        if (it == resources.end() || it->second.pool != pool || it->second.bytes != bytes || it->second.mapped)
            { valid = false; return false; }
        for (const auto& item : resources) if (item.second.mapped && item.second.pool == pool
            && offset < item.second.offset+item.second.bytes && item.second.offset < offset+bytes)
            { valid = false; return false; }
        it->second.offset = offset; it->second.mapped = true; return true;
    }
    bool access(unsigned pool, std::size_t offset, std::size_t bytes) noexcept override {
        if (fault("access")) return false;
        for (auto& item : resources) if (item.second.pool == pool && item.second.offset == offset
            && item.second.bytes == bytes && item.second.mapped) {
            item.second.accessible = true; dirty = true; return true;
        }
        valid = false; return false;
    }
    bool drain() noexcept override {
        ++drains; if (fault("drain")) return false;
        dirty = false; return true;
    }
    bool unmap(unsigned pool, std::size_t offset, std::size_t bytes) noexcept override {
        if (dirty) { valid = false; return false; }
        if (fault("unmap")) return false;
        for (auto& item : resources) if (item.second.pool == pool && item.second.offset == offset
            && item.second.bytes == bytes && item.second.mapped) {
            item.second.mapped = item.second.accessible = false; return true;
        }
        valid = false; return false;
    }
    bool release(Id id) noexcept override {
        if (fault("release")) return false;
        auto it = resources.find(id);
        if (it == resources.end() || it->second.mapped || dirty) { valid = false; return false; }
        resources.erase(it); return true;
    }
    void reconcile(const State& owner) const {
        const auto u = owner.usage();
        std::size_t pending = 0, committed = 0, mapped = 0, created = 0;
        for (const auto& t : tickets) (t.second.committed ? committed : pending) += t.second.bytes;
        for (const auto& r : resources) (r.second.mapped ? mapped : created) += r.second.bytes;
        require(valid && pending == u.pending && committed == u.committed && mapped == u.mapped
            && created == u.created_unmapped && tickets.size() == u.records,
            "Owner differs from independent actual ticket/resource inventory");
        require(u.pending+u.committed == u.mapped+u.created_unmapped+u.ticket_only,
            "A physical resource or failed-refund ticket lost ownership");
    }
    std::vector<unsigned char> bytes(unsigned pool, std::size_t count) const {
        std::vector<unsigned char> result(count);
        for (const auto& r : resources) if (r.second.pool == pool && r.second.mapped)
            for (std::size_t k = 0; k < r.second.bytes && r.second.offset+k < count; ++k)
                result[r.second.offset+k] = r.second.data[k];
        return result;
    }
    std::vector<std::uint64_t> inventory() const {
        std::vector<std::uint64_t> result{tickets.size(),resources.size()};
        for (const auto& t:tickets) {
            result.push_back(t.first); result.push_back(t.second.bytes); result.push_back(t.second.committed);
        }
        for (const auto& r:resources) {
            const auto& v=r.second;
            for (auto value:{r.first,static_cast<Id>(v.pool),static_cast<Id>(v.offset),static_cast<Id>(v.bytes),
                    v.ticket,static_cast<Id>(v.mapped),static_cast<Id>(v.accessible)}) result.push_back(value);
            result.insert(result.end(),v.data.begin(),v.data.end());
        }
        return result;
    }
};

void arithmetic() {
    const auto a = layout(4), b = layout(9);
    require(a.physical(0) == 64 && a.physical(1) == 128 && a.physical(2) == 256,
        "Independent projection allocation padding/granules were replaced by aggregate rounding");
    std::size_t expected = 0;
    for (unsigned p = 0; p != 3; ++p) expected += b.physical(p)-a.physical(p);
    require(a.growth_delta(b) == expected && a.total()+expected == b.total(), "Growth delta is not immutable additive ownership");
    rejected([&] { b.growth_delta(a); }, "Growth accepted a future shrink credit");
    for (unsigned mutation = 0; mutation != 8; ++mutation) {
        auto specs = std::array<Projection,3>{a.projection(0),a.projection(1),a.projection(2)};
        auto& s = specs[1];
        switch (mutation) {
            case 0: s.stride=0; break; case 1: s.allocation=1; break;
            case 2: s.granule=0; break; case 3: s.chunk=65; break;
            case 4: s.maximum_chunk=32; break; case 5: s.maximum_buffer=64; break;
            case 6: s.reserved_va=64; break; case 7: s.reserved_va=16385; break;
        }
        rejected([&] { Layout::make(4,specs); }, "Invalid projection limit reached an adapter");
    }
    rejected([] { Layout::add(Layout::limit(),1); }, "Signed charge sum overflow accepted");
    rejected([] { Layout::multiply(Layout::limit(),2); }, "Signed charge product overflow accepted");
    rejected([] { Layout::multiply((std::numeric_limits<std::size_t>::max)(),0); }, "Out-of-range zero product accepted");
    rejected([] { Layout::round(Layout::limit(),128); }, "Rounded charge overflow accepted");
    rejected([] {
        Projection p{1,Layout::limit()/2+1,1,1,1,Layout::limit(),Layout::limit()};
        Layout::make(1,{p,p,p});
    }, "Three individually valid projections overflowed the aggregate charge");
    const int occupied[] = {3,-1,7,2,-1,-1,-1,-1,-1}, tail[] = {3,-1,7,2,6,-1,-1,-1,-1};
    require(EmptyTail::prove(9,4,7,occupied,9).has_value() && !EmptyTail::prove(9,4,7,tail,9),
        "Empty-tail proof discarded an occupied source slot");
    require(!EmptyTail::prove(9,4,0,occupied,9) && !EmptyTail::prove(9,4,7,nullptr,9)
        && !EmptyTail::prove(9,4,7,occupied,3) && !EmptyTail::prove(9,4,7,occupied,8),
        "Invalid occupancy/generation certified shrink");
    auto invalid=std::array<int,9>{3,-1,7,-2,-1,-1,-1,-1,-1};
    require(!EmptyTail::prove(9,4,7,invalid.data(),invalid.size()),"Invalid empty-slot sentinel certified shrink");
}

void growth_faults() {
    for (const auto* operation : {"reserve","create","map","access","commit","drain"}) {
        const int fault_points=std::string(operation)=="drain" ? 1 : 4;
        for (int index=0;index!=fault_points;++index) {
            Fake fake; State owner(fake); const auto a=layout(4), b=layout(9);
            fake.cycle(owner.current(),a); require(owner.grow(a,fake)==Result::published,"Initial owner failed");
            std::array<std::vector<unsigned char>,3> old;
            for (unsigned p=0;p!=3;++p) old[p]=fake.bytes(p,a.physical(p));
            fake.cycle(a,b); fake.faults[operation]=index;
            const auto result = owner.grow(b,fake); fake.reconcile(owner);
            require(result==Result::unchanged || result==Result::quarantined,"Injected failure published capacity");
            require(owner.current().capacity()==a.capacity() && fake.faults.empty(),"Failure hook was not reached or published geometry");
            if (result==Result::quarantined) {
                require(!owner.usable() && owner.usage().quarantined>0,"Failed drain dropped its owner");
                require(owner.retry(fake)==Result::unchanged,"Growth quarantine did not retry cleanly");
            }
            fake.reconcile(owner);
            for (unsigned p=0;p!=3;++p) require(fake.bytes(p,a.physical(p))==old[p],"Growth/rollback changed retained prefix bytes");
            require(owner.close(fake) && owner.clean(),"Growth fault teardown failed"); fake.reconcile(owner);
        }
    }
    Fake fake; State owner(fake); const auto a=layout(4),b=layout(9);
    fake.cycle(owner.current(),a); require(owner.grow(a,fake)==Result::published,"Initial state failed");
    fake.cycle(a,b); fake.faults["reserve"]=2; fake.faults["refund"]=0;
    require(owner.grow(b,fake)==Result::quarantined && owner.usage().ticket_only>0
        && owner.usage().quarantined>0,"Failed ticket-only refund was silently discarded");
    fake.reconcile(owner);
    const auto calls=fake.calls;
    require(owner.grow(b,fake)==Result::refused && fake.calls==calls,"Quarantine admitted another transaction");
    require(owner.retry(fake)==Result::unchanged && owner.close(fake),"Ticket-only quarantine could not retire");
    fake.reconcile(owner);
    for (const auto* operation : {"create","map","access"}) {
        Fake driver; State held(driver); driver.cycle(held.current(),a);
        require(held.grow(a,driver)==Result::published,"Rollback fence fixture initial state failed");
        driver.cycle(a,b);
        driver.faults[operation]=std::string(operation)=="create" ? 2 : 0;
        driver.faults[std::string(operation)=="create" ? "drain" : std::string(operation)=="map" ? "release" : "unmap"]=0;
        const auto prior_refunds=driver.refunds;
        require(held.grow(b,driver)==Result::quarantined && !held.usable()
            && held.current().capacity()==a.capacity(),"Rollback failure lost the old published geometry or quarantine");
        const auto usage=held.usage();
        if (std::string(operation)=="create")
            require(driver.refunds==prior_refunds && usage.mapped>a.total(),"Failed rollback drain unmapped/refunded live queued storage");
        if (std::string(operation)=="map") require(usage.created_unmapped>0 && usage.pending>0,
            "Failed creation rollback forgot its unmapped handle/pending ticket");
        driver.reconcile(held);
        require(held.retry(driver)==Result::unchanged && held.close(driver),"Rollback quarantine retry failed");
        driver.reconcile(held);
    }
}

void shrink_faults_and_limits() {
    for (const auto* operation : {"drain","unmap","release","refund"}) {
        Fake fake; State owner(fake); const auto large=layout(9),small=layout(4);
        fake.cycle(owner.current(),large); require(owner.grow(large,fake)==Result::published,"Initial grow failed");
        std::array<std::vector<unsigned char>,3> retained;
        for (unsigned p=0;p!=3;++p) retained[p]=fake.bytes(p,small.physical(p));
        const int occupied[]={1,-1,2,3,-1,-1,-1,-1,-1}; const auto proof=EmptyTail::prove(9,4,91,occupied,9);
        const auto calls=fake.calls;
        require(owner.shrink(small,*proof,92,fake)==Result::refused && fake.calls==calls,
            "Stale slot generation released physical memory");
        fake.faults[operation]=0;
        require(owner.shrink(small,*proof,91,fake)==Result::quarantined && !owner.usable(),
            "Partial shrink remained usable or hid failure");
        fake.reconcile(owner);
        const auto u=owner.usage();
        if (std::string(operation)=="release") require(u.created_unmapped>0,"Failed release forgot created-unmapped handle");
        if (std::string(operation)=="refund") require(u.ticket_only>0,"Failed committed refund forgot its ticket");
        require(owner.retry(fake)==Result::published && owner.current().capacity()==4 && owner.usable(),
            "Shrink retry failed to finish the same empty-tail transaction");
        for (unsigned p=0;p!=3;++p) require(fake.bytes(p,small.physical(p))==retained[p],"Shrink changed surviving prefix bytes");
        require(owner.usage().committed==small.total(),"Shrink did not charge actual rounded remaining pools");
        fake.reconcile(owner);
        fake.faults["refund"]=0;
        require(!owner.close(fake) && owner.usage().ticket_only>0,"Close auto-refunded a refused ticket");
        fake.reconcile(owner);
        require(owner.close(fake) && fake.tickets.empty() && fake.resources.empty(),"Checked close did not release everything");
    }
    Fake fake; State owner(fake); const auto a=layout(4),b=layout(9);
    fake.cycle(owner.current(),a); require(owner.grow(a,fake)==Result::published,"Initial limit state failed");
    fake.cycle(a,b); fake.budget=a.total()+a.growth_delta(b)-1; const auto before=fake.creates;
    require(owner.grow(b,fake)==Result::unchanged && fake.creates==before,"Prephysical total reservation refusal created a resource");
    fake.reconcile(owner);
    const auto calls=fake.calls;
    require(owner.grow(layout(200),fake)==Result::refused && fake.calls==calls,"Bounded owner exceeded its record capacity");
    require(owner.close(fake),"Bounded owner cleanup failed"); fake.reconcile(owner);

    Fake same_fake; State same(same_fake); const auto seven=layout(7),six=layout(6);
    same_fake.cycle(same.current(),seven);
    require(same.grow(seven,same_fake)==Result::published && seven.total()==six.total(),
        "Fixture failed to isolate a logical-only shrink within rounded chunks");
    const int low[]={1,-1,-1,-1,-1,3,-1}; const auto proof=EmptyTail::prove(7,6,4,low,7);
    const auto resources=same_fake.resources.size();
    const auto refunds=same_fake.refunds;
    require(same.shrink(six,*proof,4,same_fake)==Result::published && same.current().capacity()==6
        && same_fake.resources.size()==resources && same_fake.refunds==refunds,
        "Logical-only shrink invented physical credit");
    same_fake.reconcile(same);
    auto specs=std::array<Projection,3>{six.projection(0),six.projection(1),six.projection(2)};
    ++specs[0].allocation;
    rejected([&] { same.grow(Layout::make(6,specs),same_fake); },"Same-capacity allocation certificate changed silently");
    require(same.close(same_fake),"Logical-only shrink cleanup failed"); same_fake.reconcile(same);
}

void adapter_identity() {
    Fake a,b; State owner(a),other(b); const auto large=layout(9),small=layout(4),next=layout(12);
    a.cycle(owner.current(),large); b.cycle(other.current(),large);
    require(owner.grow(large,a)==Result::published && other.grow(large,b)==Result::published
        && a.inventory()==b.inventory(),"Two-adapter fixture did not create colliding opaque ID spaces");
    const int slots[]={1,-1,2,3,-1,-1,-1,-1,-1}; const auto proof=EmptyTail::prove(9,4,1,slots,9);
    const auto held_a=a.inventory(),held_b=b.inventory(); const auto actions_a=a.calls,actions_b=b.calls;
    require(owner.grow(next,b)==Result::refused && owner.shrink(small,*proof,1,b)==Result::refused
        && owner.retry(b)==Result::refused && !owner.close(b),"Wrong adapter accepted an opaque ownership operation");
    require(a.calls==actions_a && b.calls==actions_b && a.inventory()==held_a && b.inventory()==held_b,
        "Adapter mismatch touched a colliding resource or charge");
    a.cycle(large,next); a.faults["create"]=0; a.faults["refund"]=0;
    require(owner.grow(next,a)==Result::quarantined,"Quarantine adapter fixture failed");
    const auto quarantined=a.inventory(); const auto after_a=a.calls;
    require(owner.retry(b)==Result::refused && !owner.close(b) && owner.grow(next,b)==Result::refused
        && a.calls==after_a && b.calls==actions_b && a.inventory()==quarantined && b.inventory()==held_b,
        "Quarantine borrowed another adapter's lifetime or ticket identity");
    require(owner.retry(a)==Result::unchanged && owner.close(a) && other.close(b),"Bound adapter cleanup failed");
    a.reconcile(owner); b.reconcile(other);
}
} // namespace

int main() {
    static_assert(!std::is_copy_constructible<State>::value && !std::is_move_constructible<State>::value,
        "Opaque physical/ticket ownership may not be copied or silently moved");
    static_assert(State::metadata_bytes()<16384 && State::record_capacity()==96,
        "Portable metadata must be bounded independently of adapter charge owners");
    try {
        arithmetic(); growth_faults(); shrink_faults_and_limits(); adapter_identity();
        std::printf("PASS portable VMM layout/ownership checks=%u; no CUDA/GGML resources or model; metadata=%zu adapter_records=%zu\n",
            checks,State::metadata_bytes(),State::record_capacity()); return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"FAIL portable VMM state: %s (checks=%u)\n",error.what(),checks); return 1;
    }
}
