#include "server-cache.h"
#include "server-task.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/resource.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/statvfs.h>
#include <thread>
#include <chrono>
#include <atomic>

namespace fs = std::filesystem;
static void check(bool ok, const char * msg) { if (!ok) { throw std::runtime_error(msg); } }
struct fixture {
    std::string path;
    int64_t time = 1700000000;
    server_cache_disk_config c;
    fixture() {
        char name[] = "/tmp/h07-policy-XXXXXX";
        path = mkdtemp(name);
        c.path = path; c.identity = std::string(64, 'a'); c.max_bytes = 100000;
        c.max_state_bytes = 12000; c.write_bytes = 100000; c.free_bytes = 0;
        c.min_tokens = 8; c.max_tokens = 128; c.ttl_seconds = 100;
        c.clock = [this]() { return time; };
    }
    ~fixture() { fs::remove_all(path); }
    size_t files() const {
        size_t n = 0;
        for (auto & p : fs::directory_iterator(path)) { if (p.path().extension() == ".bin") { ++n; } }
        return n;
    }
    fs::path entry() const {
        for (auto & p : fs::directory_iterator(path)) { if (p.path().extension() == ".bin") { return p.path(); } }
        throw std::runtime_error("no entry");
    }
};
static server_prompt_cache_state state(int branch, size_t tokens = 16) {
    server_prompt_cache_state s;
    for (size_t i = 0; i < tokens; ++i) { s.prompt.tokens.push_back(branch + i); }
    s.data.main.assign(4096, branch);
    s.data.drft.assign(128, branch + 1);
    common_prompt_checkpoint cp{};
    cp.n_tokens = tokens - 1; cp.pos_min = 0; cp.pos_max = tokens - 2; cp.id_task = 42;
    cp.data_tgt.assign(1024, 3); cp.data_dft.assign(64, 4); cp.data_spec.assign(32, 5);
    s.prompt.checkpoints.push_back(std::move(cp));
    // Incompressible policy fixtures keep byte caps meaningful with a codec.
    uint32_t entropy = uint32_t(branch) + 1;
    for (auto * bytes : {&s.data.main, &s.data.drft, &s.prompt.checkpoints.front().data_tgt,
            &s.prompt.checkpoints.front().data_dft, &s.prompt.checkpoints.front().data_spec}) {
        for (auto & byte : *bytes) { entropy ^= entropy << 13; entropy ^= entropy >> 17; entropy ^= entropy << 5; byte = entropy; }
    }
    return s;
}
static void expect_disabled(const server_cache_disk_config & c) {
    bool failed = false;
    try { server_cache_disk cache(c); } catch (const std::exception &) { failed = true; }
    check(failed, "unsafe cache was accepted");
}
int main(int argc, char ** argv) {
    {
        server_prompt_cache cache(1, 0);
        auto a = state(10), b = state(100);
        check(cache.alloc(a.prompt, 600000, 0) != nullptr, "RAM initial allocation");
        auto writer = cache.states.front();
        check(cache.alloc(b.prompt, 600000, 0) == nullptr && cache.size() <= cache.limit_size, "writer pin remains charged");
        check(!cache.clear_unpinned(), "cannot grant decoder memory while pinned");
        writer.reset();
        check(cache.clear_unpinned() && cache.alloc(b.prompt, 600000, 0), "released pin permits bounded allocation");
    }
    if (argc == 2) {
        fixture f; f.c.path = std::string(argv[1]) + "/cache";
        check(mkdir(f.c.path.c_str(), 0700) == 0, "ENOSPC cache mkdir");
        server_cache_disk d(f.c);
        struct statvfs space{}; check(statvfs(argv[1], &space) == 0, "ENOSPC statvfs");
        const uint64_t free = space.f_bavail * space.f_frsize;
        int filler = open((std::string(argv[1]) + "/filler").c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        check(filler >= 0 && free > 8192 && posix_fallocate(filler, 0, free - 4096) == 0, "ENOSPC bounded filler");
        close(filler);
        d.save(state(10));
        size_t entries = 0;
        for (auto & p : fs::directory_iterator(f.c.path)) { if (p.path().extension() == ".bin") { ++entries; } }
        check(entries == 0 && !fs::exists(f.c.path + "/.entry.tmp"), "ENOSPC partial cleanup");
        check(fs::exists(f.c.path + "/budget.json"), "ENOSPC reservation lost");
        std::cout << "Near-full filesystem preflight and cleanup gate passed on dedicated bounded tmpfs\n";
        return 0;
    }
    {
        fixture f; auto a = std::make_shared<server_prompt_cache_state>(state(10));
        std::atomic<int> wakes{0};
        server_cache_async d(f.c, [&]() { ++wakes; });
        check(d.save(a), "async save accepted");
        auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (a.use_count() > 1 && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        check(f.files() == 1, "async save published");
        auto phase = d.poll(1, a->prompt.tokens, 0);
        while (phase == server_cache_async::status::pending && std::chrono::steady_clock::now() < until) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); phase = d.poll(1, a->prompt.tokens, 0);
        }
        const int id = 1;
        check(phase == server_cache_async::status::memory && !d.memory_reserved(), "decoder waits for RAM grant");
        d.grant_memory(id);
        while ((phase = d.poll(id, a->prompt.tokens, 0)) != server_cache_async::status::ready && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        check(phase == server_cache_async::status::ready && d.memory_reserved(), "ready state remains charged");
        auto loaded = d.take(id);
        check(loaded && loaded->data.main == a->data.main && !d.memory_reserved(), "async immutable roundtrip");
        phase = d.poll(3, a->prompt.tokens, 0);
        while (phase == server_cache_async::status::pending && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); phase = d.poll(3, a->prompt.tokens, 0); }
        check(phase == server_cache_async::status::memory, "cancel fixture reached allocation gate");
        d.cancel(3);
        while (d.poll(4, a->prompt.tokens, 0) == server_cache_async::status::pending && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        check(d.poll(4, a->prompt.tokens, 0) == server_cache_async::status::memory, "cancel releases reader without retiring record");
        d.cancel(4); check(wakes > 0, "worker wakes scheduler");
    }
    {
        fixture f; auto a = state(10); server_prompt_cache_state out;
        { server_cache_disk d(f.c);
          d.save(state(10, 7)); check(f.files() == 0, "short admission");
          d.save(a); check(f.files() == 1, "save"); d.save(a); check(f.files() == 1, "deduplicate");
          check(!d.load(out, a.prompt.tokens, 16), "live tie must win");
          check(d.load(out, a.prompt.tokens, 0), "load");
          check(out.data.main == a.data.main && out.data.drft == a.data.drft &&
            out.prompt.checkpoints.front().data_spec == a.prompt.checkpoints.front().data_spec &&
            out.prompt.checkpoints.front().id_task == -1, "roundtrip payload and process-local task IDs");
          expect_disabled(f.c); check(d.load(out, a.prompt.tokens, 0), "exclusive lock harmed owner"); }
        { server_cache_disk d(f.c); check(d.load(out, a.prompt.tokens, 0), "restart"); }
        f.c.identity = std::string(64, 'b');
        { server_cache_disk d(f.c); check(!d.load(out, a.prompt.tokens, 0) && f.files() == 0, "identity"); }
    }
    {
        fixture f; auto a = state(10); server_prompt_cache_state out;
        { server_cache_disk d(f.c); d.save(a); f.time += 101;
          check(!d.load(out, a.prompt.tokens, 0) && f.files() == 0, "expiry"); }
    }
    {
        fixture f; auto a = state(10); server_prompt_cache_state out;
        a.captured_at = f.time - 50;
        server_cache_disk d(f.c); d.save(a); f.time += 51;
        check(!d.load(out, a.prompt.tokens, 0), "queue delay must not extend capture TTL");
        d.save(a); check(f.files() == 0, "expired RAM snapshot must not restart TTL");
    }
    {
        fixture f; f.c.write_bytes = 7000; auto a = state(10);
        { server_cache_disk d(f.c); d.save(a); check(f.files() == 1, "budget initial"); }
        { server_cache_disk d(f.c); d.save(state(100)); check(f.files() == 1, "budget restart"); }
        f.time -= 1000;
        { server_cache_disk d(f.c); d.save(state(100)); check(f.files() == 1, "budget rollback"); }
        f.time += 87401;
        { server_cache_disk d(f.c); d.save(state(100)); check(f.files() == 1, "budget renewal"); }
    }
    {
        fixture f; f.c.max_bytes = 78000; auto a = state(10), b = state(100), c = state(200);
        server_cache_disk d(f.c); server_prompt_cache_state out;
        d.save(a); f.time++; d.save(b); f.time++;
        check(d.load(out, a.prompt.tokens, 0), "LRU touch"); f.time++; d.save(c);
        check(f.files() == 2 && !d.load(out, b.prompt.tokens, 0) && d.load(out, a.prompt.tokens, 0), "LRU cap");
        uint64_t total = 0; for (auto & p : fs::directory_iterator(f.path)) { total += p.file_size(); }
        check(total <= f.c.max_bytes, "total directory bytes");
    }
    for (int mode = 0; mode < 3; ++mode) {
        fixture f; auto a = state(10); server_prompt_cache_state out; server_cache_disk d(f.c); d.save(a);
        if (mode == 0) { std::fstream file(f.entry(), std::ios::binary | std::ios::in | std::ios::out); file.seekp(-33, std::ios::end); file.put('\xff'); }
        if (mode == 1) { fs::resize_file(f.entry(), 40); }
        if (mode == 2) { chmod(f.entry().c_str(), 0644); }
        check(!d.load(out, a.prompt.tokens, 0) && f.files() == 0, "invalid payload fallback");
    }
    {
        fixture f; f.c.max_bytes = 78000; auto a = state(10), b = state(100), c = state(200);
        server_prompt_cache_state out;
        { server_cache_disk d(f.c); d.save(a); f.time++; d.save(b); f.time++;
          check(d.load(out, a.prompt.tokens, 0), "restart LRU touch"); }
        { server_cache_disk d(f.c); } // Indexing must not count as using entries.
        f.time++;
        { server_cache_disk d(f.c); d.save(c);
          check(d.load(out, a.prompt.tokens, 0) && !d.load(out, b.prompt.tokens, 0), "LRU survives repeated restart"); }
    }
    {
        fixture f; f.c.free_bytes = 1ull << 62;
        { server_cache_disk d(f.c); d.save(state(10)); check(f.files() == 0, "free-space floor"); }
        f.c.free_bytes = 0; f.c.write_bytes = 7000;
        { server_cache_disk d(f.c); d.save(state(10)); check(f.files() == 0, "failed writes stay charged"); }
    }
    {
        fixture f;
        { std::ofstream(f.path + "/.entry.tmp") << "interrupted"; chmod((f.path + "/.entry.tmp").c_str(), 0600); }
        { server_cache_disk d(f.c); check(!fs::exists(f.path + "/.entry.tmp"), "orphan cleanup"); }
        chmod(f.path.c_str(), 0755); expect_disabled(f.c); chmod(f.path.c_str(), 0700);
        std::ofstream(f.path + "/foreign") << "keep";
        expect_disabled(f.c); check(fs::exists(f.path + "/foreign"), "foreign file preservation");
    }
    {
        fixture f;
        std::ofstream(f.path + "/budget.json") << "truncated"; chmod((f.path + "/budget.json").c_str(), 0600);
        expect_disabled(f.c);
    }
    {
        fixture f; { server_cache_disk d(f.c); d.save(state(10)); }
        fs::remove(f.path + "/budget.json"); expect_disabled(f.c);
    }
    {
        fixture f; f.c.max_bytes = 1000000; f.c.write_bytes = 1000000;
        server_cache_disk d(f.c); server_prompt_cache_state out;
        for (int i = 0; i < 70; ++i) { d.save(state(i * 1000)); }
        check(f.files() == 64, "entry count cap");
        auto last = state(69000);
        check(d.load(out, last.prompt.tokens, 0), "last loaded record");
        d.reject_last_load(); check(!d.load(out, last.prompt.tokens, 0), "backend refusal retires disk record");
    }
    {
        fixture f; server_cache_disk d(f.c);
        chmod(f.path.c_str(), 0500); d.save(state(10));
        check(f.files() == 0, "read-only cache fallback"); chmod(f.path.c_str(), 0700);
        d.save(state(10)); check(f.files() == 1, "read-only recovery");
    }
    {
        fixture f; f.c.write_bytes = 12000;
        struct rlimit old{}, small{};
        check(getrlimit(RLIMIT_FSIZE, &old) == 0, "getrlimit");
        small = old; small.rlim_cur = 4096;
        signal(SIGXFSZ, SIG_IGN);
        { server_cache_disk d(f.c);
          check(setrlimit(RLIMIT_FSIZE, &small) == 0, "setrlimit");
          d.save(state(10));
          check(setrlimit(RLIMIT_FSIZE, &old) == 0, "restore rlimit");
          check(f.files() == 0 && !fs::exists(f.path + "/.entry.tmp"), "partial write cleanup"); }
        { server_cache_disk d(f.c); d.save(state(100)); check(f.files() == 1, "partial write reservation recovery");
          d.save(state(200)); check(f.files() == 1, "partial write reservation retained"); }
    }
    std::cout << "SSD policy gates passed: admission, roundtrip, dedup, restart, identity, TTL, budget persistence/rollback/renewal, LRU, checksum, truncation, permissions, free-space, crash reservation, orphan, lock, foreign files, corrupt ledger\n";
}
