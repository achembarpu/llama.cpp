#include "server-cache.h"
#include "server-task.h"
#include "server-common.h"
#include "hash/hash.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

struct server_cache_interrupted : std::runtime_error { using std::runtime_error::runtime_error; };

#if defined(__linux__) && defined(CPPHTTPLIB_OPENSSL_SUPPORT) && defined(LLAMA_SERVER_CACHE_ZSTD)
#include <zstd.h>
#include <openssl/evp.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <sched.h>

namespace {
constexpr uint64_t metadata_reserve = 65536;
constexpr size_t max_header = 2 * 1024 * 1024;
constexpr size_t max_entries = 64;
constexpr size_t max_checkpoints = 32;
constexpr uint32_t magic = 0x32445348; // HSD2; one Zstd frame, private format

struct fd_owner {
    int fd;
    explicit fd_owner(int fd = -1) : fd(fd) {}
    ~fd_owner() { if (fd >= 0) { close(fd); } }
    fd_owner(const fd_owner &) = delete;
    operator int() const { return fd; }
    fd_owner & operator=(int value) {
        if (fd >= 0) { close(fd); }
        fd = value;
        return *this;
    }
};

void require(bool ok, const char * reason) {
    if (!ok) { throw std::runtime_error(reason); }
}

void zcheck(size_t result) { require(!ZSTD_isError(result), ZSTD_getErrorName(result)); }

struct digest_sha256 {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(), EVP_MD_CTX_free};
    digest_sha256() { require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA256 init failed"); }
    void update(const uint8_t * data, size_t size) { require(EVP_DigestUpdate(ctx.get(), data, size) == 1, "SHA256 update failed"); }
    void final(uint8_t * out) {
        unsigned n = 0;
        require(EVP_DigestFinal_ex(ctx.get(), out, &n) == 1 && n == 32, "SHA256 final failed");
    }
};

void transfer(int fd, void * data, size_t size, bool writing, digest_sha256 * hash = nullptr) {
    auto * bytes = static_cast<uint8_t *>(data);
    while (size) {
        const size_t chunk = std::min<size_t>(size, 1024 * 1024);
        const ssize_t n = writing ? write(fd, bytes, chunk) : read(fd, bytes, chunk);
        if (n < 0 && errno == EINTR) { continue; }
        if (n < 0) { throw std::runtime_error(std::string(writing ? "write: " : "read: ") + strerror(errno)); }
        require(n > 0, writing ? "write failed" : "truncated record");
        if (hash) { hash->update(bytes, n); }
        bytes += n;
        size -= n;
    }
}

bool private_file(int fd, uint64_t max_size) {
    struct stat s{};
    return fd >= 0 && fstat(fd, &s) == 0 && S_ISREG(s.st_mode) && s.st_uid == geteuid() &&
        (s.st_mode & 077) == 0 && s.st_nlink == 1 && s.st_size >= 0 && uint64_t(s.st_size) <= max_size;
}

bool hex_id(const std::string & s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}

bool entry_name(const std::string & s) {
    return s.size() == 68 && s.substr(64) == ".bin" && hex_id(s.substr(0, 64));
}

std::vector<const std::vector<uint8_t> *> buffers(const server_prompt_cache_state & state) {
    std::vector<const std::vector<uint8_t> *> result = {&state.data.main, &state.data.drft};
    for (const auto & c : state.prompt.checkpoints) {
        result.insert(result.end(), {&c.data_tgt, &c.data_dft, &c.data_spec});
    }
    return result;
}
}

struct server_cache_disk::impl {
    server_cache_disk_config c;
    fd_owner dir;
    fd_owner lock;
    bool enabled = true;
    int64_t high_time = 0;
    json reservations = json::array();
    std::string last_loaded;
    struct entry {
        std::string name;
        json header;
        server_tokens tokens;
        uint64_t size;
        int64_t used;
    };
    std::list<entry> entries;

    int64_t now() {
        high_time = std::max(high_time, c.clock ? c.clock() : int64_t(time(nullptr)));
        return high_time;
    }

    void sync_dir() { require(fsync(dir) == 0, "directory sync failed"); }

    void atomic_small(const std::string & name, const std::string & contents) {
        const std::string tmp = "." + name + ".tmp";
        fd_owner out(openat(dir, tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
        require(out.fd >= 0, "metadata temporary file failed");
        try {
            transfer(out.fd, const_cast<char *>(contents.data()), contents.size(), true);
            require(fsync(out.fd) == 0, "metadata sync failed");
            require(renameat(dir, tmp.c_str(), dir, name.c_str()) == 0, "metadata rename failed");
            sync_dir();
        } catch (...) {
            unlinkat(dir, tmp.c_str(), 0);
            throw;
        }
    }

    void ledger() {
        require(reservations.size() <= 256, "too many write reservations");
        atomic_small("budget.json", json{{"version", 1}, {"time", high_time}, {"writes", reservations}}.dump());
    }

    void forget(std::list<entry>::iterator it) {
        require(unlinkat(dir, it->name.c_str(), 0) == 0 || errno == ENOENT, "eviction failed");
        entries.erase(it);
    }

    void expire() {
        const auto t = now();
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->header.at("created").get<int64_t>() <= t - c.ttl_seconds) {
                auto old = it++;
                forget(old);
            } else { ++it; }
        }
    }

    uint64_t size() const {
        uint64_t n = metadata_reserve;
        for (const auto & e : entries) { n += e.size; }
        return n;
    }

    void make_room(uint64_t bytes) {
        require(bytes <= c.max_bytes - metadata_reserve, "entry exceeds disk cap");
        while (!entries.empty() && (size() + bytes > c.max_bytes || entries.size() >= max_entries)) {
            auto oldest = std::min_element(entries.begin(), entries.end(), [](const entry & a, const entry & b) {
                return a.used < b.used;
            });
            forget(oldest);
        }
        struct statvfs fs{};
        require(fstatvfs(dir, &fs) == 0, "free-space query failed");
        require(uint64_t(fs.f_bavail) * fs.f_frsize >= bytes + c.free_bytes, "free-space reserve");
    }

    json read_header(int fd, uint64_t file_size, digest_sha256 * hash = nullptr) {
        uint32_t pre[2]{};
        transfer(fd, pre, sizeof(pre), false, hash);
        require(pre[0] == magic && pre[1] > 0 && pre[1] <= max_header, "invalid header prefix");
        std::string bytes(pre[1], '\0');
        transfer(fd, &bytes[0], bytes.size(), false, hash);
        const json h = json::parse(bytes);
        require(h.at("version") == 2 && h.at("codec") == "zstd-fast1-long29" && h.at("identity") == c.identity &&
            h.at("seq_version") == LLAMA_STATE_SEQ_VERSION, "incompatible identity or version");
        const auto & tokens = h.at("tokens");
        require(tokens.is_array() && tokens.size() >= c.min_tokens && tokens.size() <= c.max_tokens, "invalid token count");
        for (const auto & t : tokens) { require(t.is_number_integer() && t.get<int64_t>() >= 0 && t.get<int64_t>() < INT32_MAX, "invalid token"); }
        const auto & ckpts = h.at("checkpoints");
        require(ckpts.is_array() && ckpts.size() <= max_checkpoints, "invalid checkpoints");
        const auto & sizes = h.at("sizes");
        require(sizes.is_array() && sizes.size() == 2 + 3 * ckpts.size(), "invalid buffer count");
        uint64_t total = 0;
        for (const auto & s : sizes) {
            require(s.is_number_integer() && s.get<int64_t>() >= 0, "invalid buffer size");
            const uint64_t n = s.get<uint64_t>();
            require(n <= c.max_state_bytes - total, "state exceeds RAM bound");
            total += n;
        }
        require(sizes.at(0).get<uint64_t>() > 0 && file_size > sizeof(pre) + bytes.size() + 32 &&
            file_size <= sizeof(pre) + bytes.size() + ZSTD_compressBound(total) + 32, "invalid file size");
        const int64_t created = h.at("created").get<int64_t>();
        require(created > 0 && created <= now() && created > now() - c.ttl_seconds, "expired or future record");
        for (const auto & ck : ckpts) {
            require(ck.at("n").get<int64_t>() >= 0 && ck.at("n").get<uint64_t>() <= tokens.size() &&
                ck.at("min").get<int64_t>() >= 0 && ck.at("max").get<int64_t>() >= ck.at("min").get<int64_t>() &&
                ck.at("max").get<int64_t>() < int64_t(c.max_tokens), "invalid checkpoint positions");
            (void) ck.at("task").get<int>();
        }
        return h;
    }

    explicit impl(server_cache_disk_config config) : c(std::move(config)) {
        require(ZSTD_versionNumber() == 10507, "SSD cache requires pinned Zstd 1.5.7");
        require(hex_id(c.identity), "cache SSD identity must be a lowercase SHA256 manifest digest");
        require(c.max_bytes > metadata_reserve && c.ttl_seconds > 0 && c.min_tokens > 0 &&
            c.max_tokens >= c.min_tokens && c.max_state_bytes > 0, "invalid SSD bounds");
        // Require a dedicated pre-created directory. Never change permissions or
        // ownership of an existing path, and never follow its final symlink.
        dir = open(c.path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        struct stat ds{};
        require(dir >= 0 && fstat(dir, &ds) == 0 && ds.st_uid == geteuid() && (ds.st_mode & 077) == 0, "SSD directory must be owned by this UID with mode 0700");
        lock = openat(dir, ".lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
        require(private_file(lock, 0) && flock(lock, LOCK_EX | LOCK_NB) == 0, "SSD directory already locked or unsafe lock");
        // Discard only our two incomplete temporary files, under the lock.
        for (const char * tmp : {".entry.tmp", ".budget.json.tmp"}) {
            struct stat s{};
            if (fstatat(dir, tmp, &s, AT_SYMLINK_NOFOLLOW) == 0) {
                require(S_ISREG(s.st_mode) && s.st_uid == geteuid(), "unsafe orphan temporary file");
                require(unlinkat(dir, tmp, 0) == 0, "orphan cleanup failed");
            }
        }
        fd_owner budget(openat(dir, "budget.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
        const bool had_budget = budget.fd >= 0;
        if (budget.fd >= 0) {
            require(private_file(budget.fd, metadata_reserve / 2), "unsafe budget file");
            struct stat s{}; fstat(budget.fd, &s);
            std::string bytes(s.st_size, '\0');
            if (!bytes.empty()) { transfer(budget.fd, &bytes[0], bytes.size(), false); }
            const auto b = json::parse(bytes);
            require(b.at("version") == 1, "incompatible budget");
            high_time = b.at("time").get<int64_t>();
            reservations = b.at("writes");
            require(reservations.is_array() && reservations.size() <= 256, "invalid write ledger");
            for (const auto & w : reservations) {
                require(w.at(0).get<int64_t>() > 0 && w.at(0).get<int64_t>() <= high_time &&
                    w.at(1).is_number_integer() && w.at(1).get<int64_t>() >= 0 &&
                    w.at(1).get<uint64_t>() <= c.write_bytes, "invalid reservation");
            }
        } else { require(errno == ENOENT, "budget unavailable"); }
        now();
        fd_owner scan_fd(dup(dir));
        DIR * scan = fdopendir(dup(scan_fd.fd));
        require(scan != nullptr, "directory scan failed");
        try {
            while (true) {
                errno = 0;
                auto * item = readdir(scan);
                if (!item) { require(errno == 0, "directory read failed"); break; }
                const std::string name = item->d_name;
                if (name == "." || name == ".." || name == ".lock" || name == "budget.json") { continue; }
                require(entry_name(name), "cache directory contains an unrecognized file");
                require(had_budget, "entries exist without their persistent write budget");
                fd_owner in(openat(dir, name.c_str(), O_RDONLY | O_NOFOLLOW | O_NOATIME | O_CLOEXEC));
                require(private_file(in.fd, c.max_bytes), "unsafe cache file");
                struct stat s{}; fstat(in.fd, &s);
                try {
                    json h = read_header(in.fd, s.st_size);
                    server_tokens tokens(h.at("tokens").get<llama_tokens>(), false);
                    entries.push_back({name, std::move(h), std::move(tokens), uint64_t(s.st_size), s.st_atim.tv_sec});
                    while (entries.size() > max_entries || size() > c.max_bytes) {
                        auto oldest = std::min_element(entries.begin(), entries.end(), [](const entry & a, const entry & b) { return a.used < b.used; });
                        forget(oldest);
                    }
                } catch (const std::exception &) {
                    require(unlinkat(dir, name.c_str(), 0) == 0, "invalid entry cleanup failed");
                }
            }
            closedir(scan);
        } catch (...) { closedir(scan); throw; }
        while (entries.size() > max_entries || size() > c.max_bytes) {
            auto oldest = std::min_element(entries.begin(), entries.end(), [](const entry & a, const entry & b) { return a.used < b.used; });
            forget(oldest);
        }
        ledger();
        SRV_INF("SSD cache enabled: %zu entries, %.3f MiB, TTL %lld s\n", entries.size(), size() / 1048576.0, (long long)c.ttl_seconds);
    }

    void save(const server_prompt_cache_state & state) {
        if (!enabled || state.prompt.tokens.has_mtmd || state.prompt.tokens.size() < c.min_tokens ||
            state.prompt.tokens.size() > c.max_tokens || state.prompt.checkpoints.size() > max_checkpoints ||
            state.size() > c.max_state_bytes || state.data.main.empty()) { return; }
        bool published = false;
        try {
            expire();
            for (const auto & e : entries) {
                if (size_t(e.tokens.get_common_prefix(state.prompt.tokens)) == state.prompt.tokens.size()) { return; }
            }
            const auto t = now();
            const int64_t captured = state.captured_at > 0 ? state.captured_at : t;
            require(captured <= t && captured > t - c.ttl_seconds, "capture expired or in future");
            json h = {{"version", 2}, {"codec", "zstd-fast1-long29"}, {"seq_version", LLAMA_STATE_SEQ_VERSION}, {"identity", c.identity},
                {"created", captured}, {"tokens", state.prompt.tokens.get_text_tokens()}, {"sizes", json::array()}, {"checkpoints", json::array()}};
            for (const auto & b : buffers(state)) { h["sizes"].push_back(b->size()); }
            for (const auto & ck : state.prompt.checkpoints) {
                h["checkpoints"].push_back({{"n", ck.n_tokens}, {"task", ck.id_task}, {"min", ck.pos_min}, {"max", ck.pos_max}});
            }
            const std::string header = h.dump();
            require(header.size() <= max_header, "header too large");
            // Reserve the incompressible bound before writing. Settle to actual
            // bytes only after publication; interrupted attempts stay charged.
            const uint64_t bound = 8 + header.size() + ZSTD_compressBound(state.size()) + 32;
            json recent = json::array();
            uint64_t spent = 0;
            for (const auto & w : reservations) {
                if (w.at(0).get<int64_t>() > t - 86400) {
                    const uint64_t n = w.at(1).get<uint64_t>();
                    require(n <= c.write_bytes - std::min(spent, c.write_bytes), "write ledger overflow");
                    spent += n; recent.push_back(w);
                }
            }
            reservations = std::move(recent);
            require(bound <= c.write_bytes - spent && reservations.size() < 256, "rolling-day write cap");
            // Reserve durably before attempting data writes, including failures
            // or crashes. Restart and clock rollback cannot reset the budget.
            reservations.push_back(json::array({t, bound}));
            ledger();
            make_room(bound);
            fd_owner out(openat(dir, ".entry.tmp", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
            require(out.fd >= 0, "entry temporary file failed");
            digest_sha256 sha;
            uint32_t pre[2] = {magic, uint32_t(header.size())};
            transfer(out.fd, pre, sizeof(pre), true, &sha);
            transfer(out.fd, const_cast<char *>(header.data()), header.size(), true, &sha);
            std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> codec(ZSTD_createCCtx(), ZSTD_freeCCtx);
            require(bool(codec), "compression allocation failed");
            zcheck(ZSTD_CCtx_setParameter(codec.get(), ZSTD_c_compressionLevel, -1));
            zcheck(ZSTD_CCtx_setParameter(codec.get(), ZSTD_c_enableLongDistanceMatching, 1));
            zcheck(ZSTD_CCtx_setParameter(codec.get(), ZSTD_c_windowLog, 29));
            zcheck(ZSTD_CCtx_setParameter(codec.get(), ZSTD_c_nbWorkers, 0));
            zcheck(ZSTD_CCtx_setPledgedSrcSize(codec.get(), state.size()));
            std::vector<uint8_t> chunk(1024 * 1024);
            const auto sources = buffers(state);
            for (size_t i = 0; i < sources.size(); ++i) {
                ZSTD_inBuffer input{sources[i]->data(), sources[i]->size(), 0};
                const auto mode = i + 1 == sources.size() ? ZSTD_e_end : ZSTD_e_continue;
                size_t remaining;
                do {
                    if (c.checkpoint) { c.checkpoint(); }
                    ZSTD_outBuffer output{chunk.data(), chunk.size(), 0};
                    remaining = ZSTD_compressStream2(codec.get(), &output, &input, mode);
                    zcheck(remaining);
                    transfer(out.fd, chunk.data(), output.pos, true, &sha);
                } while (input.pos < input.size || (mode == ZSTD_e_end && remaining));
            }
            std::array<uint8_t, 32> digest{}; sha.final(digest.data());
            transfer(out.fd, digest.data(), digest.size(), true);
            const auto end = lseek(out.fd, 0, SEEK_CUR);
            require(end > 0 && uint64_t(end) <= bound, "compressed size exceeds reservation");
            const uint64_t bytes = end;
            struct timespec times[2] = {{t, 0}, {0, UTIME_OMIT}};
            require(futimens(out.fd, times) == 0, "initial LRU timestamp failed");
            require(fsync(out.fd) == 0, "entry sync failed");
            const std::string name = hash_sha256_hex(header.data(), header.size()) + ".bin";
            require(renameat(dir, ".entry.tmp", dir, name.c_str()) == 0, "entry rename failed");
            published = true;
            sync_dir();
            reservations.back().at(1) = bytes;
            ledger();
            entries.push_back({name, std::move(h), state.prompt.tokens.clone(), bytes, t});
            // Only retire a shorter prefix after its replacement is durable.
            for (auto it = entries.begin(); it != std::prev(entries.end());) {
                if (size_t(it->tokens.get_common_prefix(state.prompt.tokens)) == it->tokens.size()) {
                    auto old = it++; forget(old);
                } else { ++it; }
            }
            SRV_INF("SSD cache saved %zu tokens, %llu bytes\n", state.prompt.tokens.size(), (unsigned long long)bytes);
        } catch (const std::exception & e) {
            if (published) { enabled = false; }
            unlinkat(dir, ".entry.tmp", 0);
            SRV_WRN("SSD cache save skipped: %s\n", e.what());
        }
    }

    bool load(server_prompt_cache_state & state, const server_tokens & tokens, size_t min_lcp,
              const std::function<void()> & before_allocate) {
        last_loaded.clear();
        if (!enabled || tokens.has_mtmd) { return false; }
        auto best = entries.end();
        try {
            expire();
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                const size_t lcp = it->tokens.get_common_prefix(tokens);
                if (lcp >= c.min_tokens && lcp > min_lcp && lcp * 4 >= it->tokens.size()) {
                    best = it; min_lcp = lcp;
                }
            }
            if (best == entries.end()) { return false; }
            fd_owner in(openat(dir, best->name.c_str(), O_RDONLY | O_NOFOLLOW | O_NOATIME | O_CLOEXEC));
            require(private_file(in.fd, c.max_bytes), "unsafe cache hit");
            struct stat s{}; fstat(in.fd, &s);
            digest_sha256 sha;
            const json h = read_header(in.fd, s.st_size, &sha);
            require(h == best->header, "record changed since indexing");
            if (before_allocate) { before_allocate(); }
            server_prompt_cache_state result;
            result.prompt.tokens = best->tokens.clone();
            result.disk_name = best->name;
            result.captured_at = h.at("created").get<int64_t>();
            for (const auto & ck : h.at("checkpoints")) {
                common_prompt_checkpoint cp{};
                cp.n_tokens = ck.at("n").get<int64_t>();
                // Task IDs are local to one server process. Restored checkpoints
                // must never collide with a new task's checkpoint-pruning group.
                cp.id_task = -1;
                cp.pos_min = ck.at("min").get<int>(); cp.pos_max = ck.at("max").get<int>();
                result.prompt.checkpoints.push_back(std::move(cp));
            }
            size_t i = 0;
            for (const auto * cb : buffers(result)) {
                auto * b = const_cast<std::vector<uint8_t> *>(cb);
                b->resize(h.at("sizes").at(i++).get<size_t>());
            }
            std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> codec(ZSTD_createDCtx(), ZSTD_freeDCtx);
            require(bool(codec), "decompression allocation failed");
            zcheck(ZSTD_DCtx_setParameter(codec.get(), ZSTD_d_windowLogMax, 29));
            const auto destinations = buffers(result);
            std::vector<uint8_t> chunk(1024 * 1024);
            const auto payload_offset = lseek(in.fd, 0, SEEK_CUR);
            require(payload_offset > 0 && uint64_t(s.st_size) > uint64_t(payload_offset) + 32, "invalid payload offset");
            uint64_t left = s.st_size - payload_offset - 32;
            size_t buffer = 0, position = 0, remaining = 1;
            while (left) {
                if (c.checkpoint) { c.checkpoint(); }
                const size_t n = std::min<uint64_t>(left, chunk.size());
                transfer(in.fd, chunk.data(), n, false, &sha);
                left -= n;
                ZSTD_inBuffer input{chunk.data(), n, 0};
                while (input.pos < input.size) {
                    while (buffer < destinations.size() && position == destinations[buffer]->size()) { ++buffer; position = 0; }
                    uint8_t extra;
                    auto * dest = buffer < destinations.size() ? const_cast<std::vector<uint8_t> *>(destinations[buffer]) : nullptr;
                    ZSTD_outBuffer output{dest ? dest->data() + position : &extra, dest ? dest->size() - position : 1, 0};
                    const size_t before = input.pos;
                    remaining = ZSTD_decompressStream(codec.get(), &output, &input);
                    zcheck(remaining);
                    require(dest || !output.pos, "decoded state exceeds declared size");
                    position += output.pos;
                    require(input.pos > before || output.pos > 0, "decoder made no progress");
                    if (!remaining) { require(!left && input.pos == input.size, "trailing or concatenated frame"); break; }
                }
            }
            while (buffer < destinations.size() && position == destinations[buffer]->size()) { ++buffer; position = 0; }
            require(!remaining && buffer == destinations.size(), "truncated decoded state");
            std::array<uint8_t, 32> saved{}, actual{};
            transfer(in.fd, saved.data(), saved.size(), false);
            sha.final(actual.data());
            require(saved == actual, "checksum mismatch");
            best->used = now();
            struct timespec times[2] = {{best->used, 0}, {0, UTIME_OMIT}};
            require(futimens(in.fd, times) == 0, "LRU update failed");
            state = std::move(result);
            last_loaded = best->name;
            SRV_INF("SSD cache hit: %zu prefix tokens, %llu bytes\n", min_lcp, (unsigned long long)best->size);
            return true;
        } catch (const std::exception & e) {
            SRV_WRN("SSD cache miss: %s\n", e.what());
            if (best != entries.end() && !dynamic_cast<const server_cache_interrupted *>(&e)) {
                try { forget(best); } catch (...) { enabled = false; }
            }
            return false;
        }
    }

    void reject_last_load() {
        reject(last_loaded);
        last_loaded.clear();
    }

    void reject(const std::string & name) {
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->name == name) {
                try { forget(it); } catch (...) { enabled = false; }
                break;
            }
        }
    }
};
#else
struct server_cache_disk::impl {
    explicit impl(server_cache_disk_config) { throw std::runtime_error("SSD cache requires Linux, OpenSSL Crypto and LLAMA_SERVER_CACHE_ZSTD"); }
    void save(const server_prompt_cache_state &) {}
    void reject_last_load() {}
    void reject(const std::string &) {}
    bool load(server_prompt_cache_state &, const server_tokens &, size_t, const std::function<void()> &) { return false; }
};
#endif

server_cache_disk::server_cache_disk(server_cache_disk_config config) : p(new impl(std::move(config))) {}
server_cache_disk::~server_cache_disk() = default;
void server_cache_disk::save(const server_prompt_cache_state & state) { p->save(state); }
void server_cache_disk::reject_last_load() { p->reject_last_load(); }
void server_cache_disk::reject(const std::string & name) { p->reject(name); }
bool server_cache_disk::load(server_prompt_cache_state & state, const server_tokens & tokens, size_t min_lcp,
                             const std::function<void()> & before_allocate) {
    return p->load(state, tokens, min_lcp, before_allocate);
}

struct server_cache_async::impl {
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> stop{false};
    bool cancel_read = false;
    bool reserved = false;
    bool save_active = false;
    int reader = -1;
    status phase = status::pending;
    server_tokens tokens;
    size_t min_lcp = 0;
    std::shared_ptr<server_prompt_cache_state> result;
    std::shared_ptr<const server_prompt_cache_state> pending_save;
    std::vector<std::string> rejected;
    std::function<void()> wake;
    server_cache_disk disk;
    std::thread thread;

    server_cache_disk_config configure(server_cache_disk_config c) {
        c.checkpoint = [this]() {
            if (stop.load()) { throw server_cache_interrupted("cache worker stopping"); }
            std::lock_guard<std::mutex> lock(mutex);
            if (!save_active && cancel_read) { throw server_cache_interrupted("cache read cancelled"); }
        };
        return c;
    }

    impl(server_cache_disk_config c, std::function<void()> wake) : wake(std::move(wake)), disk(configure(c)) {
        thread = std::thread([this, cpu = c.worker_cpu]() {
#if defined(__linux__)
            if (cpu >= 0) {
                cpu_set_t selected{};
                if (cpu >= CPU_SETSIZE) {
                    SRV_WRN("%s", "SSD worker CPU is unavailable; cache worker stopped\n");
                    stop = true; this->wake(); return;
                }
                CPU_ZERO(&selected); CPU_SET(cpu, &selected);
                // The parent may already be pinned to inference P cores. The
                // kernel checks the container cpuset when selecting this CPU.
                if (sched_setaffinity(0, sizeof(selected), &selected) != 0) {
                    SRV_WRN("%s", "SSD worker CPU affinity failed; cache worker stopped\n");
                    stop = true; this->wake(); return;
                }
            }
#else
            (void) cpu;
#endif
            run();
        });
    }

    ~impl() { stop = true; cv.notify_all(); if (thread.joinable()) { thread.join(); } }

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stop) {
            cv.wait(lock, [&]() { return stop || (reader >= 0 && phase == status::pending) || pending_save || !rejected.empty(); });
            if (stop) { break; }
            if (!rejected.empty()) {
                auto names = std::move(rejected);
                rejected.clear();
                lock.unlock(); for (const auto & name : names) { disk.reject(name); } lock.lock();
            } else if (reader >= 0 && phase == status::pending) {
                // A pending best-effort save must not pin the RAM needed by
                // this read's allocation handshake.
                pending_save.reset();
                const int id = reader;
                auto input = tokens.clone();
                const size_t prefix = min_lcp;
                lock.unlock();
                auto decoded = std::make_shared<server_prompt_cache_state>();
                const bool hit = disk.load(*decoded, input, prefix, [this, id]() {
                    std::unique_lock<std::mutex> guard(mutex);
                    if (stop || cancel_read || reader != id) { throw server_cache_interrupted("cache read cancelled"); }
                    phase = status::memory;
                    guard.unlock(); wake(); guard.lock();
                    cv.wait(guard, [&]() { return stop || cancel_read || reserved; });
                    if (stop || cancel_read) { throw server_cache_interrupted("cache read cancelled"); }
                });
                lock.lock();
                if (cancel_read || stop) {
                    decoded.reset(); result.reset(); reader = -1; reserved = false; cancel_read = false;
                } else {
                    result = hit ? std::move(decoded) : nullptr;
                    phase = hit ? status::ready : status::miss;
                    if (!hit) { reserved = false; }
                }
                lock.unlock(); wake(); lock.lock();
            } else if (pending_save) {
                auto state = std::move(pending_save);
                save_active = true;
                lock.unlock(); disk.save(*state); state.reset(); lock.lock();
                save_active = false;
                lock.unlock(); wake(); lock.lock();
            }
        }
    }
};

server_cache_async::server_cache_async(server_cache_disk_config c, std::function<void()> wake) : p(new impl(std::move(c), std::move(wake))) {}
server_cache_async::~server_cache_async() = default;
bool server_cache_async::save(std::shared_ptr<const server_prompt_cache_state> state) {
    std::lock_guard<std::mutex> lock(p->mutex);
    if (p->stop || p->reserved) { return false; }
    if (p->pending_save && size_t(p->pending_save->prompt.tokens.get_common_prefix(state->prompt.tokens)) != p->pending_save->prompt.tokens.size()) { return false; }
    p->pending_save = std::move(state); // coalesce; no deep copy of KV/checkpoints
    p->cv.notify_one();
    return true;
}
server_cache_async::status server_cache_async::poll(int task, const server_tokens & tokens, size_t min_lcp) {
    std::lock_guard<std::mutex> lock(p->mutex);
    if (p->stop) { return status::miss; }
    if (p->reader == task && !p->cancel_read) { return p->phase; }
    if (p->reader >= 0) { return status::pending; }
    p->reader = task; p->tokens = tokens.clone(); p->min_lcp = min_lcp; p->phase = status::pending;
    p->cv.notify_one();
    return status::pending;
}
void server_cache_async::grant_memory(int task) {
    std::lock_guard<std::mutex> lock(p->mutex);
    if (p->reader == task && p->phase == status::memory && !p->cancel_read) { p->reserved = true; p->cv.notify_one(); }
}
std::shared_ptr<server_prompt_cache_state> server_cache_async::take(int task) {
    std::lock_guard<std::mutex> lock(p->mutex);
    if (p->reader != task || (p->phase != status::ready && p->phase != status::miss)) { return {}; }
    auto result = std::move(p->result);
    p->reader = -1; p->reserved = false; p->tokens.clear();
    p->cv.notify_one();
    return result;
}
void server_cache_async::cancel(int task) {
    std::lock_guard<std::mutex> lock(p->mutex);
    if (p->reader != task) { return; }
    if (p->phase == status::ready || p->phase == status::miss) {
        p->result.reset(); p->reader = -1; p->reserved = false;
    } else { p->cancel_read = true; }
    p->cv.notify_one();
}
bool server_cache_async::memory_reserved() const { std::lock_guard<std::mutex> lock(p->mutex); return p->reserved; }
void server_cache_async::reject(const std::string & name) {
    if (name.empty()) { return; }
    std::lock_guard<std::mutex> lock(p->mutex);
    if (p->rejected.size() < 64) { p->rejected.push_back(name); p->cv.notify_one(); }
}
void server_cache_async::interrupt() { p->stop = true; p->cv.notify_all(); }
