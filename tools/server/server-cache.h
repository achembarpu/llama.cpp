#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

struct server_prompt_cache_state;
struct server_tokens;

// The SSD tier is opt-in. Its identity must hash the exact engine, target/head,
// template and context/KV configuration manifest used by the trusted launcher.
struct server_cache_disk_config {
    std::string path;
    std::string identity;
    uint64_t max_bytes = 32ull << 30;
    uint64_t write_bytes = 16ull << 30;
    uint64_t free_bytes = 2ull << 30;
    int64_t ttl_seconds = 14 * 86400;
    size_t min_tokens = 8192;
    size_t max_tokens = 131072;
    uint64_t max_state_bytes = 16ull << 30;
    std::function<int64_t()> clock;
    std::function<void()> checkpoint; // bounded I/O/codec cancellation point
    int worker_cpu = -1;
};

// One I/O owner, one immutable save in flight and one coalesced pending save.
// Restores get priority over pending saves. Only the model executor may grant
// memory or apply the decoded state; this worker never accesses a llama context.
class server_cache_async {
public:
    enum class status { pending, memory, miss, ready };
    explicit server_cache_async(server_cache_disk_config config, std::function<void()> wake);
    ~server_cache_async();
    bool save(std::shared_ptr<const server_prompt_cache_state> state);
    status poll(int task, const server_tokens & tokens, size_t min_lcp);
    size_t result_prefix(int task, const server_tokens & tokens) const;
    void grant_memory(int task);
    std::shared_ptr<server_prompt_cache_state> take(int task);
    void cancel(int task);
    void reject(const std::string & name);
    bool memory_reserved() const;
    void interrupt();
private:
    struct impl;
    std::unique_ptr<impl> p;
};

class server_cache_disk {
public:
    explicit server_cache_disk(server_cache_disk_config config);
    ~server_cache_disk();
    server_cache_disk(const server_cache_disk &) = delete;
    server_cache_disk & operator=(const server_cache_disk &) = delete;

    void save(const server_prompt_cache_state & state);
    void reject_last_load();
    void reject(const std::string & name);
    // Load only if the disk prefix beats both live and RAM prefixes. No GPU state
    // is touched until the complete record and checksum have been validated.
    bool load(server_prompt_cache_state & state, const server_tokens & tokens, size_t min_lcp,
              const std::function<void()> & before_allocate = {});

private:
    struct impl;
    std::unique_ptr<impl> p;
};
