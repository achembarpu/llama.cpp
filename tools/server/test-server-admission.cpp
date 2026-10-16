#include "server-admission.h"
#include "server-queue.h"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <thread>

static void check(bool value) {
    if (!value) { std::abort(); }
}

static server_task request(int id, int64_t posted) {
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = id;
    task.t_queued_ms = posted;
    return task;
}

int main() {
    const int64_t pool = 131072;
    const auto long_chat = server_token_reservation(108415, 1536, pool, 4);
    const auto short_chat = server_token_reservation(8192, 2048, pool, 4);
    check(server_token_fits(long_chat, short_chat, pool));
    const auto large = server_token_reservation(70000, 4096, pool, 4);
    check(server_token_fits(0, large, pool));
    check(!server_token_fits(large, large, pool));
    check(server_token_reservation(1000, -1, pool, 4) == pool);
    check(!server_token_fits(1, pool, pool));
    check(!server_token_fits(0, server_token_reservation(pool - 3, 1, pool, 4), pool));
    check(!server_token_fits(0, server_token_reservation(120000, 20000, pool, 4), pool));
    check(server_token_fits(pool - 4, 4, pool));
    check(!server_token_fits(INT64_MAX, 1, pool));

    server_queue queue;
    auto first = request(1, 100), second = request(2, 200), third = request(3, 300);
    check(!queue.defer(std::move(first), 0));
    check(first.id == 1);
    check(queue.defer(std::move(first), 2));
    check(queue.defer(std::move(second), 2));
    check(!queue.defer(std::move(third), 2));
    check(third.id == 3 && queue.queue_tasks_deferred_size() == 2);
    auto expired = queue.expire_deferred(150);
    check(expired.size() == 1 && expired[0].id == 1 && expired[0].t_queued_ms == 100);
    check(queue.defer(std::move(third), 2));
    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id = 4;
    cancel.id_target = 2;
    queue.post(std::move(cancel));
    check(queue.queue_tasks_deferred_size() == 1);
    queue.retry_deferred();
    check(queue.queue_tasks_deferred_size() == 0);
    cancel = server_task(SERVER_TASK_TYPE_CANCEL);
    cancel.id = 5;
    cancel.id_target = 3;
    queue.post(std::move(cancel)); // also removes a reoffered waiter
    check(queue.expire_deferred(INT64_MAX).empty());
    // Other clients' notifications must not postpone cancellation polling.
    server_response responses;
    responses.add_waiting_task_id(1);
    responses.add_waiting_task_id(2);
    std::atomic<bool> stop{false};
    std::thread noisy([&]() {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!stop && std::chrono::steady_clock::now() < until) {
            auto result = std::make_unique<server_task_result_error>();
            result->id = 2;
            responses.send(std::move(result));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    const auto start = std::chrono::steady_clock::now();
    auto result = responses.recv_with_timeout({1}, 1);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    stop = true;
    noisy.join();
    std::printf("response polling under unrelated traffic: %.3fs\n", elapsed);
    std::fflush(stdout);
    check(!result && elapsed < 2.0);
    std::puts("admission boundaries, bounded waiting, expiry, retry and cancellation passed");
}
