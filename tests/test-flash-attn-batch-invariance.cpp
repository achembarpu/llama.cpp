#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static bool check(ggml_backend_t backend, int64_t d, int64_t heads_kv, int64_t gqa,
                  int64_t kv, int64_t n, int64_t row, int64_t tail) {
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n, heads_kv * gqa, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv + tail, heads_kv, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv + tail, heads_kv, 1);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv + tail, n, 1, 1);
    auto * mask_short = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, n, 1, 1);
    for (auto * t : {q, k, v, mask, mask_short}) ggml_set_input(t);
    auto * k_short = ggml_view_4d(ctx, k, d, kv, heads_kv, 1, k->nb[1], k->nb[2], k->nb[3], 0);
    auto * v_short = ggml_view_4d(ctx, v, d, kv, heads_kv, 1, v->nb[1], v->nb[2], v->nb[3], 0);
    auto * all = ggml_flash_attn_ext(ctx, q, k_short, v_short, mask_short, 1.0f / std::sqrt(float(d)), 0, 0);
    ggml_prec_set_acc(all, GGML_PREC_F32);
    auto * one_q = ggml_view_4d(ctx, q, d, 1, heads_kv * gqa, 1, q->nb[1], q->nb[2], q->nb[3], row * q->nb[1]);
    auto * one_mask = ggml_view_4d(ctx, mask, kv + tail, 1, 1, 1, mask->nb[1], mask->nb[2], mask->nb[3], row * mask->nb[1]);
    auto * one = ggml_flash_attn_ext(ctx, one_q, k, v, one_mask, 1.0f / std::sqrt(float(d)), 0, 0);
    ggml_prec_set_acc(one, GGML_PREC_F32);
    ggml_set_output(all);
    ggml_set_output(one);
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, all);
    ggml_build_forward_expand(graph, one);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("backend allocation");
    uint32_t seed = 0x45fa1234;
    for (auto * t : {q, k, v, mask, mask_short}) {
        std::vector<float> values(ggml_nelements(t));
        for (size_t i = 0; i < values.size(); ++i) {
            seed = seed * 1664525u + 1013904223u;
            values[i] = t == mask || t == mask_short ? (int64_t(i % t->ne[0]) <= kv - n + int64_t(i / t->ne[0]) ? 0.0f : -INFINITY)
                                 : float(seed >> 8) / 8388608.0f - 1.0f;
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
        } else {
            std::vector<uint8_t> data(ggml_nbytes(t));
            if (ggml_quantize_chunk(t->type, values.data(), data.data(), 0, ggml_nrows(t), t->ne[0], nullptr) != data.size()) {
                throw std::runtime_error("F16 conversion");
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size());
        }
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute");
    const int64_t width = d * heads_kv * gqa;
    std::vector<float> a(width), b(width);
    ggml_backend_tensor_get(all, a.data(), row * all->nb[2], width * sizeof(float));
    ggml_backend_tensor_get(one, b.data(), 0, width * sizeof(float));
    size_t different = 0;
    float max_abs = 0;
    for (int64_t i = 0; i < width; ++i) {
        different += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        max_abs = std::max(max_abs, std::abs(a[i] - b[i]));
    }
    std::printf("{\"d\":%lld,\"gqa\":%lld,\"kv\":%lld,\"tail\":%lld,\"n\":%lld,\"row\":%lld,\"different\":%zu,\"max_abs\":%.9g}\n",
                (long long) d, (long long) gqa, (long long) kv, (long long) tail, (long long) n, (long long) row, different, max_abs);
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return different == 0;
}

int main() {
    try {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!dev) throw std::runtime_error("CUDA0 missing");
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) throw std::runtime_error("backend init");
        size_t passed = 0, total = 0;
        for (int64_t tail : {0, 256}) {
            for (int64_t d : {256, 512}) {
                const std::vector<int64_t> depths = d == 256 ? std::vector<int64_t>{1536} : std::vector<int64_t>{4096, 4608, 8192};
                for (int64_t kv : depths) {
                    for (int64_t n : {2, 3, 8, 17, 128}) {
                        for (int64_t row : {int64_t(0), n - 1}) {
                            passed += check(backend, d, d == 256 ? 8 : 1, d == 256 ? 2 : 16, kv, n, row, tail);
                            ++total;
                        }
                    }
                }
            }
        }
        std::printf("{\"passed\":%zu,\"total\":%zu}\n", passed, total);
        ggml_backend_free(backend);
        return passed == total ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}
