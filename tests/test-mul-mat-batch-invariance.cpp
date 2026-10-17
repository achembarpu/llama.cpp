#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static std::vector<float> values(size_t n, uint32_t & seed) {
    std::vector<float> out(n);
    for (float & v : out) {
        seed = seed * 1664525u + 1013904223u;
        v = float(seed >> 8) / 8388608.0f - 1.0f;
    }
    return out;
}

static bool check(ggml_backend_t backend, int64_t k, int64_t m, int64_t n, int64_t row) {
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);
    auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_set_name(a, "batch_invariance_weights");
    ggml_set_name(b, "batch_invariance_input");
    auto * all = ggml_mul_mat(ctx, a, b);
    auto * one_input = ggml_view_2d(ctx, b, k, 1, b->nb[1], row * b->nb[1]);
    auto * one = ggml_mul_mat(ctx, a, one_input);
    ggml_set_output(all);
    ggml_set_output(one);
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, all);
    ggml_build_forward_expand(graph, one);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("backend allocation");
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    uint32_t seed = 0x45a31701;
    const auto weights = values(k * m, seed);
    const auto input = values(k * n, seed);
    std::vector<uint8_t> quant(ggml_nbytes(a));
    if (ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), quant.data(), 0, m, k, nullptr) != quant.size()) {
        throw std::runtime_error("weight quantization");
    }
    ggml_backend_tensor_set(a, quant.data(), 0, quant.size());
    ggml_backend_tensor_set(b, input.data(), 0, input.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("graph compute");
    }
    std::vector<float> x(m), y(m);
    ggml_backend_tensor_get(all, x.data(), row * all->nb[1], m * sizeof(float));
    ggml_backend_tensor_get(one, y.data(), 0, m * sizeof(float));
    size_t different = 0;
    float max_abs = 0;
    for (int64_t i = 0; i < m; ++i) {
        different += std::memcmp(&x[i], &y[i], sizeof(float)) != 0;
        max_abs = std::max(max_abs, std::abs(x[i] - y[i]));
    }
    std::printf("{\"backend\":\"%s\",\"k\":%lld,\"m\":%lld,\"n\":%lld,\"row\":%lld,\"different\":%zu,\"max_abs\":%.9g}\n",
                ggml_backend_name(backend), (long long) k, (long long) m, (long long) n,
                (long long) row, different, max_abs);
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return different == 0;
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        const char * name = argc > 1 ? argv[1] : "CUDA0";
        auto * device = ggml_backend_dev_by_name(name);
        if (!device) throw std::runtime_error("backend missing");
        auto * backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) throw std::runtime_error("backend init");
        size_t passed = 0, total = 0;
        for (int64_t n : {2, 3, 4, 8, 17, 32, 128}) {
            for (int64_t row : {int64_t(0), n - 1}) {
                passed += check(backend, 3840, 2048, n, row);
                ++total;
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
