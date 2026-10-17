#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (n <= 0) throw std::runtime_error("token count");
    std::vector<llama_token> tokens(n);
    n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) throw std::runtime_error("tokenize");
    tokens.resize(n);
    return tokens;
}

static void decode(llama_context * ctx, const std::vector<llama_token> & tokens,
                   int begin, int end, int chunk, bool output) {
    auto batch = llama_batch_init(chunk, 0, 1);
    for (int pos = begin; pos < end; pos += chunk) {
        batch.n_tokens = std::min(chunk, end - pos);
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[pos + i];
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = output && pos + i == end - 1;
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("decode failed");
    }
    llama_batch_free(batch);
}

static std::vector<float> capture(llama_context * ctx, int n_vocab,
                                  const std::filesystem::path & out, const std::string & label) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) throw std::runtime_error("missing logits");
    std::vector<float> values(logits, logits + n_vocab);
    std::ofstream scores(out / (label + ".f32"), std::ios::binary);
    scores.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    const size_t size = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> state(size);
    if (llama_state_seq_get_data(ctx, state.data(), size, 0) != size) throw std::runtime_error("state read");
    std::ofstream cache(out / (label + ".state"), std::ios::binary);
    cache.write(reinterpret_cast<const char *>(state.data()), state.size());
    if (!scores || !cache) throw std::runtime_error("output write");
    std::printf("{\"label\":\"%s\",\"state_bytes\":%zu,\"argmax\":%td}\n",
                label.c_str(), size, std::max_element(values.begin(), values.end()) - values.begin());
    std::fflush(stdout);
    return values;
}

static bool compare(const char * label, const std::vector<float> & a, const std::vector<float> & b) {
    size_t different = 0;
    float max_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        different += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        max_abs = std::max(max_abs, std::abs(a[i] - b[i]));
    }
    std::printf("{\"comparison\":\"%s\",\"different_logits\":%zu,\"max_abs\":%.9g}\n",
                label, different, max_abs);
    std::fflush(stdout);
    return different == 0;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL PROMPT OUTPUT_DIRECTORY SWA_FULL\n", argv[0]);
        return 2;
    }
    try {
        const std::filesystem::path out(argv[3]);
        std::filesystem::create_directories(out);
        std::ifstream in(argv[2]);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!in || text.empty()) throw std::runtime_error("prompt input");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        mp.load_mtp = false;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load");
        auto cp = llama_context_default_params();
        cp.n_ctx = 6144;
        cp.n_batch = 512;
        cp.n_ubatch = 128;
        cp.n_seq_max = 2;
        cp.n_threads = cp.n_threads_batch = 6;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.kv_unified = true;
        cp.swa_full = std::stoi(argv[4]) != 0;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context load");
        const auto * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const auto tokens = tokenize(vocab, text);
        const int n = tokens.size();
        if (n < 1000 || n > 5000) throw std::runtime_error("unexpected prompt size");
        std::printf("{\"tokens\":%d,\"vocab\":%d,\"swa_full\":%s}\n", n, n_vocab, cp.swa_full ? "true" : "false");
        llama_memory_clear(llama_get_memory(ctx), true);
        decode(ctx, tokens, 0, n - 17, 128, false);
        decode(ctx, tokens, n - 17, n, 17, true);
        const auto a = capture(ctx, n_vocab, out, "batch17");
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n - 1, -1)) throw std::runtime_error("remove last");
        decode(ctx, tokens, n - 1, n, 1, true);
        const auto b = capture(ctx, n_vocab, out, "replay1");
        const bool replay_exact = compare("batch17-replay1", a, b);
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n - 1, -1)) throw std::runtime_error("remove last twice");
        decode(ctx, tokens, n - 1, n, 1, true);
        const auto c = capture(ctx, n_vocab, out, "repeat1");
        const bool repeat_exact = compare("replay1-repeat1", b, c);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return replay_exact && repeat_exact ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "diagnostic failed: %s\n", e.what());
        return 1;
    }
}
