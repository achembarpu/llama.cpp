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

static void feed(llama_context * ctx, const std::vector<llama_token> & tokens,
                 int seq, int begin, int end) {
    auto b = llama_batch_init(128, 0, 1);
    for (int pos = begin; pos < end; pos += 128) {
        b.n_tokens = std::min(128, end - pos);
        for (int i = 0; i < b.n_tokens; ++i) {
            b.token[i] = tokens[(pos + i) % tokens.size()]; b.pos[i] = pos + i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = seq; b.logits[i] = false;
        }
        if (llama_decode(ctx, b) != 0) throw std::runtime_error("feed");
    }
    llama_batch_free(b);
}

static std::vector<std::vector<float>> pair_scores(llama_context * ctx,
        const std::vector<llama_token> & tokens, int nv, int which, bool shared = false, int seq1_count = 1) {
    const int positions[] = {2999, shared ? 2999 : 3299};
    auto b = llama_batch_init(1 + seq1_count, 0, 1);
    b.n_tokens = which < 0 ? 1 + seq1_count : (which == 1 ? seq1_count : 1);
    for (int i = 0; i < b.n_tokens; ++i) {
        const int seq = which < 0 ? (i == 0 ? 0 : 1) : which;
        const int pos = positions[seq] + (seq == 1 ? i - (which < 0 ? 1 : 0) : 0);
        b.token[i] = tokens[pos]; b.pos[i] = pos;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = seq; b.logits[i] = true;
    }
    if (llama_decode(ctx, b) != 0) throw std::runtime_error("pair");
    std::vector<std::vector<float>> result;
    for (int i = 0; i < b.n_tokens; ++i) {
        const float * p = llama_get_logits_ith(ctx, i);
        if (!p) throw std::runtime_error("logits");
        result.emplace_back(p, p + nv);
    }
    llama_batch_free(b);
    return result;
}

static std::vector<uint8_t> state(llama_context * ctx, int seq, llama_state_seq_flags flags) {
    std::vector<uint8_t> data(llama_state_seq_get_size_ext(ctx, seq, flags));
    if (llama_state_seq_get_data_ext(ctx, data.data(), data.size(), seq, flags) != data.size()) {
        throw std::runtime_error("state");
    }
    return data;
}

static void dump(llama_context * ctx, int seq, const std::filesystem::path & out,
                 const std::string & label) {
    const auto data = state(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::ofstream f(out / (label + "-" + std::to_string(seq) + ".state"), std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), data.size());
    if (!f) throw std::runtime_error("dump");
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL PROMPT OUTPUT_DIRECTORY SWA_FULL\n", argv[0]);
        return 2;
    }
    try {
        const std::filesystem::path out(argv[3]); std::filesystem::create_directories(out);
        std::ifstream in(argv[2]);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        ggml_backend_load_all(); llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 99; mp.load_mtp = false;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model");
        auto cp = llama_context_default_params(); cp.n_ctx = 8192; cp.n_batch = 512; cp.n_ubatch = 128;
        cp.n_seq_max = 2; cp.n_threads = cp.n_threads_batch = 6;
        cp.type_k = cp.type_v = GGML_TYPE_F16; cp.kv_unified = true; cp.swa_full = std::stoi(argv[4]) != 0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context");
        auto * mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);
        const auto tokens = tokenize(llama_model_get_vocab(model), text);
        if (tokens.size() < 3600) throw std::runtime_error("prompt");
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (int pos = 0; pos < 3299; pos += 128) {
            if (pos < 2999) feed(ctx, tokens, 0, pos, std::min(pos + 128, 2999));
            feed(ctx, tokens, 1, pos, std::min(pos + 128, 3299));
        }
        const auto partial0 = state(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const auto partial1 = state(ctx, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        dump(ctx, 0, out, "before"); dump(ctx, 1, out, "before");
        const auto batched = pair_scores(ctx, tokens, nv, -1);
        if (!llama_memory_seq_rm(mem, 0, 2999, -1)) throw std::runtime_error("remove0");
        const auto solo0 = pair_scores(ctx, tokens, nv, 0);
        bool exact = compare("two-seq-batch-solo0", batched[0], solo0[0]);
        if (!llama_memory_seq_rm(mem, 1, 3299, -1)) throw std::runtime_error("remove1");
        const auto solo1 = pair_scores(ctx, tokens, nv, 1);
        exact = compare("two-seq-batch-solo1", batched[1], solo1[0]) && exact;
        for (int pos = 0; pos < 256; pos += 128) {
            feed(ctx, tokens, 0, 3000 + pos, 3000 + pos + 128);
            feed(ctx, tokens, 1, 3300 + pos, 3300 + pos + 128);
        }
        if (llama_state_seq_set_data_ext(ctx, partial0.data(), partial0.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != partial0.size() ||
            llama_state_seq_set_data_ext(ctx, partial1.data(), partial1.size(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != partial1.size()) {
            throw std::runtime_error("partial restore");
        }
        if (!llama_memory_seq_rm(mem, 0, 2999, -1) || !llama_memory_seq_rm(mem, 1, 3299, -1)) throw std::runtime_error("rewind");
        dump(ctx, 0, out, "restored"); dump(ctx, 1, out, "restored");
        const auto replay = pair_scores(ctx, tokens, nv, -1);
        exact = compare("partial-replay0", batched[0], replay[0]) && exact;
        exact = compare("partial-replay1", batched[1], replay[1]) && exact;
        if (!llama_memory_seq_rm(mem, 0, 2999, -1) || !llama_memory_seq_rm(mem, 1, 3299, -1)) {
            throw std::runtime_error("mixed rewind");
        }
        const auto mixed = pair_scores(ctx, tokens, nv, -1, false, 17);
        exact = compare("mixed-decode", batched[0], mixed[0]) && exact;
        if (!llama_memory_seq_rm(mem, 1, 3299, -1)) throw std::runtime_error("prefill rewind");
        const auto prefill = pair_scores(ctx, tokens, nv, 1, false, 17);
        for (int i = 0; i < 17; ++i) {
            exact = compare("mixed-prefill", mixed[1 + i], prefill[i]) && exact;
        }
        if (!llama_memory_seq_rm(mem, 1, 0, -1) || !llama_memory_seq_rm(mem, 0, 2999, -1)) {
            throw std::runtime_error("shared rewind");
        }
        llama_memory_seq_cp(mem, 0, 1, 0, -1);
        const auto common = pair_scores(ctx, tokens, nv, -1, true);
        for (int seq = 0; seq < 2; ++seq) {
            if (!llama_memory_seq_rm(mem, seq, 2999, -1)) throw std::runtime_error("shared solo rewind");
            const auto solo = pair_scores(ctx, tokens, nv, seq, true);
            exact = compare("shared-prefix-solo", common[seq], solo[0]) && exact;
        }
        std::printf("{\"partial0_bytes\":%zu,\"partial1_bytes\":%zu}\n", partial0.size(), partial1.size());
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return exact ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
