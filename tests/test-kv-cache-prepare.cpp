#include "models/models.h"
#include "llama-kv-cache.h"
#include "llama-batch.h"
#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

static llama_ubatch batch(const std::vector<std::pair<int, int>> & rows) {
    llama_ubatch out{};
    out.data = std::make_shared<llama_ubatch::data_t>();
    auto & d = *out.data;
    const size_t n = rows.size();
    d.token.resize(n, 1); d.pos.resize(n); d.n_seq_id.resize(n, 1);
    d.seq_id_data.resize(n); d.seq_id.resize(n); d.output.resize(n, 0);
    for (size_t i = 0; i < n; ++i) {
        d.seq_id_data[i] = rows[i].first; d.seq_id[i] = &d.seq_id_data[i]; d.pos[i] = rows[i].second;
        if (std::find(d.seq_id_unq.begin(), d.seq_id_unq.end(), rows[i].first) == d.seq_id_unq.end()) d.seq_id_unq.push_back(rows[i].first);
    }
    out.n_tokens = n; out.n_seq_tokens = n; out.n_seqs = 1;
    out.n_seqs_unq = d.seq_id_unq.size(); out.n_pos = 1;
    out.token = d.token.data(); out.pos = d.pos.data(); out.n_seq_id = d.n_seq_id.data();
    out.seq_id = d.seq_id.data(); out.seq_id_unq = d.seq_id_unq.data(); out.output = d.output.data();
    return out;
}

static bool check(int n_verify, bool fail) {
    llama_model_gemma4 model(llama_model_default_params());
    model.arch = LLM_ARCH_GEMMA4;
    model.hparams.n_layer_all = 0;
    llama_kv_cache cache(model, model.hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true,
        2304, 2, 256, 1024, LLAMA_SWA_TYPE_STANDARD, nullptr, {}, {}, {}, "");
    std::vector<std::pair<int, int>> rows;
    std::vector<uint32_t> indices;
    auto add = [&](int seq, int pos, uint32_t cell) { rows.emplace_back(seq, pos); indices.push_back(cell); };
    add(1, 5122, 0);
    for (int pos = 5123; pos < 6143; ++pos) add(1, pos, pos - 5122);
    add(1, 5120, 2048); add(1, 5121, 2049);
    for (int pos = 7168; pos < 8192; ++pos) add(0, pos, 1021 + pos - 7168);
    auto initial = batch(rows);
    llama_kv_cache::slot_info slot{}; slot.s0 = slot.s1 = 0; slot.strm = {0}; slot.idxs = {indices};
    cache.apply_ubatch(slot, initial);
    const int before_min = cache.seq_pos_min(1);
    const int before_max = cache.seq_pos_max(1);
    const auto before = cache.get_cells(0);
    std::vector<llama_ubatch> ubatches;
    int long_pos = 8192;
    for (int i = 0; i < 4; ++i) {
        rows.clear();
        if (i == 0) for (int pos = 6143; pos < 6143 + n_verify; ++pos) rows.emplace_back(1, pos);
        while (rows.size() < 128) rows.emplace_back(0, long_pos++);
        ubatches.push_back(batch(rows));
    }
    const auto first_slot = cache.find_slot(ubatches[0], false);
    if (fail) {
        rows.clear();
        for (int pos = 9000; pos < 11305; ++pos) rows.emplace_back(0, pos);
        ubatches.push_back(batch(rows));
    }
    const auto planned = cache.prepare(ubatches);
    const auto first_slot_after = cache.find_slot(ubatches[0], false);
    const auto & after = cache.get_cells(0);
    bool exact = planned.size() == (fail ? 0 : 4) && first_slot.idxs == first_slot_after.idxs;
    exact = exact && before.get_used() == after.get_used();
    for (uint32_t i = 0; i < before.size(); ++i) {
        exact = exact && before.pos_get(i) == after.pos_get(i);
        for (int seq = 0; seq < 2; ++seq) exact = exact && before.seq_has(i, seq) == after.seq_has(i, seq);
    }
    std::printf("{\"verify\":%d,\"failed_plan\":%d,\"planned\":%zu,\"before_min\":%d,\"after_min\":%d,\"before_max\":%d,\"after_max\":%d,\"exact\":%d}\n", n_verify, fail, planned.size(), before_min, cache.seq_pos_min(1), before_max, cache.seq_pos_max(1), exact);
    if (exact && !fail) {
        cache.apply_ubatch(planned[0], ubatches[0]);
        exact = cache.seq_pos_min(1) <= 5120 && cache.seq_pos_max(1) == 6142 + n_verify;
        for (size_t i = 1; i < planned.size(); ++i) cache.apply_ubatch(planned[i], ubatches[i]);
        exact = exact && cache.seq_pos_max(1) == 6142 + n_verify;
    }
    return exact;
}

int main() {
    bool exact = true;
    for (const int n_verify : {1, 3, 8}) {
        for (const bool fail : {false, true}) exact = check(n_verify, fail) && exact;
    }
    return exact ? 0 : 1;
}
