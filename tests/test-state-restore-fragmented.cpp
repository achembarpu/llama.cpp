// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/issues/17527
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix changes find_slot(ubatch, true) to find_slot(ubatch, false)
// in state_read_meta(), allowing non-contiguous slot allocation.

// TODO: merge with test-save-load-state.cpp
// TODO: merge with test-recurrent-state-rollback.cpp

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // init

    llama_backend_init();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    GGML_UNUSED(model);

    // tokenize prompt
    std::vector<llama_token> tokens(70, 1);

    // interleave the 3 sequences:
    // 01201230123...
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            batch.add(tokens[i], i, s, false);
        }
    }
    batch.set_output(batch.size() - 1, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, tokens.size());

    // Save state of seq 1
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 1));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (ncopy != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : saved seq 1 state, %zu bytes\n", __func__, ncopy);

    // clear seq 1 to create a "hole" in the KV cache (fragmentation)
    // 0.20.20.20.2....
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Now the cache has holes where seq 1 was
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 1 state if we only look for contiguous slots

    // Restore seq 1 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (nset != seq_state.size()) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);

    // Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    batch.clear();
    batch.add(next_token, (int)tokens.size(), 1, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        return 1;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    // A single-sequence restore must retain foreign allocation geometry without exposing foreign K/V as live sequences.
    llama_memory_clear(mem, true);
    GGML_ASSERT(llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) == seq_state.size());
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 0) == (llama_pos) tokens.size() - 1);
    GGML_ASSERT(llama_memory_seq_pos_min(mem, 1) == -1);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 2) == -1);

    // MTP rollback uses partial checkpoints. It must not turn reservations into real sequences or lose the owner.
    const auto partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> partial(llama_state_seq_get_size_ext(ctx, 0, partial_flags));
    GGML_ASSERT(llama_state_seq_get_data_ext(ctx, partial.data(), partial.size(), 0, partial_flags) == partial.size());
    batch.clear();
    batch.add(tokens[0], tokens.size(), 0, true);
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    GGML_ASSERT(llama_state_seq_set_data_ext(ctx, partial.data(), partial.size(), 0, partial_flags) == partial.size());
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 0) == (llama_pos) tokens.size() - 1);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 1) == -1);

    // Starting another sequence relinquishes its allocation-only history before attention can use it.
    batch.clear();
    batch.add(tokens[0], 0, 2, true);
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    GGML_ASSERT(llama_memory_seq_pos_min(mem, 2) == 0);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 2) == 0);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 1) == -1);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 0) == (llama_pos) tokens.size() - 1);

    // Explicit owner removal must also release every reservation.
    llama_memory_clear(mem, true);
    GGML_ASSERT(llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) == seq_state.size());
    GGML_ASSERT(llama_memory_seq_rm(mem, 0, -1, -1));
    for (int s = 0; s < params.n_parallel; ++s) {
        GGML_ASSERT(llama_memory_seq_pos_max(mem, s) == -1);
    }
    fprintf(stderr, "%s : allocator reservation remap, rollback, activation and cleanup passed\n", __func__);

    const uint8_t layout_magic[] = {'K', 'V', 'L', '2'};
    auto corrupt = seq_state;
    auto marker = std::search(corrupt.begin(), corrupt.end(), std::begin(layout_magic), std::end(layout_magic));
    GGML_ASSERT(marker != corrupt.end());
    const size_t offset = marker - corrupt.begin();
    const size_t foreign_count_offset = offset + 3*sizeof(uint32_t) + tokens.size()*sizeof(uint32_t) + sizeof(llama_seq_id);
    GGML_ASSERT(foreign_count_offset + sizeof(uint32_t) <= corrupt.size());
    const uint32_t invalid_count = UINT32_MAX;
    memcpy(corrupt.data() + foreign_count_offset, &invalid_count, sizeof(invalid_count));
    GGML_ASSERT(llama_state_seq_set_data(ctx, corrupt.data(), corrupt.size(), 0) == 0);
    for (int s = 0; s < params.n_parallel; ++s) {
        GGML_ASSERT(llama_memory_seq_pos_max(mem, s) == -1);
    }
    GGML_ASSERT(llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 0) == seq_state.size());
    fprintf(stderr, "%s : malformed allocator bounds rejected without reservations\n", __func__);

    llama_sampler_free(smpl);

    return 0;
}
