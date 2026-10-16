#pragma once

#include <algorithm>
#include <cstdint>

// Reserve the full prompt even if its prefix might be shared. An unbounded
// generation owns the remaining pool; cache hits never increase admission.
inline int64_t server_token_reservation(int64_t prompt, int64_t predict, int64_t capacity, int64_t headroom) {
    if (prompt < 0 || prompt + headroom >= capacity) {
        return capacity + 1;
    }
    return predict < 0 ? capacity : prompt + std::max<int64_t>(1, predict) + headroom;
}

inline bool server_token_fits(int64_t used, int64_t requested, int64_t capacity) {
    return used >= 0 && requested >= 0 && requested <= capacity && used <= capacity - requested;
}
