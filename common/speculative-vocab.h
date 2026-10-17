#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

// Bounded history plus a low-ID prefix. IDs stay in the existing draft head.
struct common_speculative_vocab {
    int32_t n_vocab = 0;
    int32_t capacity = 0;
    std::deque<int32_t> recent;

    void configure(int32_t vocab, int32_t count) {
        n_vocab = vocab;
        capacity = count >= 16 && count < vocab ? count : 0;
        recent.clear();
    }

    void observe(int32_t id) {
        if (capacity == 0 || id < 0 || id >= n_vocab) {
            return;
        }
        recent.push_back(id);
        if (recent.size() > 3072) {
            recent.pop_front();
        }
    }

    std::vector<int32_t> select(const int32_t * priority, size_t n_priority) const {
        if (capacity == 0) {
            return {};
        }
        std::vector<uint8_t> seen(n_vocab, 0);
        int32_t selected = 0;
        const auto add = [&](int32_t id) {
            if (id >= 0 && id < n_vocab && !seen[id] && selected < capacity) {
                seen[id] = 1;
                ++selected;
            }
        };
        for (size_t i = 0; i < n_priority; ++i) {
            add(priority[i]);
        }
        for (int32_t i = 0; i < capacity/2; ++i) {
            add(i);
        }
        for (auto it = recent.rbegin(); it != recent.rend(); ++it) {
            add(*it);
        }
        for (int32_t i = 0; selected < capacity; ++i) {
            add(i);
        }
        std::vector<int32_t> ids;
        ids.reserve(capacity);
        // Emit in token-ID order directly; sorting tens of thousands of rows every draft round
        // can erase the savings from the smaller head.
        for (int32_t i = 0; i < n_vocab && ids.size() < size_t(capacity); ++i) {
            if (seen[i]) { ids.push_back(i); }
        }
        return ids;
    }
};
