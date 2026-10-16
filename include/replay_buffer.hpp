#pragma once

#include "types.hpp"
#include <vector>
#include <mutex>
#include <torch/torch.h>

namespace az {

struct Sample {
    std::vector<float> state;  // 18 * 8 * 8
    std::vector<float> policy; // 4672
    float value;               // in [-1, 1]
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);

    void push(const std::vector<float>& state, const std::vector<float>& policy, float value);
    void push_game(const std::vector<Sample>& game_samples);

    size_t size() const;
    size_t capacity() const { return capacity_; }

    // Sample a mini-batch of (states, policies, values)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_batch(size_t batch_size);

    void save(const std::string& filepath) const;
    bool load(const std::string& filepath);

private:
    size_t capacity_;
    std::vector<Sample> buffer_;
    size_t pos_ = 0;
    bool full_ = false;
    mutable std::mutex mutex_;
};

} // namespace az
