#include "replay_buffer.hpp"
#include <random>
#include <fstream>
#include <algorithm>
#include <iostream>

namespace az {

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity_(capacity) {
    buffer_.resize(capacity);
}

void ReplayBuffer::push(const std::vector<float>& state, const std::vector<float>& policy, float value) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_[pos_] = {state, policy, value};
    pos_ = (pos_ + 1) % capacity_;
    if (pos_ == 0) full_ = true;
}

void ReplayBuffer::push_game(const std::vector<Sample>& game_samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& sample : game_samples) {
        buffer_[pos_] = sample;
        pos_ = (pos_ + 1) % capacity_;
        if (pos_ == 0) full_ = true;
    }
}

size_t ReplayBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return full_ ? capacity_ : pos_;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ReplayBuffer::sample_batch(size_t batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t current_size = full_ ? capacity_ : pos_;
    batch_size = std::min(batch_size, current_size);

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);

    std::vector<float> states_vec;
    std::vector<float> policies_vec;
    std::vector<float> values_vec;

    states_vec.reserve(batch_size * NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE);
    policies_vec.reserve(batch_size * NUM_ACTIONS);
    values_vec.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = dist(rng);
        const auto& s = buffer_[idx];
        states_vec.insert(states_vec.end(), s.state.begin(), s.state.end());
        policies_vec.insert(policies_vec.end(), s.policy.begin(), s.policy.end());
        values_vec.push_back(s.value);
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto states_t = torch::from_blob(states_vec.data(),
        {static_cast<int64_t>(batch_size), NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE}, options).clone();
    auto policies_t = torch::from_blob(policies_vec.data(),
        {static_cast<int64_t>(batch_size), NUM_ACTIONS}, options).clone();
    auto values_t = torch::from_blob(values_vec.data(),
        {static_cast<int64_t>(batch_size), 1}, options).clone();

    return {states_t, policies_t, values_t};
}

void ReplayBuffer::save(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return;

    size_t cur_sz = full_ ? capacity_ : pos_;
    out.write(reinterpret_cast<const char*>(&cur_sz), sizeof(cur_sz));
    out.write(reinterpret_cast<const char*>(&capacity_), sizeof(capacity_));

    for (size_t i = 0; i < cur_sz; ++i) {
        const auto& s = buffer_[i];
        out.write(reinterpret_cast<const char*>(s.state.data()), s.state.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(s.policy.data()), s.policy.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(&s.value), sizeof(float));
    }
}

bool ReplayBuffer::load(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ifstream in(filepath, std::ios::binary | std::ios::ate);
    if (!in) return false;

    size_t file_size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    if (file_size < 2 * sizeof(size_t)) return false;

    size_t count = 0, cap = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    in.read(reinterpret_cast<char*>(&cap), sizeof(cap));

    if (count == 0) return true;

    size_t data_size = file_size - 2 * sizeof(size_t);
    constexpr size_t kSample18Bytes = 18 * 64 * sizeof(float) + NUM_ACTIONS * sizeof(float) + sizeof(float); // 23300
    constexpr size_t kSample21Bytes = 21 * 64 * sizeof(float) + NUM_ACTIONS * sizeof(float) + sizeof(float); // 24068

    bool is_18_channels = false;
    if (data_size == count * kSample18Bytes) {
        is_18_channels = true;
        std::cout << "[ReplayBuffer] Detected 18-channel replay buffer (" << count
                  << " samples). Migrating to 21 channels on the fly.\n";
    } else if (data_size == count * kSample21Bytes) {
        is_18_channels = false;
    } else {
        std::cerr << "[ReplayBuffer] Warning: Unexpected replay buffer file size ("
                  << file_size << " bytes for " << count << " samples). Discarding incompatible buffer.\n";
        return false;
    }

    if (count > capacity_) count = capacity_;

    for (size_t i = 0; i < count; ++i) {
        Sample s;
        s.state.assign(NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE, 0.0f);
        s.policy.resize(NUM_ACTIONS);

        if (is_18_channels) {
            in.read(reinterpret_cast<char*>(s.state.data()), 18 * 64 * sizeof(float));
            // Channels 18..20 remain zero-initialized
        } else {
            in.read(reinterpret_cast<char*>(s.state.data()), NUM_CHANNELS * 64 * sizeof(float));
        }

        in.read(reinterpret_cast<char*>(s.policy.data()), s.policy.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(&s.value), sizeof(float));
        buffer_[i] = std::move(s);
    }

    pos_ = count % capacity_;
    full_ = (count == capacity_);
    return true;
}

} // namespace az
