#pragma once

#include "model.hpp"
#include "replay_buffer.hpp"
#include "config.hpp"
#include <string>

namespace az {

struct TrainStats {
    float total_loss = 0.0f;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
};

class Trainer {
public:
    Trainer(ChessNet& model, const Config& cfg, torch::Device device);

    TrainStats train_step(ReplayBuffer& buffer, size_t new_samples = 0);

    void save_checkpoint(const std::string& path, int iteration, int total_games);
    int load_checkpoint(const std::string& path, int& out_total_games);

private:
    ChessNet& model_;
    Config config_;
    torch::Device device_;
    std::unique_ptr<torch::optim::SGD> optimizer_;
};

} // namespace az
