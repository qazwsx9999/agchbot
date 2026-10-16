#include "trainer.hpp"
#include <c10/core/DeviceGuard.h>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace az {

Trainer::Trainer(ChessNet& model, const Config& cfg, torch::Device device)
    : model_(model),
      config_(cfg),
      device_(device) {
    auto options = torch::optim::SGDOptions(cfg.learning_rate)
                       .momentum(cfg.momentum)
                       .weight_decay(cfg.weight_decay);
    optimizer_ = std::make_unique<torch::optim::SGD>(model_->parameters(), options);
}

TrainStats Trainer::train_step(ReplayBuffer& buffer) {
    c10::DeviceGuard guard(device_);
    model_->to(device_);
    model_->train();

    size_t batch_size = config_.batch_size;
    size_t num_samples = buffer.size();
    size_t steps_per_epoch = std::max(size_t(1), num_samples / batch_size);
    int epochs = std::max(1, config_.num_epochs);

    float sum_total_loss = 0.0f;
    float sum_policy_loss = 0.0f;
    float sum_value_loss = 0.0f;
    long n_batches = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
    for (size_t step = 0; step < steps_per_epoch; ++step) {
        auto [states, policies, values] = buffer.sample_batch(batch_size);

        states = states.to(device_);
        policies = policies.to(device_);
        values = values.to(device_);

        optimizer_->zero_grad();

        auto [pred_policy_logits, pred_values] = model_->forward(states);

        // Policy Loss: Cross-entropy with soft target distribution
        auto log_probs = torch::log_softmax(pred_policy_logits, 1);
        auto loss_pi = -torch::mean(torch::sum(policies * log_probs, 1));

        // Value Loss: MSE
        auto loss_v = torch::mse_loss(pred_values, values);

        auto total_loss = loss_pi + loss_v;
        total_loss.backward();
        optimizer_->step();

        sum_total_loss += total_loss.item<float>();
        sum_policy_loss += loss_pi.item<float>();
        sum_value_loss += loss_v.item<float>();
        n_batches++;
    }
    }

    long denom = (n_batches > 0) ? n_batches : 1;
    TrainStats stats;
    stats.total_loss = sum_total_loss / denom;
    stats.policy_loss = sum_policy_loss / denom;
    stats.value_loss = sum_value_loss / denom;

    return stats;
}

void Trainer::save_checkpoint(const std::string& path, int iteration, int total_games) {
    torch::serialize::OutputArchive archive;
    model_->save(archive);
    archive.write("iteration", torch::tensor(iteration));
    archive.write("total_games", torch::tensor(total_games));
    archive.save_to(path);

    // Persist optimizer state (SGD momentum buffers) so training is fully
    // resumable, exactly like the Python checkpoint.
    torch::serialize::OutputArchive opt_archive;
    optimizer_->save(opt_archive);
    opt_archive.save_to(path + ".opt");
}

int Trainer::load_checkpoint(const std::string& path, int& out_total_games) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        model_->load(archive);
        model_->to(device_);

        // Re-bind optimizer with the loaded parameters on the target device
        auto options = torch::optim::SGDOptions(config_.learning_rate)
                           .momentum(config_.momentum)
                           .weight_decay(config_.weight_decay);
        optimizer_ = std::make_unique<torch::optim::SGD>(model_->parameters(), options);

        torch::Tensor iter_t, games_t;
        int iter = 0;
        if (archive.try_read("iteration", iter_t)) {
            iter = iter_t.item<int>();
        }
        if (archive.try_read("total_games", games_t)) {
            out_total_games = games_t.item<int>();
        }

        // Restore optimizer state (momentum buffers) if present.
        std::string opt_path = path + ".opt";
        if (std::ifstream(opt_path).good()) {
            try {
                torch::serialize::InputArchive opt_archive;
                opt_archive.load_from(opt_path);
                optimizer_->load(opt_archive);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not restore optimizer momentum: " << e.what() << "\n";
            }
        }
        return iter;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load checkpoint: " << e.what() << "\n";
        return 0;
    }
}

} // namespace az
