#pragma once

#include "types.hpp"
#include "config.hpp"
#include <torch/torch.h>
#include <utility>

namespace az {

class ResBlockImpl : public torch::nn::Module {
public:
    explicit ResBlockImpl(int nf);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
};
TORCH_MODULE(ResBlock);

class ChessNetImpl : public torch::nn::Module {
public:
    explicit ChessNetImpl(const Config& cfg);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv_in{nullptr};
    torch::nn::BatchNorm2d bn_in{nullptr};
    torch::nn::ModuleList res_tower{nullptr};

    // Policy head
    torch::nn::Conv2d policy_conv{nullptr};

    // Value head
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm2d value_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};
};
TORCH_MODULE(ChessNet);

} // namespace az
