#include "model.hpp"

namespace az {

ResBlockImpl::ResBlockImpl(int nf) {
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(nf, nf, 3).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(nf));
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(nf, nf, 3).padding(1).bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(nf));
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    auto out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    return torch::relu(out + x);
}

ChessNetImpl::ChessNetImpl(const Config& cfg) {
    int nf = cfg.num_filters;

    conv_in = register_module("conv_in",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_CHANNELS, nf, 3).padding(1).bias(false)));
    bn_in = register_module("bn_in", torch::nn::BatchNorm2d(nf));

    res_tower = register_module("res_tower", torch::nn::ModuleList());
    for (int i = 0; i < cfg.num_res_blocks; ++i) {
        res_tower->push_back(ResBlock(nf));
    }

    // Policy head
    policy_conv = register_module("policy_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(nf, 73, 1)));

    // Value head
    value_conv = register_module("value_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(nf, 1, 1).bias(false)));
    value_bn = register_module("value_bn", torch::nn::BatchNorm2d(1));
    value_fc1 = register_module("value_fc1", torch::nn::Linear(64, 256));
    value_fc2 = register_module("value_fc2", torch::nn::Linear(256, 1));
}

std::pair<torch::Tensor, torch::Tensor> ChessNetImpl::forward(torch::Tensor x) {
    // Shared trunk
    auto out = torch::relu(bn_in(conv_in(x)));
    for (size_t i = 0; i < res_tower->size(); ++i) {
        out = res_tower[i]->as<ResBlock>()->forward(out);
    }

    // Policy head: (B, 73, 8, 8) -> (B, 73, 64) -> (B, 64, 73) -> (B, 4672)
    auto p = policy_conv(out);
    p = p.reshape({p.size(0), 73, 64});
    p = p.permute({0, 2, 1}).reshape({p.size(0), -1});

    // Value head: (B, 1, 8, 8) -> (B, 64) -> (B, 256) -> (B, 1) in [-1, +1]
    auto v = torch::relu(value_bn(value_conv(out)));
    v = v.view({v.size(0), -1});
    v = torch::relu(value_fc1(v));
    v = torch::tanh(value_fc2(v));

    return {p, v};
}

} // namespace az
