#include "model.hpp"
#include <iostream>

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

void ChessNetImpl::load(torch::serialize::InputArchive& archive) {
    torch::NoGradGuard no_grad;

    // 1. Load conv_in with backward-compatible weight surgery (18 -> 21 channels)
    torch::serialize::InputArchive conv_archive;
    if (archive.try_read("conv_in", conv_archive)) {
        torch::Tensor saved_weight;
        if (conv_archive.try_read("weight", saved_weight)) {
            if (saved_weight.dim() == 4 && saved_weight.size(1) == 18 && conv_in->weight.size(1) == 21) {
                std::cout << "[ChessNet] Upgrading 18-channel checkpoint to 21 channels: "
                          << "preserving channels 0..17, zero-initializing repetition/halfmove planes 18..20.\n";
                conv_in->weight.zero_();
                conv_in->weight.slice(/*dim=*/1, /*start=*/0, /*end=*/18).copy_(
                    saved_weight.to(conv_in->weight.device()));
            } else {
                conv_in->weight.copy_(saved_weight.to(conv_in->weight.device()));
            }
        }
        torch::Tensor saved_bias;
        if (conv_in->bias.defined() && conv_archive.try_read("bias", saved_bias)) {
            conv_in->bias.copy_(saved_bias.to(conv_in->bias.device()));
        }
    }

    // 2. Load all remaining submodules
    for (auto& child : named_children()) {
        if (child.key() == "conv_in") continue;
        torch::serialize::InputArchive child_archive;
        if (archive.try_read(child.key(), child_archive)) {
            child.value()->load(child_archive);
        }
    }
}

} // namespace az
