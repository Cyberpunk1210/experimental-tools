#include <iostream>
#include <torch/torch.h>

using namespace nn = torch::nn;


struct RMSNormImpl : nn::Module{
    RMSNormImpl(int hidden_size, float eps=1e-6)
        : weight(nn::Parameter(torch::ones(hidden_size)))
        {
            register_module("weight"),
    }

    torch::Tensor forward(torch::Tensor hidden_states) {
        auto variance = hidden_states.to(torch::kFloat32).pow(2).mean(-1, true);
        hidden_states = hidden_states * torch::rsqrt(variance + eps);
        if ((this->weight.dtype.eq(torch::kFloat16)) || (this->weight.dtype.eq(torch::kBFloat16))) {
            hidden_states = hidden_states.to(this->weight.dtype);
        }
        return this->weight * hidden_states;
    }

    nn::Parameter weight{nullptr};
};
TORCH_MODULE(RMSNorm);