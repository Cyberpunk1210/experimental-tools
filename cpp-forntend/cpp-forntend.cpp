#include <iostream>
#include <torch/torch.h>

struct Net: torch::nn::Module{
    Net(int64_t N, int64_t M): linear(register_module("linear", torch::nn::Linear(N, M))){
        another_bias = register_parameter("b", torch::randn(M));
    }
    torch::Tensor forward(torch::Tensor input){
        return linear(input) + another_bias;
    }
    torch::nn::Linear linear;
    torch::Tensor another_bias;
};

int main()
{
    Net net(4, 5);
    net.to(torch::kCUDA);
    for (const auto& p : net.parameters()){
        std::cout << p << std::endl;
    }
    return 0;
}
