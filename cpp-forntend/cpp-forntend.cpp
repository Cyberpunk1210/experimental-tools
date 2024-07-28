#include <iostream>
#include <torch/torch.h>

struct Net: torch::nn::Module{
    Net(int64_t N, int64_t M){
        W = register_parameter("W", torch::randn({N, M}));
        b = register_parameter("b", torch::randn(M));
    }

    torch::Tensor forward(torch::Tensor input){
        return torch::addmm(b, input, W);
    }
    torch::Tensor W, b;
};

int main()
{
    Net net(4, 5);
    for (const auto& p : net.parameters()){
        std::cout << p << std::endl;
    }
    return 0;
}
