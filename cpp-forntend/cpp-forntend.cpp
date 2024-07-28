#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <cfloat>
// #include <ATen/ATen.h>
// #include <torch/extension.h>

// struct Net: torch::nn::Module{
//     Net(int64_t N, int64_t M): linear(register_module("linear", torch::nn::Linear(N, M))){
//         another_bias = register_parameter("b", torch::randn(M));
//     }
//     torch::Tensor forward(torch::Tensor input){
//         return linear(input) + another_bias;
//     }
//     torch::nn::Linear linear;
//     torch::Tensor another_bias;
// };

// struct Model : torch::nn::Module {
//     Model(int64_t N, int64_t M)
//      : linear(register_module("linear". torch.nn.Linear(N, M)))
//     {}
//     torch::nn::Linear linear;
// };

// void a(std::shared_ptr<Model> model) {}

// int main()
// {
//     Net net(4, 5);
//     net.to(torch::kCUDA);
//     // for (const auto& pair : net.named_parameters()){
//     //     std::cout << pair.key() << ": " << pair.value() << std::endl;
//     // }
//     std::cout << net.forward(torch::ones({2, 4}).to(torch::kCUDA)) << std::endl;

//     Model model;
//     // auto mode = std::make_shared<Model>();
//     // a(mode);
//     return 0;
// }

struct GPTConfig{
    int n_layer;
    int block_size;
    int vocab_size;
    int n_embd;
    int n_head;
    float n_dropout;
};

struct GPTConfig config = {
    .n_layer = 12,
    .block_size = 1024,
    .vocab_size = 50257,
    .n_embd = 768,
    .n_head = 12,
    .n_dropout = 0.0f
};

struct Attention : torch::nn::Module{
    Attention(GPTConfig config) 
      : c_attn(register_module("c_attn", torch::nn::Linear(config.n_embd, config.n_embd*3))),
        c_proj(register_module("c_proj", torch::nn::Linear(config.n_embd, config.n_embd)))
    {
        register_module("c_attn", c_attn);
        register_module("c_proj", c_proj);
        // bias = register_module("bias", torch::tril(torch::ones({config.block_size, config.block_size})));
    }
    torch::Tensor forward(torch::Tensor x){
        auto sizex = x.sizes();
        auto attx = c_attn(x);
        auto qkv = torch::split(attx, 2);
        auto q = qkv[0].contiguous().view({sizex[0], sizex[1], config.n_head, sizex[2] / config.n_head}).transpose(1, 2);
        auto k = qkv[1].contiguous().view({sizex[0], sizex[1], config.n_head, sizex[2] / config.n_head}).transpose(1, 2);
        auto v = qkv[2].contiguous().view({sizex[0], sizex[1], config.n_head, sizex[2] / config.n_head}).transpose(1, 2);
        auto std = k.sizes();
        auto att = torch::mul(k, v).transpose(-2, -1) * (1.0 / std[-1]);
        auto bias = torch::tril(torch::ones({config.block_size, config.block_size}));
        att = att.masked_fill_(att.select(sizex[1], sizex[1]) == 0, FLT_MIN);
        att = torch::nn::functional::softmax(att, -1);
        auto y = torch::mul(att, v).transpose(1, 2).contiguous().view({sizex});
        y = c_proj(y);
        return y;
    }
    torch::nn::Linear c_attn, c_proj;
    torch::Tensor bias;
};
// TORCH_MODULE(LINEAR);

// Attention attn(config);
int main()
{

    return 0;
}