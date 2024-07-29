#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <cfloat>


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

class AttentionImpl : public torch::nn::Module{
public:
    AttentionImpl(GPTConfig config);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Linear c_attn{ nullptr };
    torch::nn::Linear c_proj{ nullptr };
    torch::Tensor bias{ nullptr };
};

AttentionImpl::AttentionImpl(GPTConfig config){
    c_attn = register_module("c_attn", torch::nn::Linear(config.n_embd, config.n_embd * 3));
    c_proj = register_module("c_proj", torch::nn::Linear(config.n_embd, config.n_embd));
    // bias = register_module("bias", torch::tril(torch::ones({config.block_size, config.block_size})));
}
    torch::Tensor AttentionImpl::forward(torch::Tensor x){
        auto sizex = x.sizes();
        auto attx = c_attn->forward(x);
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
        y = c_proj->forward(y);
        return y;
}


int main()
{
    return 0;
}