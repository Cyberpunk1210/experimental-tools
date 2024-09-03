import torch
import torch.nn as nn
from torch import Tensor


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs: Tensor):
        dims = [-(i+1) for i in range(len(inputs.shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = (inputs - mean)**2
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = y * self.gamma + self.beta
        return out

# TODO implement with C/CUDA