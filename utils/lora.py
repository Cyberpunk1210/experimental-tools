from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LoRALinear(nn.Module):

    def __init__(self, in_feature, out_feature, rank, lora_alpha=1.0, lora_dropout=0, merge_weights=True):
        super(LoRALinear, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dorpout = lora_dropout
        self.merge_weights = merge_weights
        self.merged = False

        if self.rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_feature)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_feature, rank)))
            self.scaling = self.lora_alpha / self.rank
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr("lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        if mode:
            if self.merge_weights and self.merged:
                if self.rank > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.rank > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def forward(self, x: Tensor):
        pass
