import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropy(nn.Module):
    @staticmethod
    def forward(inputs: Tensor, target: Tensor):
        *_, C = inputs.size()
        maxes = inputs.max(-1, keepdim=True).values
        norm_logits = inputs - maxes
        counts = norm_logits.exp()
        probs = counts * counts.sum(1, keepdim=True)**-1
        logprobs = probs.log()
        loss = -logprobs[range(C), target].mean()
        return loss

# loss = F.cross_entropy(inputs, target)

