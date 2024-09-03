import torch
from torch import Tensor
import torch.nn.functional as F


inp = torch.randn(4, 4)

def softmax(inp: Tensor):
    ex = inp.exp()
    ey = inp.exp().sum(-1, keepdim=True)
    return ex/ey

res = F.softmax(inp, dim=-1)
manual_res = softmax(inp)

print(res)
# tensor([[0.1451, 0.6952, 0.0755, 0.0842],
#         [0.2918, 0.3313, 0.2047, 0.1722],
#         [0.4286, 0.2971, 0.0537, 0.2205],
#         [0.1481, 0.5943, 0.1632, 0.0944]])

print(manual_res)
# tensor([[0.1451, 0.6952, 0.0755, 0.0842],
#         [0.2918, 0.3313, 0.2047, 0.1722],
#         [0.4286, 0.2971, 0.0537, 0.2205],
#         [0.1481, 0.5943, 0.1632, 0.0944]])