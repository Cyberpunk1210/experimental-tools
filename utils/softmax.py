from __future__ import annotations

from typing import Any, TypeVar
from functools import lru_cache
import torch
import triton
import triton.language as tl
from torch import Tensor
import torch.nn.functional as F
import time


inp = torch.randn(10240, 10240).to("cuda:0")
inp_Y = torch.empty_like(inp)

@lru_cache
def softmax(inp: Tensor) -> Tensor:
    ex: Tensor = (inp-inp.max(-1, keepdim=True).values).exp()
    return ex/ex.sum(-1, keepdim=True)

@triton.jit
def tri_softmax(inp_X, int_Y, n_elements: int, BLOCK_SIZE: tl.constexpr) -> None:
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    X = tl.load(inp_X + offsets, mask=mask)
    max_val = tl.max(X, axis=-1)
    X_exp = tl.exp(X - max_val)
    sum_val = tl.sum(X_exp, axis=-1)
    Y = X_exp / sum_val
    tl.store(int_Y + offsets, Y, mask=mask)


start_time = time.time()
res = F.softmax(inp, dim=-1)
print(f"Elapsed time of functional: {(time.time() - start_time):.4f}")

start_time = time.time()
manual_res = softmax(inp)
print(f"Elapsed time of manual: {(time.time() - start_time):.4f}")

start_time = time.time()
tri_softmax[(inp.numel() // 64,)](inp, inp_Y, inp.numel(), BLOCK_SIZE=64)
print(f"Elapsed time of triton: {(time.time() - start_time):.4f}")

# tensor([[0.1451, 0.6952, 0.0755, 0.0842],
#         [0.2918, 0.3313, 0.2047, 0.1722],
#         [0.4286, 0.2971, 0.0537, 0.2205],
#         [0.1481, 0.5943, 0.1632, 0.0944]])

# print(manual_res)
# tensor([[0.1451, 0.6952, 0.0755, 0.0842],
#         [0.2918, 0.3313, 0.2047, 0.1722],
#         [0.4286, 0.2971, 0.0537, 0.2205],
#         [0.1481, 0.5943, 0.1632, 0.0944]])
assert torch.allclose(res, manual_res)