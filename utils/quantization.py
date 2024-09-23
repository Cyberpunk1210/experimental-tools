import numpy as np
import torch

QK8_0 = 32
BLOCK_Q8_0 = np.dtype([('d', "<f2"), ("qs", "i1", (QK8_0, ))])
def quantize_array_q8_0(arr):
    assert arr.size % QK8_0 == 0 and arr.size != 0, f"Bad array size {arr.size}"
    assert arr.dtype == np.float32, f"Bad array type {arr.dtype}"
    n_blocks = arr.size // QK8_0
    blocks = arr.reshape((n_blocks, QK8_0))
    return np.fromiter(map(quantize_array_q8_0, blocks), count=n_blocks, dtype=BLOCK_Q8_0)

def quantize_block_q8_0(blk):
    d = abs(blk).max() / np.float32(127)
    if d == np.float32(0):
        return (np.float16(d), np.int8(0),) * QK8_0
    return (np.float16(d), (blk * (np.float32(1) / d)).round())

# Linearity quantization
"""
Q = R / S + Z
R = (Q - Z) * S

R: float num after quant
Q: quantized float number
S: Scalar factor
Z: Zero point number
"""

x = torch.Tensor([[2., 45., -1., -17., -1.],
                  [0., 12.,  3., -63.,  2.],
                  [-1., 37., -1., -83., 0.]]).to(torch.float16)
w = torch.Tensor([[-1., 0.],
                  [2, 0],
                  [0, -2],
                  [3, -2],
                  [-1, 2]]).to(torch.float16)

# 8-bit
# find vector-wise constants: Cw & Cx
Cx = x[:, [0, 2, 4]].abs().max(1).values
Cw = w[[0, 2, 4], :].abs().max(0).values
# quantize
qxI8 = (x[:, [0, 2, 4]] * (127 / Cx)).round().to(dtype=torch.int8)
qwI8 = (w[[0, 2, 4], :] * (127 / Cw)).round().to(dtype=torch.int8)
# matmul
outI32 = (qxI8 @ qwI8).to(dtype=torch.int32)
# dequantize
out8bF16 = outI32 * (Cx.reshape(-1, 1) * Cw.reshape(1, -1)) / 127 * 127

# 16bit
out16bF16 = x[:, [1, 3]] @ w[[1, 3], :]

out = out8bF16 + out16bF16
