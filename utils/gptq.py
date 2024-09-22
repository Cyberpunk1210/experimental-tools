import numpy as np

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
