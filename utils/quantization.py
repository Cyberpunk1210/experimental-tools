import os
import functools

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer, OPTForCausalLM, Int8OPTForCausalLM
from smoothquant import smooth_lm


model_name = "facebook/opt-13b"
act_scales = "act_scales/opt-13b.pt"
dataset_path = "val.json.zst"
output_path = "int8_models"

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
"""
scaler = 127/max(abs(w))
w = scaler * round(w / scaler)
"""
def quantize_weight_per_channel_absmax(w: Tensor, n_bits: int=8):
    wscaler = w.abs().max(-1, keepdim=True)[0]
    q_max = 2<<(n_bits-1) - 1
    wscaler.clamp_(min=1e-5).div_(q_max)
    w.div_(wscaler).round_().mul_(wscaler)
    return w

def quantize_activation_per_tensor_absmax(t: Tensor, n_bit: int=8):
    t_shape = t.shape
    t.view(-1, t_shape[1])
    tscaler = t.abs().max()
    q_max = 2<<(n_bit-1) - 1
    tscaler.clamp_(min=1e-5).div_(q_max)
    t.div_(tscaler).round_().mul_(tscaler)
    return t


def get_act_scales(model: torch.nn.Module, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {} # save the result of the activations statictics(max absolute)

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x) # statictics the result of input data

    hooks = []
    for name, m in model.name_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    # get verify data
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    # scales factor
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()
    return act_scales

model = OPTForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
act_scales = torch.load(act_scales)
smooth_lm(model, act_scales, 0.5)

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    # 计算权重的最大绝对值
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    # 按照公式(4)计算S
    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)
    # 把激活X的scales提前放到前一层的layer_norm的权重中去
    ln.weight.div_(scales)
    ln.bias.div_(scales)
    # 权重直接乘以S
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))




