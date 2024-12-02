from __future__ import annotations

import time

import torch
from torch import Tensor

import tabulate
import triton
import triton.language as tl
from triton.runtime import driver

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def native_softmax(x: Tensor) -> Tensor:
    x_max: Tensor = torch.max(x, dim=-1, keepdim=True).values
    numerator: Tensor = torch.exp(x - x_max)
    denominator: Tensor = numerator.sum(-1, keepdim=True)
    ret: Tensor = numerator / denominator
    return ret

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
                   output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr) -> None:
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x: Tensor) -> Tensor:
    samples, feature = x.size()
    BLOCK_SIZE = triton.next_power_of_2(feature)

    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    y = torch.empty_like(x)

    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0),
                                       samples, feature, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared

        if is_hip():
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2
    
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs *  WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, samples)

    kernel[(num_programs, 1, 1)](
        y, x, x.stride(0), y.stride(0), samples, feature
    )
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=[
            "Triton",
            "Torch",
        ],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax_performance",
        args={"M": 4096},
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

def _dropout(
        x_ptr,
        x_keep_ptr,
        output_ptr,
        n_elements,
        p,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # load data
    x = tl.load(x_ptr, offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1-p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    x = torch.randn(1024, 512, device="cuda")  # 2D tensor
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    target = triton.runtime.driver.active.get_current_target()
    kernels = {}

    y_triton = softmax(x)
    y_torch = torch.softmax(x, dim=-1)
    # assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    # benchmark.run(show_plots=True, print_data=True)  # softmax benchmark has been exported as png file.
    print(f"The maximum different between softmax_triton from softmax_naive {torch.max(torch.abs(y_triton - y_torch))}")

    torch.cuda.empty_cache()
    x = torch.randn(size=(10, )).cuda()
    p = 0.5
    x_keep = (torch.rand(size=(10, )) > p).to(torch.int32).cuda()
    output = dropout(x, x_keep=x_keep, p=p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x.tolist(),
        ["output"] + output.tolist()
    ]))
