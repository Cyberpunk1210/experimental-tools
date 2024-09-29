import gc

import ggml
import ctypes

params = ggml.ggml_init_params(mem_size=16*1024*1024, mem_buffer=None)
ctx = ggml.ggml_init(params)

x = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 256, 128)
weight = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 128, 32)
bias = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 32)

y = ggml.ggml_add(ctx, ggml.ggml_mul_mat(ctx, x, weight), bias)
tanh = ggml.ggml_div(ctx, ggml.ggml_sub(ctx, ggml.ggml_elu(y), ggml.ggml_elu(ctx, ggml.ggml_neg(y))), ggml.ggml_add(ctx, ggml.ggml_elu(y), ggml.ggml_elu(ggml.ggml_neg(y))))

gf = ggml.ggml_new_graph(ctx)
ggml.ggml_build_forward_expand(gf, tanh)

ggml.ggml_set_2d(x, 2.0)
ggml.ggml_set_2d(weight, 3.0)
ggml.ggml_set_2d(bias, 4.0)

ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)

output = ggml.ggml_get_f32_1d(tanh, 0)


ggml.ggml_free(ctx)
gc.collect()