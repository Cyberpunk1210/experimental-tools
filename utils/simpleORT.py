import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import onnx

x = torch.randn(1024, 32)

model = nn.Sequential(
    nn.Linear(32, 128, bias=True),
    nn.Linear(128, 32, bias=True),
)

torch.nn.init.normal_(model[0].weight)
torch.nn.init.kaiming_normal_(model[1].weight)

yhat = model(x)
# print(yhat)

torch.onnx.export(model, 
                  torch.randn(1, 32),
                  "fashion_model.onnx",
                  input_names=["input"],
                  output_names=["output"])

onnx_model = onnx.load("fashion_model.onnx")
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession("fashion_model.onnx")
input_name = ort_sess.get_inputs()[0].name
expr = list(ort_sess.run(None, {input_name: i[None, ...]})[0] for i in x.numpy())
onnx_yhat = np.concatenate(expr, axis=0)

assert np.allclose(onnx_yhat, yhat.detach().numpy()) # "True"
