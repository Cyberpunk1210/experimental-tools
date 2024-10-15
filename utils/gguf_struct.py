import os
import struct
import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType
from transformers import BitsAndBytesConfig
import bitsandbytes.nn.modules as bnb

prefix = os.environ["DATASETPATH"]
file_path = os.path.join(prefix, "gguf")
if not os.path.exists(file_path):
    os.mkdir(file_path)


inputs = torch.randn(1024, 32, device="cuda", dtype=torch.float32)

model = torch.nn.Sequential(torch.nn.Linear(32, 128, dtype=torch.float16), torch.nn.Linear(128, 32, dtype=torch.float16))

# BitsAndBytes
int8_model = torch.nn.Sequential(bnb.Linear8bitLt(32, 128, has_fp16_weights=False), bnb.Linear8bitLt(128, 32, has_fp16_weights=False))
int8_model.load_state_dict(model.state_dict())
int8_model.to(0)

with torch.no_grad():
    outputs = int8_model(inputs)

# Write with GGUF
gguf_file = os.path.join(file_path, "test.gguf")
gguf_writer = GGUFWriter(gguf_file, "test")
gguf_writer.add_block_count(12)
gguf_writer.add_uint32("answer", 42)  # Write a 32-bit integer
gguf_writer.add_float32("answer_in_float", 42.0)  # Write a 32-bit float
gguf_writer.add_custom_alignment(64)

for names, weight in model.state_dict().items():
    gguf_writer.add_tensor(names, weight, raw_dtype=GGMLQuantizationType.F16)
    gguf_writer.add_tensor(names, weight, raw_dtype=GGMLQuantizationType.F16)
    gguf_writer.add_tensor(names, weight, raw_dtype=GGMLQuantizationType.F16)
    gguf_writer.add_tensor(names, weight, raw_dtype=GGMLQuantizationType.F16)

gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()

gguf_writer.close()

# Read

