import os
import struct
from collections import OrderedDict
import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType
from transformers import BitsAndBytesConfig
import bitsandbytes.nn.modules as bnb

prefix = os.environ["DATASETPATH"]
file_path = os.path.join(prefix, "gguf")
pt_path = os.path.join(prefix, "pt")
gguf_file = os.path.join(file_path, "model.gguf")
pt_file = os.path.join(pt_path, "model.pt")
if not os.path.exists(file_path):
    os.mkdir(file_path)
if not os.path.exists(pt_path):
    os.mkdir(pt_path)


inputs = torch.randn(1024, 32, device="cuda", dtype=torch.float32)

model = torch.nn.Sequential(torch.nn.Linear(32, 128, dtype=torch.float32), torch.nn.Linear(128, 32, dtype=torch.float32))
torch.save(model.state_dict(), pt_file)

# BitsAndBytes
int8_model = torch.nn.Sequential(bnb.Linear8bitLt(32, 128), bnb.Linear8bitLt(128, 32))
# int8_model.load_state_dict(model.state_dict())
int8_model.to(0)

with torch.no_grad():
    outputs = int8_model(inputs)

# Write with GGUF
gguf_writer = GGUFWriter(gguf_file, "model")
gguf_writer.add_block_count(12)
gguf_writer.add_uint32("answer", 42)  # Write a 32-bit integer
gguf_writer.add_float32("answer_in_float", 42.0)  # Write a 32-bit float
gguf_writer.add_custom_alignment(64)

state_dict = model.state_dict()
gguf_writer.add_tensor("0.weight", state_dict["0.weight"].numpy())
gguf_writer.add_tensor("0.bias", state_dict["0.bias"].numpy())
gguf_writer.add_tensor("1.weight", state_dict["1.weight"].numpy())
gguf_writer.add_tensor("1.bias", state_dict["1.bias"].numpy())

gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()

gguf_writer.close()

# Read
model_bin = open(file=gguf_file, mode="rb")
size = os.path.getsize(gguf_file)
tensors_name, tensors_data, tensors_size = [], [], []

# read the magic num
gguf_magic_num = struct.unpack("<I", model_bin.read(4))[0]
gguf_magic = struct.pack("<I", gguf_magic_num)
print(f"gguf_magic: {gguf_magic}")

# read the version of gguf file
gguf_version = struct.unpack("<I", model_bin.read(4))[0]
print(f"gguf_version: {gguf_version}") 

# load tensor information counts
ti_data_count = struct.unpack("<Q", model_bin.read(8))[0]
print(f"ti_data_count: {ti_data_count}")

kv_data_count = struct.unpack("<Q", model_bin.read(8))[0]
print(f"kv_data_count: {kv_data_count}")

key1_len = struct.unpack("<Q", model_bin.read(8))[0]
key1_data = model_bin.read(key1_len).decode("utf-8")
value1_type = struct.unpack("<I", model_bin.read(4))[0]
value1_len = struct.unpack("<Q", model_bin.read(8))[0]
value1_data = model_bin.read(value1_len).decode("utf-8")
print(f"{key1_data} : {value1_data}")

key2_len = struct.unpack("<Q", model_bin.read(8))[0]
key2_data = model_bin.read(key2_len).decode('utf-8')
value2_type = struct.unpack("<I", model_bin.read(4))[0]
value2_data = struct.unpack("<I", model_bin.read(4))[0]
print(f"{key2_data} : {value2_data}")

key3_len = struct.unpack('<Q', model_bin.read(8))[0]
key3_data = model_bin.read(key3_len).decode('utf-8')
value3_type = struct.unpack('<I', model_bin.read(4))[0]
value3_data = struct.unpack('<I', model_bin.read(4))[0]
print(f"{key3_data} : {value3_data}")

key4_len = struct.unpack('<Q', model_bin.read(8))[0]
key4_data = model_bin.read(key4_len).decode('utf-8')
value4_type = struct.unpack('<I', model_bin.read(4))[0]
value4_data = struct.unpack('<f', model_bin.read(4))[0]
print(f"{key4_data} : {value4_data}")

key5_len = struct.unpack('<Q', model_bin.read(8))[0]
key5_data = model_bin.read(key5_len).decode('utf-8')
value5_type = struct.unpack('<I', model_bin.read(4))[0]
value5_data = struct.unpack('<I', model_bin.read(4))[0]
print(f"{key5_data} : {value5_data}")

tensor1_name_len = struct.unpack("<Q", model_bin.read(8))[0]
tensor1_name_data = model_bin.read(tensor1_name_len).decode("utf-8")
tensor1_dim = struct.unpack("<I", model_bin.read(4))[0]
tensor1_shape = []
for i in range(tensor1_dim):
    tensor1_shape.append(struct.unpack("<Q", model_bin.read(8))[0])
tensor1_type = struct.unpack("<I", model_bin.read(4))[0]
tensor1_offset = struct.unpack("<Q", model_bin.read(8))[0]
tensors_name.append(tensor1_name_data)
tensors_size.append(tensor1_shape[::-1])
print(f"{tensor1_name_data}==>dim: {tensor1_dim}, type: {tensor1_type}, shape: {tensor1_shape}, offset: {tensor1_offset}")

tensor2_name_len = struct.unpack('<Q', model_bin.read(8))[0]
tensor2_name_data = model_bin.read(tensor2_name_len).decode('utf-8')
tensor2_dim = struct.unpack('<I', model_bin.read(4))[0]
tensor2_shape = []
for i in range(tensor2_dim):
    tensor2_shape.append(struct.unpack('<Q', model_bin.read(8))[0])
tensor2_type = struct.unpack('<I', model_bin.read(4))[0]
tensor2_offset = struct.unpack('<Q', model_bin.read(8))[0]
tensors_name.append(tensor2_name_data)
tensors_size.append(tensor2_shape[::-1])
print(f"{tensor2_name_data}==>dim: {tensor2_dim}, type:{tensor2_type}, shape: {tensor2_shape}, offset: {tensor2_offset}")

tensor3_name_len = struct.unpack('<Q', model_bin.read(8))[0]
tensor3_name_data = model_bin.read(tensor3_name_len).decode('utf-8')
tensor3_dim = struct.unpack('<I', model_bin.read(4))[0]
tensor3_shape = []
for i in range(tensor3_dim):
    tensor3_shape.append(struct.unpack('<Q', model_bin.read(8))[0])
tensor3_type = struct.unpack('<I', model_bin.read(4))[0]
tensor3_offset = struct.unpack('<Q', model_bin.read(8))[0]
tensors_name.append(tensor3_name_data)
tensors_size.append(tensor3_shape[::-1])
print(f"{tensor3_name_data}==>dim: {tensor3_dim}, type: {tensor3_type}, shape: {tensor3_shape}, offset: {tensor3_offset}")

tensor4_name_len = struct.unpack('<Q', model_bin.read(8))[0]
tensor4_name_data = model_bin.read(tensor4_name_len).decode('utf-8')
tensor4_dim = struct.unpack('<I', model_bin.read(4))[0]
tensor4_shape = []
for i in range(tensor4_dim):
    tensor4_shape.append(struct.unpack('<Q', model_bin.read(8))[0])
tensor4_type = struct.unpack('<I', model_bin.read(4))[0]
tensor4_offset = struct.unpack('<Q', model_bin.read(8))[0]
tensors_name.append(tensor4_name_data)
tensors_size.append(tensor4_shape[::-1])
print(f"{tensor4_name_data}==>dim: {tensor4_dim}, type: {tensor4_type}, shape: {tensor4_shape}, offset: {tensor4_offset}")

n = model_bin.tell()
offset = ((n + 32 - 1) // 32) * 32 - model_bin.tell()
model_bin.seek(n + offset)

tensor1_data = struct.unpack(f'<{32*128}f', model_bin.read(4*32*128))
tensors_data.append(tensor1_data)
print(len(tensor1_data))

tensor2_data = struct.unpack('<128f', model_bin.read(4*128))
tensors_data.append(tensor2_data)
print(len(tensor2_data))

tensor3_data = struct.unpack(f'<{128*32}f', model_bin.read(4*32*128))
tensors_data.append(tensor3_data)
print(len(tensor3_data))

tensor4_data = struct.unpack('<32f', model_bin.read(4*32))
tensors_data.append(tensor4_data)
print(len(tensor4_data))

if model_bin.tell() == size:
    model_bin.close()

# restore
od = OrderedDict()
for n, w, s in zip(tensors_name, tensors_data, tensors_size):
    od[n] = torch.as_tensor(w).reshape(s)
model.load_state_dict(od)
model.to(0)

with torch.no_grad():
    gguf_output = model(inputs)

# torch.allclose(outputs, gguf_output)
# compare pt with gguf
pt_size = os.path.getsize(pt_file)
print(f"Weather gguf model small than pt model: {pt_size>size}")