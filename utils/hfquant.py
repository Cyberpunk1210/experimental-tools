import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GPTQConfig

# bitsandbytes
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

# model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype

model_id = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id, BitsAndBytesConfig(load_in_4bit=True))
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.dequantize()

text = tokenizer("Hello my name is", return_tensors="pt").to(0)
out = model.generate(**text)
print(tokenizer.decode(out[0]))

# gptq
# pip install auto-gptq && pip install --upgrade accelerate optimum transformers
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory={0: "4GiB", "cpu": "30GiB"}, quantization_config=gptq_config)
quantized_model.to("cpu").save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")


# AWQ
# pip install git+https://github.com/casper-hansen/AutoAWQ.git
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

quantization_config = AwqConfig(version="exllama")

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    quantization_config=quantization_config,
    device_map="auto",
)

input_ids = torch.randint(0, 100, (1, 128), dtype=torch.long, device="cuda")
output = model(input_ids)
print(output.logits)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-AWQ")
input_ids = tokenizer.encode("How to make a cake", return_tensors="pt").to(model.device)
output = model.generate(input_ids, do_sample=True, max_length=50, pad_token_id=50256)
print(tokenizer.decode(output[0], skip_special_tokens=True))