from __future__ import annotations

import os
from os import path
from typing import List, Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton


# names = "facebook/opt-125m"
root_data = os.environ["DATASETPATH"]
names = path.join(root_data, "pretrained/opt-125m")
model = AutoModelForCausalLM.from_pretrained(names)
tokenizer = AutoTokenizer.from_pretrained(names)


# TODO Optimize a qwen model inference performance using triton.



if __name__ == "__main__":
    prompt = "Donald Trump is the best president the American in 200 years."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=64)
    print(tokenizer.batch_decode(outputs))




