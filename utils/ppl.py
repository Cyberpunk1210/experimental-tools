import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "facebook/opt-125m"
dataset_id = "wikitext"

# prepare
model = AutoModelForCausalLM.from_pretrained(model_id, device=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
test = load_dataset(dataset_id, "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")


# ppl
max_length = model.config.max_length
stride = 512  # if stride is 1, the sliding window strategy will be used.
seq_len = encodings.input_ids.size(1)

nulls = []
pred_end_loc = 0
for begin_loc in tqdm(range(0, seq_len,stride)):
    end_loc = min(max_length + begin_loc, seq_len)
    trg_len = end_loc - pred_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    
    nulls.append(neg_log_likelihood)
    pred_end_loc = end_loc
    if end_loc >= seq_len:
        break
ppl = torch.exp(torch.stack(nulls).mean())
print(f"The perplexity of the {model_id} is {ppl}, lower numbers equate to better preformance.")
