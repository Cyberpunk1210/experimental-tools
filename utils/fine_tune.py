import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForAudioClassification, TrainingArguments, Trainer

model_id = "facebook/opt-125m"
dataset_id = "yelp_review_full"

dataset = load_dataset(dataset_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForAudioClassification.from_pretrained(model_id)

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()