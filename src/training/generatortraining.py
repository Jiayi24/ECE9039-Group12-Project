import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="2021_reviews_100k_sampled.csv")
parser.add_argument("--output_model_path", type=str, default="gpt2_finetuned")
parser.add_argument("--score_threshold", type=float, default=0.7, help="Optional threshold to filter low-score samples")
parser.add_argument("--top_k_dir", type=str, default=None,
                    help="Path to folder containing top20_round*.csv")
parser.add_argument("--model_path", type=str, required=True, help="Path to the previous generator model")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(args.input_file)

is_adversarial = 'ai_probability' in df.columns

if is_adversarial:
    df = df[df['ai_probability'] < args.score_threshold]

    # If top20 files from previous rounds exist, load and concatenate
    if args.top_k_dir:
        top_k_files = sorted(glob.glob(f"{args.top_k_dir}/top20_round*.csv"))
        top_k_dfs = [pd.read_csv(f) for f in top_k_files]
        top_k_data = pd.concat(top_k_dfs, ignore_index=True)

        # Combine current round and top-k data from previous rounds
        df = pd.concat([df, top_k_data], ignore_index=True)

else:
    # For original Yelp dataset, perform class-balanced sampling based on star ratings
    star_counts = df['stars'].value_counts().sort_index()
    max_count = star_counts.max()
    weights = {star: max_count / count for star, count in star_counts.items()}
    df['weight'] = df['stars'].map(weights)
    df = df.sample(n=len(df), weights='weight', replace=True, random_state=42).reset_index(drop=True)

# Remove duplicate reviews based on the generated_review field
df = df.drop_duplicates(subset="generated_review")

# Assign the generated review content to the 'text' field
if 'generated_review' in df.columns:
    df['text'] = df['generated_review']


# Remove repeated sentences within each review (e.g., "Good. Good. Good food." → "Good. Good food.")
def remove_repeated_sentences(text):
    if isinstance(text, str):
        sentences = text.strip().split('. ')
        seen = set()
        cleaned = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                cleaned.append(s)
                seen.add(s)
        return '. '.join(cleaned) + ('.' if cleaned else '')
    return text


df['text'] = df['text'].apply(remove_repeated_sentences)

# Convert to HuggingFace Dataset
hf_dataset = Dataset.from_pandas(df)


# Format input texts
def preprocess_function(examples):
    if 'stars' in examples and 'categories' in examples:
        texts = [f"Stars: {s} | Categories: {c.replace(',', ' &')} | Review: {t}" for s, c, t in zip(examples['stars'], examples['categories'], examples['text'])]
    else:
        texts = examples['text']
    return {"text": texts}


hf_dataset = hf_dataset.map(preprocess_function, batched=True)

# Tokenizer & Model
tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)


# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")


columns_to_remove = list(set(hf_dataset.column_names) - {"text"})
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)

print(f" Generator training samples: {len(tokenized_dataset)}")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training configuration
training_args = TrainingArguments(
    output_dir=os.path.join(args.output_model_path, "checkpoints"),
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
)

# Start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Tokenized dataset column names:", tokenized_dataset.column_names)
print("First sample from the tokenized dataset:", tokenized_dataset[0])
print("Model forward() method accepts the following arguments:", model.forward.__code__.co_varnames)

trainer.train()

# Save final model
os.makedirs(args.output_model_path, exist_ok=True)
trainer.save_model(args.output_model_path)
tokenizer.save_pretrained(args.output_model_path)

print(f"Model saved to: {args.output_model_path}")
