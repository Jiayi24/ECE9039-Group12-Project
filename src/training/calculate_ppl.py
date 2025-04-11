import argparse
import csv
import math
import os

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
model.eval()

# Read generated reviews
df = pd.read_csv(args.input_file)
texts = df["generated_review"].dropna().tolist()


# Perplexity computation function
def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


# Compute PPL for the first 100 samples (for speed)
ppl_scores = [calculate_perplexity(text) for text in texts[:100]]
avg_ppl = sum(ppl_scores) / len(ppl_scores)

print(f"[PPL] Perplexity (avg of {len(ppl_scores)} samples): {avg_ppl:.2f}")

# Append result to log file
log_file = "ppl_log.csv"
write_header = not os.path.exists(log_file)
with open(log_file, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["round", "avg_ppl"])
    round_num = args.model_path.strip().split("gpt2_round")[-1]
    writer.writerow([round_num, avg_ppl])
