import argparse
import os

import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--model_path", type=str, default="./discriminator_final")
parser.add_argument("--round_num", type=int, required=False, default=0)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
model.eval()

# Load input data
data = pd.read_csv(args.input_file)
if "generated_review" in data.columns:
    data["review"] = data["generated_review"]

# Construct input text
data["combined_text"] = "Stars: " + data["stars"].astype(str) + " | Review: " + data["review"].astype(str)

# Filter out empty or invalid rows
data = data[data["review"].notna()]
data = data[data["review"].str.strip() != ""]

# Reconstruct input after filtering
texts = data["combined_text"].tolist()


# Tokenization in batches
def batch_tokenize(texts, batch_size=32):
    for i in range(0, len(texts), batch_size):
        yield tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt")


# Batch prediction
all_preds = []
all_probs = []
with torch.no_grad():
    for batch in batch_tokenize(texts):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 0].cpu().tolist())  # Class 0 = "Generated"

# Save predictions and probabilities
data["prediction"] = all_preds
data["ai_probability"] = all_probs
data["prediction_label"] = data["prediction"].map({0: "Generated (Fake)", 1: "Real"})

# Save top-20 most human-like samples based on low AI probability
os.makedirs("top20_cache", exist_ok=True)
top20 = data.sort_values("ai_probability").head(20)
top20.to_csv(f"top20_cache/top20_round{args.round_num}.csv", index=False)

# Save full output
data.to_csv(args.output_file, index=False)
print(f"[Done] Discriminator predictions saved to {args.output_file}")
