import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--real_data", type=str, required=True)
parser.add_argument("--fake_data", type=str, required=True)
parser.add_argument("--output_model_path", type=str, default="./discriminator_final")
parser.add_argument("--discri_model_path", type=str, required=True, help="Path to the previous generator model")
args = parser.parse_args()

# Load real and generated data
df_real = pd.read_csv(args.real_data)
df_fake = pd.read_csv(args.fake_data)

# Sample real data with balanced star distribution
real_sample_size = 100
star_ratios = {1: 0.15, 2: 0.10, 3: 0.10, 4: 0.25, 5: 0.40}
real_samples = []
for star, ratio in star_ratios.items():
    n = int(real_sample_size * ratio)
    sampled = df_real[df_real["stars"] == star].sample(n=n, random_state=42)
    real_samples.append(sampled)
df_real = pd.concat(real_samples, ignore_index=True)

# Unify text column
if "text" not in df_real.columns:
    df_real["text"] = df_real["review"]
if "generated_review" in df_fake.columns:
    df_fake["text"] = df_fake["generated_review"]

# Remove missing values
df_real = df_real[df_real["text"].notna()]
df_fake = df_fake[df_fake["text"].notna()]

### Sample 100 fake reviews to match real data
df_fake = df_fake.sample(n=100, random_state=42)

# Assign labels
df_real["label"] = 1
df_fake["label"] = 0

# Combine datasets and format input
data = pd.concat([df_real, df_fake], ignore_index=True)
data["combined_text"] = "Stars: " + data["stars"].astype(str) + " | Review: " + data["text"].astype(str)
data = data[data["combined_text"].str.strip() != ""]  # Remove empty text

# Split into train and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["label"])

# Initialize model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.discri_model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.discri_model_path).to(device)


# Preprocessing
def tokenize_function(examples):
    texts = examples["combined_text"]
    if isinstance(texts, str):
        texts = [texts]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)


# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[["combined_text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["combined_text", "label"]])
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print(f" Discriminator training samples: {len(train_dataset)}")
print(f" Discriminator evaluation samples: {len(test_dataset)}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./discriminator_results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=1e-6,
    weight_decay=0.01
)


# Metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluation report
predictions, labels, _ = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(labels, predicted_labels, target_names=["Generated (Fake)", "Real"]))

# Loss curve
loss_history = trainer.state.log_history
epochs = [x["epoch"] for x in loss_history if "loss" in x]
loss_values = [x["loss"] for x in loss_history if "loss" in x]
plt.plot(epochs, loss_values, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Discriminator Training Loss")
plt.show()

# Save model and tokenizer
model.save_pretrained(args.output_model_path)
tokenizer.save_pretrained(args.output_model_path)
print(f"[Done] Model saved to: {args.output_model_path}")
