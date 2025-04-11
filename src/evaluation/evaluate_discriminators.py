import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "generated_outputs"
os.makedirs(output_dir, exist_ok=True)

# Generator files (G1, G3, G5, G7 only)
generator_files = {
    f"G{i}": os.path.join(output_dir, f"gpt2_round{i}_samples.csv")
    for i in [1, 3, 5, 7]
}

# Discriminator model directories (D0–D8)
discriminator_models = [f"discriminator_round{i}" for i in range(9)]


# Function: batch prediction with softmax + logit
def batch_predict(texts, model, tokenizer, batch_size=32):
    all_probs, all_logits = [], []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            probs = softmax(model(**batch).logits, dim=-1)
            p_real = probs[:, 1].clamp(min=1e-6, max=1 - 1e-6)
            logits = torch.log(p_real / (1 - p_real))
            all_probs.extend(p_real.cpu().tolist())
            all_logits.extend(logits.cpu().tolist())
    return all_probs, all_logits


# Evaluate each Discriminator against each Generator
logit_results = []

for d_model in discriminator_models:
    print(f"\n Evaluating {d_model}")
    tokenizer = AutoTokenizer.from_pretrained(d_model)
    model = AutoModelForSequenceClassification.from_pretrained(d_model).to(device)
    model.eval()

    row_logit = {"Discriminator": d_model}
    for g_tag, g_file in generator_files.items():
        if not os.path.exists(g_file):
            print(f"[Skipped] {g_file} not found.")
            row_logit[g_tag] = None
            continue

        df = pd.read_csv(g_file)
        df = df[df["generated_review"].notna()]
        df["combined_text"] = "Stars: " + df["stars"].astype(str) + " | Review: " + df["generated_review"]
        texts = df["combined_text"].tolist()

        _, logits = batch_predict(texts, model, tokenizer)
        row_logit[g_tag] = sum(logits) / len(logits)

    logit_results.append(row_logit)

# Save average logit scores to CSV
logit_df = pd.DataFrame(logit_results).set_index("Discriminator")
logit_df.to_csv(os.path.join(output_dir, "avg_logits_G1357.csv"))

# Plot: Logit line plot for G1, G3, G5, G7 across Discriminators
linestyles = ["-", "--", ":", "-."]
selected_generators = list(generator_files.keys())
palette = sns.color_palette("muted", len(selected_generators))

plt.figure(figsize=(10, 6))
for idx, g in enumerate(selected_generators):
    plt.plot(logit_df.index, logit_df[g], label=g, marker="o",
             linestyle=linestyles[idx % len(linestyles)],
             color=palette[idx], linewidth=2.0, markersize=6)

plt.xticks(ticks=range(len(logit_df.index)), labels=[f"D{i}" for i in range(len(logit_df.index))])
plt.title("Logit-Based Discriminator Confidence for Generators G1, G3, G5, G7", fontsize=14)
plt.xlabel("Discriminator Round", fontsize=12)
plt.ylabel("Avg Logit Score (Higher = More Realistic)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Generator", fontsize=10, title_fontsize=11, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "logit_lineplot_G1357_final.png"))
plt.clf()

print("Logit analysis and plot successfully saved.")
