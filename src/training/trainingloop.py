import os

os.environ["WANDB_DISABLED"] = "true"

import subprocess

NUM_ROUNDS = 8
SAMPLES_PER_ROUND = 1000
GEN_MODEL_BASE = "gpt2_round"
DISC_MODEL_BASE = "discriminator_round"

# Step 0: Train initial generator on real data (only once)
if not os.path.exists(f"{GEN_MODEL_BASE}0"):
    print("[Init] Training initial generator (Round 0)...")
    subprocess.run([
        "python", "generatortraining.py",
        "--input_file", "2021_reviews_100k_sampled.csv",
        "--output_model_path", f"{GEN_MODEL_BASE}0"
    ])

# If Round 0 generated data doesn't exist, generate it
if not os.path.exists("generated_round0.csv"):
    print("[Init] Generating initial samples for Round 0...")
    subprocess.run([
        "python", "generator.py",
        "--num_samples", str(SAMPLES_PER_ROUND),
        "--output_file", "generated_round0.csv",
        "--model_path", f"{GEN_MODEL_BASE}0"
    ])

# Step 0+: Train initial discriminator using real + generated data
if not os.path.exists(f"{DISC_MODEL_BASE}0"):
    print("[Init] Training initial discriminator (Round 0)...")
    subprocess.run([
        "python", "discriminatortraining.py",
        "--real_data", "2021_reviews_100k_sampled.csv",
        "--fake_data", "generated_round0.csv",
        "--output_model_path", f"{DISC_MODEL_BASE}0"
    ])

# Start adversarial training loop
for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n===== Round {round_num} =====")

    # Step 1: Generate reviews with previous generator
    print("[1] Generating reviews...")
    gen_output = f"generated_round{round_num}.csv"
    subprocess.run([
        "python", "generator.py",
        "--num_samples", str(SAMPLES_PER_ROUND),
        "--output_file", gen_output,
        "--model_path", f"{GEN_MODEL_BASE}{round_num - 1}"
    ])

    # Step 1.5: Evaluate perplexity
    print("[1.5] Calculating Perplexity...")
    subprocess.run([
        "python", "calculate_ppl.py",
        "--input_file", gen_output,
        "--model_path", f"{GEN_MODEL_BASE}{round_num - 1}"
    ])

    # Step 2: Score generated data using previous discriminator
    print("[2] Scoring with discriminator...")
    disc_output = f"predicted_round{round_num}.csv"
    subprocess.run([
        "python", "discriminator.py",
        "--input_file", gen_output,
        "--output_file", disc_output,
        "--model_path", f"{DISC_MODEL_BASE}{round_num - 1}",
        "--round_num", str(round_num)
    ])

    # Step 3: Fine-tune generator on low-score samples
    print("[3] Fine-tuning generator...")
    new_gen_model_path = f"{GEN_MODEL_BASE}{round_num}"
    subprocess.run([
        "python", "generatortraining.py",
        "--input_file", disc_output,
        "--output_model_path", new_gen_model_path,
        "--top_k_dir", "top20_cache",
        "--model_path", f"{GEN_MODEL_BASE}{round_num - 1}"
    ])

    # Step 4: Train new discriminator on generated + real data
    print("[4] Training discriminator...")
    new_disc_model_path = f"{DISC_MODEL_BASE}{round_num}"
    subprocess.run([
        "python", "discriminatortraining.py",
        "--real_data", "2021_reviews_100k_sampled.csv",
        "--fake_data", gen_output,
        "--output_model_path", new_disc_model_path,
        "--discri_model_path", f"{DISC_MODEL_BASE}{round_num - 1}"
    ])

    print(f"[Done] Round {round_num} completed: Generator → {new_gen_model_path}, Discriminator → {new_disc_model_path}")
