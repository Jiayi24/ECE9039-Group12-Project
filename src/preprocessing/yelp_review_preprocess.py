import json

import pandas as pd

with open("yelp_academic_dataset_business.json", "r", encoding="utf-8") as f:
    business_data = [json.loads(line) for line in f]

# Build a mapping from business_id to business info
business_dict = {}
target_business_ids = set()

for business in business_data:
    business_id = business["business_id"]
    categories = business.get("categories", "")
    if categories:
        categories_list = [c.strip() for c in categories.split(",")]
        if "Restaurants" in categories_list and "Food" in categories_list:
            target_business_ids.add(business_id)
            business_dict[business_id] = {
                "categories": categories
            }

# Load review.json and filter 2021 reviews
print("Loading review.json and filtering 2021 reviews...")

with open("yelp_academic_dataset_review.json", "r", encoding="utf-8") as f:
    review_data = [json.loads(line) for line in f]

reviews_2021 = []

for review in review_data:
    business_id = review["business_id"]
    if review["date"].startswith("2021") and business_id in target_business_ids:
        business_info = business_dict.get(business_id, {})
        reviews_2021.append({
            "stars": review["stars"],
            "text": review["text"],
            **business_info  # Add business category info
        })

# Convert to DataFrame and save all matched data
df = pd.DataFrame(reviews_2021)
print("\nTotal number of matched reviews from 2021:", len(df))

df.to_csv("reviews_2021_with_business.csv", index=False, encoding="utf-8")
print("Saved all filtered reviews to reviews_2021_with_business.csv")

# Randomly sample 100,000 rows and save remaining data
print("Sampling 100,000 reviews...")

df_sampled = df.sample(n=100000, random_state=42)
df_remaining = df.drop(df_sampled.index)

df_sampled.to_csv("2021_reviews_100k_sampled.csv", index=False, encoding="utf-8")
df_remaining.to_csv("2021_reviews_remaining.csv", index=False, encoding="utf-8")

print("Saved 100,000 sampled reviews to 2021_reviews_100k_sampled.csv")
print(f"Saved the remaining {len(df_remaining)} reviews to 2021_reviews_remaining.csv")
