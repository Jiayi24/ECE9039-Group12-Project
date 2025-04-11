import argparse
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--output_file", type=str, default="generated.csv")
parser.add_argument("--model_path", type=str, default="gpt2")
parser.add_argument("--prompt", type=str, default="The restaurant")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
model.eval()

print("Model loaded on:", next(model.parameters()).device)

# Predefined categories 50
categories_samples = [
    "restaurants & food & nightlife", "coffee & tea & breakfast & brunch & cafes",
    "bars & cocktail bars & wine bars & beer bar & sports bars",
    "sandwiches & delis & wraps & cheesesteaks", "american & comfort food & southern",
    "desserts & ice cream & frozen yogurt & cupcakes & chocolatiers & shops",
    "bakeries & pastries & patisserie/cake shop & custom cakes & creperies & waffles",
    "specialty food & grocery & convenience stores & farmers market",
    "burgers & fast food & hot dogs & chicken wings",
    "wine & spirits & breweries & distilleries & beer gardens & whiskey bars & champagne bars",
    "seafood & sushi bars & cajun/creole & fish & chips",
    "pizza & italian & pasta shops & brasseries & mediterranean",
    "mexican & tacos & tex-mex & latin american",
    "asian fusion & japanese & ramen & korean & thai & vietnamese",
    "vegan & vegetarian & gluten-free & live/raw food & acai bowls",
    "diners & buffets & food court",
    "food delivery services & food trucks & food stands & street vendors & pop-up restaurants",
    "steakhouses & barbeque & smokehouse",
    "poke & hawaiian & seafood markets",
    "caterers & event planning & services & venues & event spaces & party & event planning",
    "indian & pakistani & afghan & middle eastern & lebanese & turkish",
    "french & spanish & modern european & tapas bars & tapas/small plates",
    "chinese & dim sum & cantonese & szechuan",
    "cuban & puerto rican & venezuelan & peruvian & argentine & brazilian & colombian & honduran",
    "bagels & donuts & pretzels & macarons & shaved ice",
    "juice bars & smoothies & bubble tea & kombucha & tea rooms",
    "beer & brewpubs & wineries & wine tasting room",
    "hotels & travel & hotels & resorts & shared office spaces",
    "cooking classes & personal chefs & dinner theater",
    "arts & entertainment & music venues & performing arts & jazz & blues & karaoke",
    "Seafood, Wine Bars, Restaurants, Cajun/Creole, Beer, Tacos",
    "Coffee & Tea, Sandwiches, Bagels, Breakfast & Brunch, Organic Stores",
    "Desserts, Ice Cream & Frozen Yogurt, Restaurants, Pop-Up Restaurants, Cafes",
    "Pizza, Pasta Shops, Italian, Fast Food, Restaurants, Bars",
    "Vietnamese, Noodles, Asian Fusion, Food Trucks, Restaurants, Thai",
    "Mexican, Juice Bars & Smoothies, Beer Gardens, Nightlife, Restaurants, Latin American",
    "French, Tapas Bars, Wine & Spirits, Restaurants, Modern European, Bars",
    "Bubble Tea, Bakeries, Vegan, Coffee Roasteries, Breakfast & Brunch, Restaurants",
    "Southern, Comfort Food, Chicken Wings, Restaurants, Food, Nightlife",
    "Grocery, Specialty Food, Convenience Stores, Food Delivery Services, Restaurants",
    "Sushi Bars, Ramen, Japanese, Restaurants, Bars, Seafood",
    "Cocktail Bars, Whiskey Bars, Nightlife, Bars, Music Venues, Beer",
    "Burgers, Food Trucks, Fast Food, American (Traditional), Restaurants, Delis",
    "Dim Sum, Chinese, Szechuan, Cantonese, Restaurants, Tea Rooms",
    "Steakhouses, Smokehouse, Barbeque, Restaurants, Food, Beer",
    "Middle Eastern, Turkish, Halal, Mediterranean, Food, Restaurants",
    "Pubs, Breweries, Distilleries, Bars, Food, Restaurants",
    "Waffles, Donuts, Pastries, Cupcakes, Coffee & Tea, Bakeries",
    "Farmers Market, Local Flavor, Organic Stores, Food, Restaurants, Grocery",
    "Tapas/Small Plates, Dinner Theater, Performing Arts, Restaurants, Event Planning"
]

# Star rating distribution
star_ratios = {1: 0.15, 2: 0.10, 3: 0.10, 4: 0.25, 5: 0.40}
star_counts = {star: int(args.num_samples * ratio) for star, ratio in star_ratios.items()}


# Generation function
def generate_review(star, categories, max_length=200, temperature=0.8):
    prompt = f"Stars: {star} | Categories: {categories} | Review:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=inputs["input_ids"].shape[1] + max_length,
        do_sample=True, top_k=50, top_p=0.95, temperature=temperature,
        repetition_penalty=1.2,  # Try 1.1–1.5
        no_repeat_ngram_size=3,  # Prevents repeating any 3-word phrase
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    review = output_text.split("Review:")[-1].strip()
    return review.rsplit(".", 1)[0] + "." if "." in review else review


# Generation loop
print("Generating reviews...")
generated_data = []
for star, count in star_counts.items():
    for _ in tqdm(range(count), desc=f"Stars: {star}"):
        cat = random.choice(categories_samples)
        review = generate_review(star, cat)
        generated_data.append({"stars": star, "categories": cat, "generated_review": review})

# Save results
pd.DataFrame(generated_data).to_csv(args.output_file, index=False)
print(f"Saved to {args.output_file}")
