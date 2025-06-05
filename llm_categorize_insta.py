import os
import json
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import openai
import pandas as pd
from tqdm import tqdm
from more_itertools import chunked
from collections import defaultdict
import re
import zipfile
import random

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CATEGORY SCHEMA ===
ADVERTISER_CATEGORIES = [
    "CulturalEvents", "TechGaming", "FoodLifestyle", "MediaEntertainment",
    "EducationScience", "ShoppingFashion", "TravelNature", "SportsFitness", "Other"
]

CATEGORY_HIERARCHY = {
    "CulturalEvents": {
        "Performing Arts": ["Theater", "Opera", "Dance", "Classical Music"],
        "Visual Arts": ["Art Galleries", "Museums", "Photography", "Sculpture"],
        "Literature": ["Poetry", "Fiction", "Books", "Storytelling"],
        "Festivals": ["Music Festival", "Film Festival", "Ceremonies"],
        "Heritage": ["Tradition", "Folklore", "History"]
    },
    "TechGaming": {
        "Gaming Types": ["RPG", "Action", "Strategy", "MMORPG"],
        "Platforms": ["PC", "Console", "Mobile"],
        "Programming": ["Python", "JavaScript", "AI"],
        "Tech Brands": ["Apple", "Intel", "PlayStation", "Xbox"]
    },
    "FoodLifestyle": {
        "Cuisine": ["Italian", "Sushi", "Baking"],
        "Nutrition": ["Keto", "Vegan", "Low Carb"],
        "Drinks": ["Wine", "Coffee", "Tea"],
        "Dining": ["Restaurants", "Food Trucks"]
    },
    "MediaEntertainment": {
        "TV & Streaming": ["Netflix", "Reality TV"],
        "Movies": ["Film", "Cinema", "Directors"],
        "Music": ["Pop", "Hip Hop", "Rock"],
        "Celebrities": ["Actors", "Influencers"],
        "Comics": ["Anime", "Marvel", "Cartoons"]
    },
    "EducationScience": {
        "Fields": ["Philosophy", "Biology", "Math"],
        "Platforms": ["MOOC", "Online Course"],
        "Communities": ["Conference", "Journal"],
        "Academic": ["University", "School", "Scholarship"]
    },
    "ShoppingFashion": {
        "Fashion": ["Streetwear", "Vintage", "Luxury"],
        "Beauty": ["Skincare", "Makeup"],
        "Shopping": ["Amazon", "eCommerce"],
        "Accessories": ["Jewelry", "Bags", "Shoes"]
    },
    "TravelNature": {
        "Destinations": ["Cities", "Countries", "Beaches"],
        "Nature": ["National Parks", "Camping"],
        "Gear": ["Backpack", "Luggage"],
        "Conservation": ["Wildlife", "Ecotourism"]
    },
    "SportsFitness": {
        "Team Sports": ["Football", "Basketball"],
        "Individual Sports": ["Tennis", "Swimming", "Golf"],
        "Fitness": ["Yoga", "HIIT", "CrossFit"],
        "Spectator": ["Esports", "Streaming Matches"]
    },
    "Other": {
        "Misc": ["Generic", "Spam"]
    }
    }

# === MODELS ===
class AdvertiserCategory(BaseModel):
    advertiser: str
    category: str

# === PROMPTS ===
CATEGORY_PROMPT = """
You are an expert in semantic classification. Given a list of advertiser names, classify each one into one of the following categories:

- Retail
- Technology
- Food & Beverage
- Media & Entertainment
- Education
- Health & Wellness
- Fashion & Beauty
- Finance & Insurance
- Travel
- Other

Respond in JSON format as a list of objects with \"advertiser\" and \"category\".

Here is the list of advertisers:
{advertisers}
"""

SUBCATEGORY_PROMPT_TEMPLATE = """
You are an expert in semantic classification.

Given a list of advertisers already classified under the \"{main_category}\" category, assign a more specific subcategory for each advertiser.

Use the following subcategories as guidance:
{subcategory_hints}

Respond in JSON format as a list of objects with \"advertiser\" and \"subcategory\".

Here is the list of advertisers:
{advertisers}
"""

# === OPENAI CLIENT ===
def get_openai_client(api_key: Optional[str] = None):
    return openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

# === CLASSIFICATION FUNCTIONS ===
def classify_advertisers(advertisers: List[str], model="gpt-3.5-turbo", api_key: Optional[str] = None, batch_size: int = 50) -> List[AdvertiserCategory]:
    client = get_openai_client(api_key)
    results = []
    advertisers = advertisers[:200]
    for batch in chunked(advertisers, batch_size):
        logger.info(f"Classifying batch of {len(batch)} interests")
        prompt = CATEGORY_PROMPT.format(advertisers=batch)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result_text = result_text.strip()

        # Replace common encoding issues
        result_text = result_text.encode('utf-8').decode('unicode_escape', errors='replace')
        
        result_text = re.sub(r",\\s*}", "}", result_text)
        result_text = re.sub(r",\\s*]", "]", result_text)

        try:
            parsed = json.loads(result_text)
            results.extend([AdvertiserCategory(**item) for item in parsed])
        except Exception as e:
            logger.error("Error parsing classification result", exc_info=True)

    return results

def subcategorize_advertisers(classified: List[AdvertiserCategory], api_key: Optional[str] = None, model="gpt-3.5-turbo", batch_size: int = 50) -> List[dict]:
    client = get_openai_client(api_key)
    grouped = defaultdict(list)

    for item in classified:
        grouped[item.category].append(item.advertiser)

    enriched = []

    for main_cat, advertisers in grouped.items():
        subcategories = CATEGORY_HIERARCHY.get(main_cat, {})
        if not subcategories:
            continue

        for batch in chunked(advertisers, batch_size):
            hints = "\n".join([f"- {k}: {', '.join(v)}" for k, v in subcategories.items()])
            prompt = SUBCATEGORY_PROMPT_TEMPLATE.format(
                main_category=main_cat,
                subcategory_hints=hints,
                advertisers=batch
            )

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )

                result_text = response.choices[0].message.content.strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                result_text = result_text.strip()

                result_text = result_text.encode('utf-8').decode('unicode_escape', errors='replace')
                
                result_text = re.sub(r",\\s*}", "}", result_text)
                result_text = re.sub(r",\\s*]", "]", result_text)
                parsed = json.loads(result_text)

                for item in parsed:
                    enriched.append({
                        "advertiser": item["advertiser"],
                        "category": main_cat,
                        "subcategory": item.get("subcategory")
                    })

            except Exception as e:
                logger.error(f"Failed subcategorization for category {main_cat}", exc_info=True)

    return enriched

# === FILE PROCESSING ===
def extract_advertisers_from_json(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [entry['advertiser_name'] for entry in data.get("ig_custom_audiences_all_types", [])]

def process_instagram_json(file_path: str, api_key: str) -> pd.DataFrame:
    advertisers = extract_advertisers_from_json(file_path)
    classified = classify_advertisers(advertisers, api_key=api_key)
    enriched = subcategorize_advertisers(classified, api_key=api_key)
    #return classified
    return pd.DataFrame(enriched), classified

def process_instagram_zip_archives_transiently(zip_folder: str, api_key: str) -> pd.DataFrame:
    final_df = pd.DataFrame(columns=["id", "source", "advertiser", "category", "subcategory"])

    logger.info(f"Scanning ZIP files in: {zip_folder}")

    for file_name in os.listdir(zip_folder):
        if not (file_name.endswith(".zip") and "instagram" in file_name.lower()):
            continue  # skip non-instagram or non-zip files

        zip_path = os.path.join(zip_folder, file_name)
        zip_id = file_name.replace(".zip", "")

        logger.info(f"Processing ZIP: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                for entry in archive.namelist():
                    if entry.endswith("ads_information/advertisers_using_your_activity_or_information.json"):
                        with archive.open(entry) as json_file:
                            content = json.load(json_file)
                            advertisers = [
                                entry['advertiser_name']
                                for entry in content.get("ig_custom_audiences_all_types", [])
                            ]

                            if not advertisers:
                                continue

                            # Sample up to 300 advertisers
                            sampled_advertisers = random.sample(advertisers, min(300, len(advertisers)))

                            classified = classify_advertisers(sampled_advertisers, api_key=api_key)
                            enriched = subcategorize_advertisers(classified, api_key=api_key)

                            df = pd.DataFrame(enriched)
                            df["source"] = "instagram"
                            df["id"] = zip_id
                            final_df = pd.concat([final_df, df], ignore_index=True)

                        break  

        except Exception as e:
            logger.error(f"Failed to process ZIP {zip_path}", exc_info=True)

    logger.info("Finished processing all Instagram ZIP archives.")
    return final_df
