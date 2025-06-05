import os
import json
import zipfile
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import openai
import pandas as pd
from tqdm import tqdm
from more_itertools import chunked
from collections import defaultdict


# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CATEGORIES AND HIERARCHY ===
CATEGORIES = [
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
class InterestCategory(BaseModel):
    interest: str
    category: str

# === PROMPTS ===
MAIN_CATEGORY_PROMPT = """
You are an expert in semantic classification. Given a list of user interests, classify each one into one of the following categories:

- CulturalEvents
- TechGaming
- FoodLifestyle
- MediaEntertainment
- EducationScience
- ShoppingFashion
- TravelNature
- SportsFitness
- Other

Respond in JSON format as a list of objects with \"interest\" and \"category\".

Here is the list of interests:
{interests}
"""

SUBCATEGORY_PROMPT_TEMPLATE = """
You are an expert in semantic classification.

Given a list of user interests that have already been classified under the \"{main_category}\" category, assign a more specific subcategory for each interest.

Use the following subcategories and their meanings as guidance:
{subcategory_hints}

Respond in JSON format as a list of objects with \"interest\" and \"subcategory\".

Here is the list of interests:
{interests}
"""

# === OPENAI CLIENT ===
def get_openai_client(api_key: Optional[str] = None):
    return openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

# === LLM FUNCTION WITH BATCHING ===
def classify_interests_with_llm(interests: List[str], model="gpt-3.5-turbo", api_key: Optional[str] = None, batch_size: int = 50) -> List[InterestCategory]:
    client = get_openai_client(api_key)
    results = []

    for batch in chunked(interests, batch_size):
        prompt = MAIN_CATEGORY_PROMPT.format(interests=batch)
        logger.info(f"Classifying batch of {len(batch)} interests")

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
            result_text = result_text[len("```json"):].strip()
        if result_text.endswith("```"):
            result_text = result_text[:-3].strip()

        try:
            parsed = json.loads(result_text)
            results.extend([InterestCategory(**item) for item in parsed])
        except Exception as e:
            logger.error("Error parsing LLM response", exc_info=True)

    return results

def subcategorize_with_llm(classified: List[InterestCategory], api_key: Optional[str] = None, model="gpt-3.5-turbo", batch_size: int = 50, filename=None) -> List[dict]:
    client = get_openai_client(api_key)
    grouped = defaultdict(list)

    for item in classified:
        grouped[item.category].append(item.interest)

    enriched = []

    for main_cat, interests in grouped.items():
        subcategories = CATEGORY_HIERARCHY.get(main_cat, {})
        if not subcategories:
            continue

        for batch in chunked(interests, batch_size):
            hints = "\n".join([f"- {k}: {', '.join(v)}" for k, v in subcategories.items()])
            prompt = SUBCATEGORY_PROMPT_TEMPLATE.format(
                main_category=main_cat,
                subcategory_hints=hints,
                interests=batch
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
                    result_text = result_text[len("```json"):].strip()
                if result_text.endswith("```"):
                    result_text = result_text[:-3].strip()

                parsed = json.loads(result_text)

                for item in parsed:
                    enriched.append({
                        "id": filename,
                        "source": "facebook",
                        "interest": item["interest"],
                        "category": main_cat,
                        "subcategory": item.get("subcategory")
                    })

            except Exception as e:
                logger.error(f"Failed subcategorization for category {main_cat}", exc_info=True)

    return enriched

# === HELPER ===
def extract_interests_from_json(content: dict) -> List[str]:
    found = []

    def recurse(obj):
        if isinstance(obj, dict):
            if obj.get("label") == "Nombre" and "value" in obj:
                found.append(obj["value"])
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(content)
    return found

# === SINGLE FILE TEST FUNCTION ===
def test_single_json_file(json_path: str, api_key: str) -> pd.DataFrame:
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)

    zip_id = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
    interests = extract_interests_from_json(content)
    if interests:
        classified = classify_interests_with_llm(interests, api_key=api_key)
        enriched = subcategorize_with_llm(classified, api_key=api_key, filename=zip_id)
    else:
        columns = ["id", "source", "interest", "category", "subcategory"]
        df = pd.DataFrame(columns=columns)
    return pd.DataFrame(enriched)

def process_all_facebook_zip_files(zip_folder: str, api_key: str) -> pd.DataFrame:
    final_df = pd.DataFrame(columns=["id", "source", "interest", "category", "subcategory"])
    
    logger.info(f"Scanning ZIP files in: {zip_folder}")

    for file_name in os.listdir(zip_folder):
        if not (file_name.endswith(".zip") and "facebook" in file_name.lower()):
            continue  # Skip non-facebook or non-zip files

        zip_path = os.path.join(zip_folder, file_name)
        zip_id = file_name.replace(".zip", "")

        logger.info(f"Processing ZIP: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                for entry in archive.namelist():
                    if entry.endswith("ads_information/ad_preferences.json"):
                        with archive.open(entry) as json_file:
                            content = json.load(json_file)
                            interests = extract_interests_from_json(content)
                            if interests:
                                classified = classify_interests_with_llm(interests, api_key=api_key)
                                enriched = subcategorize_with_llm(
                                    classified, api_key=api_key, filename=zip_id
                                )
                                df = pd.DataFrame(enriched)
                                final_df = pd.concat([final_df, df], ignore_index=True)
                            break  # Process only one ad_preferences.json per zip

        except Exception as e:
            logger.error(f"Failed to process ZIP {zip_path}", exc_info=True)

    logger.info("Completed processing all Facebook ZIP archives.")
    return final_df
