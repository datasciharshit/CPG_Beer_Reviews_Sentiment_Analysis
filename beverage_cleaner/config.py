import os
import json
import logging
import re

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONTRACTIONS_PATH = os.path.join(DATA_DIR, "contractions.json")

# Logger setup
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Regular expression patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1+")
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAGS_PATTERN = re.compile(r"<[^>]+>")

# Load contractions
def load_contractions():
    if os.path.exists(CONTRACTIONS_PATH):
        try:
            with open(CONTRACTIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.getLogger("beverage_cleaner.config").warning(
                f"Failed to load contractions.json: {e}. Using empty fallback."
            )
    return {}

CONTRACTIONS = load_contractions()

# CPG Beverage Domain Specific Keywords for Sentiment & Aspect Categorization
# These keywords help maps review phrases/contexts to product attributes (Taste, Mouthfeel, etc.)
DOMAINS = {
    "taste_flavor": [
        "taste", "flavor", "hop", "malt", "sweet", "bitter", "citrus", "sour", 
        "fruity", "roasted", "caramel", "coffee", "chocolate", "notes", "aroma",
        "yeast", "clove", "spice", "banana", "pine", "grapefruit", "lemon", "dark fruit"
    ],
    "mouthfeel_texture": [
        "mouthfeel", "body", "carbonation", "finish", "aftertaste", "smooth", "dry", 
        "watery", "creamy", "crisp", "thick", "thin", "heavy", "light", "texture",
        "fizzy", "flat", "sticky", "oily", "astringent", "warmth"
    ],
    "appearance": [
        "appearance", "head", "retention", "lacing", "color", "pour", "golden", 
        "dark", "brown", "amber", "yellow", "hazy", "clear", "cloudy", "foam",
        "frothy", "opaque", "translucent"
    ],
    "packaging_price": [
        "bottle", "can", "pint", "glass", "packaging", "label", "price", "value", 
        "expensive", "cheap", "buy", "draft", "tap", "cost", "growler", "six-pack"
    ]
}
