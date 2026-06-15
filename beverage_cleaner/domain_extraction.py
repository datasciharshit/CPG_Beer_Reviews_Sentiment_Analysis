import logging
from typing import Dict, List, Set, Any
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from beverage_cleaner.config import DOMAINS

logger = logging.getLogger("beverage_cleaner.domain_extraction")

@Language.factory("beverage_aspect_extractor")
def create_beverage_aspect_extractor(nlp: spacy.Language, name: str):
    return BeverageAspectExtractor(nlp)

class BeverageAspectExtractor:
    """
    A custom spaCy pipeline component that extracts domain-specific CPG beverage attributes
    (Taste/Flavor, Mouthfeel/Texture, Appearance, Packaging/Price) from reviews.
    Matches tokens based on their lemmatized forms for robust lookup.
    """
    
    name = "beverage_aspect_extractor"

    def __init__(self, nlp: spacy.Language):
        # Register custom extensions on spaCy Doc if they don't already exist
        if not Doc.has_extension("beverage_aspects"):
            Doc.set_extension("beverage_aspects", default=None)
        
        # Build normalized lookups for matching
        # Mapping lowercase lemma -> aspect category
        self.lexicon: Dict[str, str] = {}
        for category, terms in DOMAINS.items():
            for term in terms:
                # We lowercase and split to handle multi-word terms if needed,
                # but for simplicity we index by lemma token.
                self.lexicon[term.lower()] = category

    def __call__(self, doc: Doc) -> Doc:
        """Processes the Doc and attaches the extracted aspect details to Doc._.beverage_aspects."""
        aspect_matches: Dict[str, Dict[str, Any]] = {
            "taste_flavor": {"count": 0, "matches": []},
            "mouthfeel_texture": {"count": 0, "matches": []},
            "appearance": {"count": 0, "matches": []},
            "packaging_price": {"count": 0, "matches": []},
        }

        # Match single-token lemmas
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in self.lexicon:
                category = self.lexicon[lemma]
                aspect_matches[category]["count"] += 1
                aspect_matches[category]["matches"].append({
                    "text": token.text,
                    "lemma": lemma,
                    "index": token.i
                })

        # Match multi-word or hyphenated terms (simple phrase search in text)
        # We also check for phrases like "head retention" or "six-pack" or "dark fruit"
        doc_text_lower = doc.text.lower()
        for category, terms in DOMAINS.items():
            for term in terms:
                if (" " in term or "-" in term) and term in doc_text_lower:
                    # Multi-word match found
                    aspect_matches[category]["count"] += 1
                    aspect_matches[category]["matches"].append({
                        "text": term,
                        "lemma": term,
                        "index": -1 # Special index for phrase match
                    })

        doc._.beverage_aspects = aspect_matches
        return doc


# Function to register the extractor component on an existing spaCy pipeline
def register_extractor(nlp: spacy.Language) -> spacy.Language:
    """Registers the BeverageAspectExtractor component onto the spaCy NLP pipeline."""
    if BeverageAspectExtractor.name not in nlp.pipe_names:
        nlp.add_pipe(BeverageAspectExtractor.name, last=True)
        logger.info(f"Registered spaCy pipeline component '{BeverageAspectExtractor.name}' successfully.")
    return nlp


def extract_beverage_features(text: str, nlp_pipeline: spacy.Language) -> Dict[str, Any]:
    """
    High-level utility to run text through the spaCy pipeline and retrieve aspect dictionaries.
    Ensure register_extractor() was run on the pipeline first.
    """
    if not text:
        return {
            "taste_flavor": {"count": 0, "matches": []},
            "mouthfeel_texture": {"count": 0, "matches": []},
            "appearance": {"count": 0, "matches": []},
            "packaging_price": {"count": 0, "matches": []},
        }
    
    doc = nlp_pipeline(text)
    
    # In case the pipeline was not registered or was modified
    if doc._.beverage_aspects is None:
        # Run local extraction manually without pipeline registration
        extractor = BeverageAspectExtractor(nlp_pipeline)
        doc = extractor(doc)
        
    return doc._.beverage_aspects
