import pytest
import spacy
from beverage_cleaner.domain_extraction import (
    BeverageAspectExtractor,
    register_extractor,
    extract_beverage_features,
)

@pytest.fixture(scope="module")
def nlp():
    # Load small spacy model and register extractor component
    model = spacy.load("en_core_web_sm")
    register_extractor(model)
    return model

def test_pipeline_registration(nlp):
    assert "beverage_aspect_extractor" in nlp.pipe_names

def test_aspect_extraction(nlp):
    text = "The head retention is excellent, and the mouthfeel is smooth but it has a bitter taste."
    doc = nlp(text)
    
    # Check that aspects are attached to doc
    assert doc._.beverage_aspects is not None
    aspects = doc._.beverage_aspects
    
    # Verify count and matches
    assert aspects["appearance"]["count"] >= 1 # matches 'head' or 'head retention'
    assert aspects["mouthfeel_texture"]["count"] >= 2 # matches 'mouthfeel', 'smooth'
    assert aspects["taste_flavor"]["count"] >= 2 # matches 'bitter', 'taste'
    
    # Verify exact lemmas
    appearance_lemmas = [m["lemma"] for m in aspects["appearance"]["matches"]]
    assert "head" in appearance_lemmas or "head retention" in appearance_lemmas
    
    mouthfeel_lemmas = [m["lemma"] for m in aspects["mouthfeel_texture"]["matches"]]
    assert "mouthfeel" in mouthfeel_lemmas
    assert "smooth" in mouthfeel_lemmas

def test_multiword_extraction(nlp):
    # Test multiword phrases from DOMAINS, e.g. "six-pack", "dark fruit"
    text = "I bought a six-pack of this stout."
    features = extract_beverage_features(text, nlp)
    
    assert features["packaging_price"]["count"] >= 1
    lemmas = [m["lemma"] for m in features["packaging_price"]["matches"]]
    assert "six-pack" in lemmas
