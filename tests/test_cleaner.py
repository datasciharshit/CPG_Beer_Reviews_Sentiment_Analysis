import pytest
from beverage_cleaner.cleaner import ReviewCleaner

@pytest.fixture
def default_cleaner():
    # Set enable_translation to False in tests to avoid hitting live network/APIs
    # and enable_lemmatization to False to avoid loading heavy spacy models where not needed
    return ReviewCleaner(enable_translation=False, enable_lemmatization=False)

def test_remove_html_tags(default_cleaner):
    raw_html = "<div>Hello <b>World</b>!</div>"
    cleaned = default_cleaner.remove_html_tags(raw_html)
    assert cleaned.strip() == "Hello World!"

def test_expand_contractions(default_cleaner):
    text = "I don't like this, shouldn't we change it?"
    expanded = default_cleaner.expand_contractions(text)
    assert "I do not like" in expanded
    assert "should not we" in expanded

def test_remove_emails_and_urls(default_cleaner):
    text = "Check out website at https://suntory.com or mail us@suntory.co.jp"
    cleaned = default_cleaner.remove_emails_and_urls(text)
    assert "https://suntory.com" not in cleaned
    assert "us@suntory.co.jp" not in cleaned
    assert "Check out website at" in cleaned

def test_normalize_repeated_characters(default_cleaner):
    text = "This stout is sooooo coooool!"
    normalized = default_cleaner.normalize_repeated_characters(text)
    assert normalized == "This stout is soo cool!"

def test_remove_special_characters(default_cleaner):
    text = "Hello, world! #Suntory-Group."
    cleaned = default_cleaner.remove_special_characters(text)
    assert "#" not in cleaned
    assert "-" not in cleaned
    assert "Hello world SuntoryGroup" in cleaned

def test_full_pipeline_orchestration():
    cleaner = ReviewCleaner(
        enable_translation=False,
        enable_html_removal=True,
        enable_contractions=True,
        enable_emails_urls=True,
        enable_special_chars=True,
        enable_repeated_chars=True,
        enable_lemmatization=False,
    )
    raw = "<p>I didn't like the cheap pricing... at us@suntory.com</p>"
    # 1. HTML removed: I didn't like the cheap pricing... at us@suntory.com
    # 2. Contractions: I did not like the cheap pricing... at us@suntory.com
    # 3. Emails/URLs: I did not like the cheap pricing... at 
    # 4. Repeated chars: I did not like the cheap pricing... at
    # 5. Special chars: I did not like the cheap pricing at
    cleaned = cleaner.clean(raw)
    assert cleaned == "I did not like the cheap pricing at"
