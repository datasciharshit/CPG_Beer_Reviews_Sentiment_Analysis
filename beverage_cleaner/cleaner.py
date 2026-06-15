import logging
from bs4 import BeautifulSoup
from textblob import Word
from typing import List, Union
import spacy

from beverage_cleaner.config import (
    EMAIL_PATTERN,
    URL_PATTERN,
    REPEATED_CHARS_PATTERN,
    SPECIAL_CHARS_PATTERN,
    WHITESPACE_PATTERN,
    CONTRACTIONS,
)
from beverage_cleaner.translation import RobustGoogleTranslator

logger = logging.getLogger("beverage_cleaner.cleaner")

class ReviewCleaner:
    """
    Main review preprocessor orchestration class.
    Configurable to toggle specific preprocessing stages on/off.
    """
    
    def __init__(
        self,
        enable_translation: bool = True,
        enable_html_removal: bool = True,
        enable_contractions: bool = True,
        enable_emails_urls: bool = True,
        enable_special_chars: bool = True,
        enable_repeated_chars: bool = True,
        enable_lemmatization: bool = True,
        enable_spellcheck: bool = False, # Set to False by default due to latency
        spacy_model: str = "en_core_web_sm",
    ):
        self.enable_translation = enable_translation
        self.enable_html_removal = enable_html_removal
        self.enable_contractions = enable_contractions
        self.enable_emails_urls = enable_emails_urls
        self.enable_special_chars = enable_special_chars
        self.enable_repeated_chars = enable_repeated_chars
        self.enable_lemmatization = enable_lemmatization
        self.enable_spellcheck = enable_spellcheck
        
        # Initialize translator if enabled
        if self.enable_translation:
            self.translator = RobustGoogleTranslator()
            logger.info("Translation module initialized successfully.")
        else:
            self.translator = None

        # Lazily load spaCy model if lemmatization is enabled
        self.nlp = None
        self.spacy_model_name = spacy_model
        if self.enable_lemmatization:
            self._load_spacy()

    def _load_spacy(self):
        """Loads the spaCy pipeline model."""
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"Loaded spaCy model '{self.spacy_model_name}' successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load spaCy model '{self.spacy_model_name}'. "
                f"Please ensure it is downloaded: python -m spacy download {self.spacy_model_name}. Error: {e}"
            )
            raise

    def remove_html_tags(self, text: str) -> str:
        """Removes HTML markup from text using BeautifulSoup."""
        if not text:
            return ""
        try:
            return BeautifulSoup(text, "lxml").get_text()
        except Exception as e:
            # Fallback to simple regex if BeautifulSoup fails
            logger.debug(f"BeautifulSoup HTML removal failed: {e}. Falling back to regex.")
            from beverage_cleaner.config import HTML_TAGS_PATTERN
            return HTML_TAGS_PATTERN.sub("", text)

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions (e.g. don't -> do not)."""
        if not text:
            return ""
        expanded_words = []
        for word in text.split():
            # Strip punctuation for contraction lookup while preserving token case
            clean_word = word.lower().strip(".,?!:;\"'")
            expanded = CONTRACTIONS.get(clean_word, word)
            
            # If word was expanded, attempt to align case (simple check for title/caps)
            if expanded != word:
                if word.isupper():
                    expanded = expanded.upper()
                elif word[0].isupper():
                    expanded = expanded.capitalize()
            expanded_words.append(expanded)
        return " ".join(expanded_words)

    def remove_emails_and_urls(self, text: str) -> str:
        """Removes emails and URLs using precompiled regular expressions."""
        if not text:
            return ""
        text = EMAIL_PATTERN.sub("", text)
        text = URL_PATTERN.sub("", text)
        return text

    def remove_special_characters(self, text: str) -> str:
        """Removes non-alphanumeric characters, keeping standard punctuation if required."""
        if not text:
            return ""
        return SPECIAL_CHARS_PATTERN.sub("", text)

    def normalize_repeated_characters(self, text: str) -> str:
        """Normalizes character repetitions like 'coooool' to 'cool' (max 2 characters)."""
        if not text:
            return ""
        return REPEATED_CHARS_PATTERN.sub(r"\1\1", text)

    def correct_spelling(self, text: str) -> str:
        """Corrects typos in text using TextBlob. Warning: slow for large datasets."""
        if not text:
            return ""
        corrected_words = []
        for word in text.split():
            corrected_words.append(str(Word(word).correct()))
        return " ".join(corrected_words)

    def lemmatize_text(self, text: str) -> str:
        """Reduces words to their base dictionary form (lemma) using spaCy."""
        if not text or not self.nlp:
            return text
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def clean(self, text: str) -> str:
        """
        Runs a single string through the complete configured cleaning pipeline.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Strip HTML
        if self.enable_html_removal:
            text = self.remove_html_tags(text)

        # Step 2: Translation (if needed)
        if self.enable_translation and self.translator:
            text = self.translator.translate(text)

        # Step 3: Expand contractions
        if self.enable_contractions:
            text = self.expand_contractions(text)

        # Step 4: Remove emails and URLs
        if self.enable_emails_urls:
            text = self.remove_emails_and_urls(text)

        # Step 5: Normalize repeated characters
        if self.enable_repeated_chars:
            text = self.normalize_repeated_characters(text)

        # Step 6: Remove special characters
        if self.enable_special_chars:
            text = self.remove_special_characters(text)

        # Step 7: Lemmatization
        if self.enable_lemmatization:
            text = self.lemmatize_text(text)

        # Step 8: Spelling correction
        if self.enable_spellcheck:
            text = self.correct_spelling(text)

        # Final cleanup of extra whitespace
        text = WHITESPACE_PATTERN.sub(" ", text).strip()
        
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Runs a list of strings through the cleaning pipeline.
        Optimized by batch-translating first before proceeding with element-wise cleaning.
        """
        if not texts:
            return []

        # Batch translation to reduce API call overhead and avoid rate limiting
        if self.enable_translation and self.translator:
            logger.info(f"Running batch translation on {len(texts)} texts...")
            texts = self.translator.translate_batch(texts)

        cleaned_texts = []
        for text in texts:
            # Run the remaining steps (translation is skipped inside clean() since it's already translated)
            cleaned_texts.append(self.clean_individually_without_translation(text))

        return cleaned_texts

    def clean_individually_without_translation(self, text: str) -> str:
        """Internal helper to clean text skipping the translation step."""
        if not text or not isinstance(text, str):
            return ""

        text = self.clean_pre_lemmatization(text)
        if self.enable_lemmatization:
            text = self.lemmatize_text(text)
        text = self.clean_post_lemmatization(text)
        return text

    def clean_pre_lemmatization(self, text: str) -> str:
        """Applies initial cleaning steps before lemmatization (HTML, contractions, regex filters)."""
        if not text or not isinstance(text, str):
            return ""
        if self.enable_html_removal:
            text = self.remove_html_tags(text)
        if self.enable_contractions:
            text = self.expand_contractions(text)
        if self.enable_emails_urls:
            text = self.remove_emails_and_urls(text)
        if self.enable_repeated_chars:
            text = self.normalize_repeated_characters(text)
        if self.enable_special_chars:
            text = self.remove_special_characters(text)
        return text

    def clean_post_lemmatization(self, text: str) -> str:
        """Applies final cleaning steps after lemmatization (spelling, extra whitespace)."""
        if not text or not isinstance(text, str):
            return ""
        if self.enable_spellcheck:
            text = self.correct_spelling(text)
        text = WHITESPACE_PATTERN.sub(" ", text).strip()
        return text

    def clean_parallel(self, texts: List[str], n_jobs: int = -1, batch_size: int = 256) -> List[str]:
        """
        Cleans a list of texts in parallel.
        Uses translation in the main process (to avoid rate limits), then runs the remaining CPU-heavy
        cleaning and lemmatization steps using spaCy's built-in nlp.pipe parallelization (which runs multi-core without pickling errors).
        """
        if not texts:
            return []

        # Step 1: Batch translate on main process (I/O bound)
        if self.enable_translation and self.translator:
            logger.info(f"Translating batch of {len(texts)} texts...")
            texts = self.translator.translate_batch(texts)

        # Step 2: CPU-bound cleaning & lemmatization
        if self.enable_lemmatization and self.nlp:
            logger.info(f"Running parallel CPU-bound cleaning & lemmatization (n_jobs={n_jobs}, batch_size={batch_size})...")
            
            # Pre-clean strings before feeding them to spaCy pipeline
            pre_cleaned = [self.clean_pre_lemmatization(t) for t in texts]
            
            # Use spaCy's built-in nlp.pipe for multi-core parallel tokenization & lemmatization
            docs = self.nlp.pipe(pre_cleaned, n_process=n_jobs, batch_size=batch_size)
            
            cleaned_texts = []
            for doc in docs:
                lemmatized = " ".join([token.lemma_ for token in doc])
                cleaned = self.clean_post_lemmatization(lemmatized)
                cleaned_texts.append(cleaned)
                
            return cleaned_texts
        else:
            # If lemmatization is disabled, it is pure regex cleaning.
            # We can use joblib to parallelize since we don't have to serialize any non-pickleable spaCy model.
            from beverage_cleaner.utils import parallel_process
            logger.info(f"Running parallel regex cleaning without lemmatization (n_jobs={n_jobs})...")
            
            def worker(chunk: List[str]) -> List[str]:
                return [self.clean_individually_without_translation(t) for t in chunk]
                
            return parallel_process(texts, worker, n_jobs=n_jobs, chunk_size=batch_size)

    def process_reviews_parallel(
        self,
        texts: List[str],
        n_jobs: int = -1,
        batch_size: int = 256,
    ) -> List[dict]:
        """
        Processes a list of reviews in parallel, performing text cleaning and extracting CPG aspect counts.
        Returns a list of dictionaries, suitable for loading directly into a pandas DataFrame.
        """
        if not texts:
            return []

        # Step 1: Translation in main process (I/O bound)
        if self.enable_translation and self.translator:
            logger.info(f"Translating batch of {len(texts)} texts...")
            texts = self.translator.translate_batch(texts)

        results = []
        if self.enable_lemmatization and self.nlp:
            logger.info(f"Running parallel review cleaning & aspect extraction (n_jobs={n_jobs}, batch_size={batch_size})...")
            pre_cleaned = [self.clean_pre_lemmatization(t) for t in texts]
            
            # Use spaCy nlp.pipe to clean and run custom pipeline aspect extractor component concurrently
            docs = self.nlp.pipe(pre_cleaned, n_process=n_jobs, batch_size=batch_size)
            
            for doc in docs:
                lemmatized = " ".join([token.lemma_ for token in doc])
                cleaned_text = self.clean_post_lemmatization(lemmatized)
                
                # Fetch custom aspects attached by BeverageAspectExtractor
                aspects = doc._.beverage_aspects
                if aspects is None:
                    # Fallback if extractor is not registered
                    aspects = {
                        "taste_flavor": {"count": 0, "matches": []},
                        "mouthfeel_texture": {"count": 0, "matches": []},
                        "appearance": {"count": 0, "matches": []},
                        "packaging_price": {"count": 0, "matches": []},
                    }
                
                res = {
                    "cleaned_text": cleaned_text,
                    "aspect_taste_flavor": aspects["taste_flavor"]["count"],
                    "aspect_mouthfeel_texture": aspects["mouthfeel_texture"]["count"],
                    "aspect_appearance": aspects["appearance"]["count"],
                    "aspect_packaging_price": aspects["packaging_price"]["count"],
                }
                results.append(res)
        else:
            logger.info("Cleaning reviews without lemmatization or aspect extraction...")
            cleaned_texts = self.clean_batch(texts)
            for cleaned in cleaned_texts:
                results.append({
                    "cleaned_text": cleaned,
                    "aspect_taste_flavor": 0,
                    "aspect_mouthfeel_texture": 0,
                    "aspect_appearance": 0,
                    "aspect_packaging_price": 0,
                })

        return results

