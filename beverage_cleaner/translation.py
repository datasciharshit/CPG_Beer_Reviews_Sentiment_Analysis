import logging
import time
import re
from typing import List, Union
from deep_translator import GoogleTranslator

logger = logging.getLogger("beverage_cleaner.translation")

# Regex to detect Japanese characters (Hiragana, Katakana, Kanji, and full-width symbols)
JAPANESE_CHAR_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\u4e00-\u9faf]"
)

class TranslationError(Exception):
    """Custom exception for translation failures."""
    pass

class BaseTranslator:
    """Abstract base class for review translators, enabling easy swapping of translation engines."""
    
    def translate(self, text: str) -> str:
        raise NotImplementedError("Subclasses must implement translate().")

    def translate_batch(self, texts: List[str]) -> List[str]:
        raise NotImplementedError("Subclasses must implement translate_batch().")


class RobustGoogleTranslator(BaseTranslator):
    """
    A robust Google Translate wrapper using the deep-translator library.
    Implements retries, exponential backoff, and batching.
    """
    
    def __init__(self, retries: int = 3, backoff_factor: float = 1.5):
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._translator = None
        self._init_translator()

    def _init_translator(self, source: str = "auto", target: str = "en"):
        """Initialize or reset the deep-translator client."""
        self._translator = GoogleTranslator(source=source, target=target)

    def needs_translation(self, text: str) -> bool:
        """Checks if the text contains any Japanese characters."""
        if not text:
            return False
        return bool(JAPANESE_CHAR_RE.search(text))

    def translate_single(self, text: str, dest: str = "en") -> str:
        """
        Translates a single string with automatic retry logic.
        """
        if not self.needs_translation(text):
            return text

        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                if self._translator.target != dest:
                    self._translator.target = dest
                res = self._translator.translate(text)
                if res:
                    return res
                raise TranslationError("Translation returned empty result.")
            except Exception as e:
                logger.warning(
                    f"Translation attempt {attempt}/{self.retries} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
                delay *= self.backoff_factor
                self._init_translator(target=dest)  # Recreate translator instance on failure

        logger.error(f"Failed to translate text after {self.retries} attempts.")
        # Fall back to original text to prevent pipeline crash
        return text

    def translate(self, text: str) -> str:
        """Interface implementation for single string translation."""
        return self.translate_single(text)

    def translate_batch(self, texts: List[str], dest: str = "en") -> List[str]:
        """
        Translates a batch of texts. Handles filtering of texts that do not need
        translation to optimize API calls, and runs translation on the rest sequentially
        with delay to avoid googletrans batch issues.
        """
        translated_results = list(texts)  # Start with copy of original texts
        
        for idx, text in enumerate(texts):
            if self.needs_translation(text):
                translated_results[idx] = self.translate_single(text, dest=dest)
                # Sleep briefly between translations to prevent rate limiting
                time.sleep(0.3)

        return translated_results
