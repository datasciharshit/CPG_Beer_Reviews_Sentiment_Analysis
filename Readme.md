# Beverage Review Cleaner

An industry-grade, highly optimized Python package for cleaning, translating, and extracting domain-specific features (attributes) from beverage and Consumer Packaged Goods (CPG) product reviews. Tailored for large-scale datasets (e.g. 100K+ reviews) and built with multiprocessing support.

## Key Features

- **Robust Preprocessing**: Regex-based cleaning (emails, URLs, special characters), HTML parsing (BeautifulSoup), and contractions expansion.
- **Multilingual Support**: Integrates `googletrans` with batching, rate-limiting, and auto-retry mechanisms to handle mix-language (English/Japanese) reviews safely.
- **Domain-Specific Aspect Extraction**: Custom spaCy pipeline component to extract sentiment categories like Taste, Mouthfeel, Appearance, and Packaging/Price.
- **High-Performance**: Parallel processing support using `joblib` for high-throughput text cleaning.
- **Enterprise Ready**: Full logging, structured error handling, type hinting, and testing suite.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from beverage_cleaner import ReviewCleaner

cleaner = ReviewCleaner(enable_translation=True, enable_spellcheck=False)

# Clean a single review (translates Japanese automatically)
cleaned = cleaner.clean("このビールは最高です！ The head retention is amazing.")
print(cleaned)
# Output: "this beer is the best ! The head retention is amazing."
```
