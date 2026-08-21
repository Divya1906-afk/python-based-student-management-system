"""
preprocess.py — Text preprocessing for intent classification.

Kept separate from training/inference so the exact same cleaning steps are
guaranteed to run at both train time and inference time (a common bug source
in NLP projects is train/inference preprocessing drifting apart).
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """
    Cleans, tokenizes, and lemmatizes text back into a space-joined string
    (TF-IDF's CountVectorizer/TfidfVectorizer expects string input, so we
    lemmatize then rejoin rather than returning a token list).
    Example: "What are your business hours?" -> "what be your business hour"
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens if tok]
    return " ".join(lemmas)


if __name__ == "__main__":
    samples = [
        "What are your business hours?",
        "I want a REFUND!! Right now.",
        "can you tell me more about this product's features"
    ]
    for s in samples:
        print(f"{s!r:55} -> {tokenize_and_lemmatize(s)!r}")
