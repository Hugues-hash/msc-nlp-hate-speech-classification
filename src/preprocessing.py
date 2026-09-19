"""Text cleaning and vocabulary helpers for the hate-speech dataset.

`clean_text` normalises a raw tweet (drops links, mentions, retweet markers and punctuation,
lowercases, optionally removes stopwords, then lemmatises). `build_vocab` / `encode_texts`
turn cleaned text into the padded integer sequences the LSTM expects.
"""

import re
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download the two NLTK resources we rely on the first time this module is imported
for _pkg in ["wordnet", "stopwords", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

lemmatiser = WordNetLemmatizer()          # turns words into their base form
stop_words = set(stopwords.words("english"))


def clean_text(text, remove_stopwords=True):
    """Take a raw tweet and return a cleaned, lemmatised version of it."""
    text = re.sub(r"http\S+|www\S+", "", text)   # drop links
    text = re.sub(r"@\w+", "", text)             # drop @mentions
    text = re.sub(r"RT\s*", "", text)            # drop the leading 'RT' on retweets
    text = re.sub(r"&amp;", "and", text)         # decode the HTML ampersand
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # keep letters and spaces only
    text = text.lower().strip()
    tokens = text.split()
    if remove_stopwords:                         # only for the traditional ML models
        tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatiser.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def build_vocab(texts, max_vocab=15000):
    """Build a word -> integer id mapping from the training texts (id 0 = PAD, id 1 = UNK)."""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    most_common = word_counts.most_common(max_vocab - 2)   # leave room for PAD and UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


def encode_texts(texts, vocab, max_len=100):
    """Turn a list of texts into a numpy array of padded integer sequences."""
    encoded = []
    for text in texts:
        tokens = text.split()
        ids = [vocab.get(w, 1) for w in tokens[:max_len]]  # unknown words map to <UNK> (id 1)
        ids = ids + [0] * (max_len - len(ids))             # pad with <PAD> (id 0) on the right
        encoded.append(ids)
    return np.array(encoded)
