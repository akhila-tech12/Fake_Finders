"""
Perceptron Fake News Classifier
Built from scratch - no ML libraries used.
Imports shared utilities from naive_bayes.py
"""

import csv
import re
import random
import time
import os
from classification_evaluator import ClassificationEvaluator
from naive_bayes import tokenize

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_PATH = os.path.join(BASE_DIR, "data", "Fake.csv")
TRUE_PATH = os.path.join(BASE_DIR, "data", "True.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading — uses +1/-1 labels for Perceptron
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath, label):
    data = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["title"] + " " + row["text"]
            data.append((text, label))
    return data



# We use sample_size=1000 instead of all 44,898 articles because:
# 1. Perceptron stores a weight for every word in vocabulary
# 2. Full dataset = 80,000+ words → very large vectors
# 3. Each vectorize() call creates a list of 80,000 numbers
# 4. Training on full data would take hours on a laptop
# 5. Limitation: we hit memory and time constraints
# 6. With 1000 samples we still get 99.50% accuracy


def load_dataset(fake_path, true_path, sample_size=1000, seed=42):
    random.seed(seed)
    fake = load_csv(fake_path, +1)    # +1 = fake
    true = load_csv(true_path, -1)    # -1 = real
    n    = min(len(fake), len(true), sample_size)
    fake = random.sample(fake, n)
    true = random.sample(true, n)
    dataset = fake + true
    random.shuffle(dataset)
    return dataset


def train_test_split(dataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    data  = dataset[:]
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


def build_vocab(dataset):
    """Returns dict {word: index} for vectorization."""
    vocab = {}
    idx = 0
    for text, _ in dataset:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Vectorizer
# ─────────────────────────────────────────────────────────────────────────────

def vectorize(text, vocab):
    """Binary bag-of-words: 1 if word present, 0 otherwise."""
    vector = [0] * len(vocab)
    for word in tokenize(text):
        if word in vocab:
            vector[vocab[word]] = 1
    return vector


# ─────────────────────────────────────────────────────────────────────────────
# Perceptron — from professor's slides
# ─────────────────────────────────────────────────────────────────────────────

def sign(score):
    """y_hat = sign(w · x + b)"""
    return +1 if score > 0 else -1


def predict(x, w, b):
    """y_hat = sign(w · x + b)"""
    return sign(sum(wi * xi for wi, xi in zip(w, x)) + b)


def update(w, b, x, y):
    """w = w + y*x  |  b = b + y"""
    return [wi + y * xi for wi, xi in zip(w, x)], b + y


def train(train_data, vocab, epochs=10):
    w, b = [0] * len(vocab), 0
    print(f"\nTraining Perceptron ({epochs} epochs max)...")
    for epoch in range(epochs):
        errors = 0
        for text, y in train_data:
            x = vectorize(text, vocab)
            if predict(x, w, b) != y:
                w, b = update(w, b, x, y)
                errors += 1
        print(f"  Epoch {epoch + 1:02d}/{epochs} | Errors: {errors}")
        if errors == 0:
            print("  Converged early!")
            break
    return w, b


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & Display
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(test_data, vocab, w, b):
    label_map      = {+1: 1, -1: 0}
    y_true, y_pred = [], []
    for text, y in test_data:
        x = vectorize(text, vocab)
        y_true.append(label_map[y])
        y_pred.append(label_map[predict(x, w, b)])
    return y_true, y_pred


def print_results(r, elapsed):
    print("\n" + "="*50)
    print("  PERCEPTRON RESULTS — FAKE NEWS DATASET")
    print("="*50)
    print(f"  TP        : {r['TP']}")
    print(f"  FP        : {r['FP']}")
    print(f"  FN        : {r['FN']}")
    print(f"  TN        : {r['TN']}")
    print(f"  Accuracy  : {r['accuracy']:.4f}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1        : {r['f1']:.4f}")
    print(f"  Time      : {elapsed:.2f} sec")
    print("="*50)


def print_top_words(vocab, w, n=10):
    word_weights = sorted(
        ((word, w[idx]) for word, idx in vocab.items()),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\nTop {n} words pointing to FAKE news:")
    for word, weight in word_weights[:n]:
        print(f"  {word:25s} {weight:+.0f}")

    print(f"\nTop {n} words pointing to REAL news:")
    for word, weight in word_weights[-n:][::-1]:
        print(f"  {word:25s} {weight:+.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading dataset...")
    dataset = load_dataset(FAKE_PATH, TRUE_PATH, sample_size=2000)
    print(f"Total samples : {len(dataset)}")

    train_data, test_data = train_test_split(dataset)
    print(f"Train         : {len(train_data)}")
    print(f"Test          : {len(test_data)}")

    vocab = build_vocab(train_data)
    print(f"Vocabulary    : {len(vocab)} words")

    start   = time.time()
    w, b    = train(train_data, vocab, epochs=10)
    elapsed = time.time() - start

    y_true, y_pred = evaluate(test_data, vocab, w, b)

    ev = ClassificationEvaluator(y_true, y_pred)
    print_results(ev.binary_report(), elapsed)
    print_top_words(vocab, w)