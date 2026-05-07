"""
Naive Bayes Fake News Classifier
Dataset: Fake.csv / True.csv
No external/machine learning libraries used.
Uses evaluation code from classification_evaluator.py
"""

import csv
import re
import random
from classification_evaluator import ClassificationEvaluator


def load_csv(filepath, label):
    data = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["title"] + " " + row["text"]
            data.append((text, label))
    return data


def load_dataset(fake_path, true_path, sample_size=1000, seed=42):
    random.seed(seed)
    fake = load_csv(fake_path, "fake")
    true = load_csv(true_path, "true")
    n    = min(len(fake), len(true), sample_size)
    fake = random.sample(fake, n)
    true = random.sample(true, n)
    dataset = fake + true
    random.shuffle(dataset)
    return dataset




def tokenize(text):
    # Better tokenizer - removes punctuation properly
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # remove punctuation
    return text.split()


def build_vocab(dataset):
    vocab = set()
    for text, _ in dataset:
        for word in tokenize(text):
            vocab.add(word)
    return vocab


def build_counts(dataset):
    total_words  = {}
    class_counts = {}
    word_counts  = {}
    for text, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
            total_words[label]  = 0
            word_counts[label]  = {}
        class_counts[label] += 1
        for word in tokenize(text):
            total_words[label] += 1
            if word not in word_counts[label]:
                word_counts[label][word] = 0
            word_counts[label][word] += 1
    return total_words, class_counts, word_counts


def compute_priors(class_counts):
    total_docs = sum(class_counts.values())
    return {label: count / total_docs for label, count in class_counts.items()}


def compute_likelihoods(word_counts, total_words, vocab):
    likelihoods = {}
    vocab_size  = len(vocab)
    for label in word_counts:
        likelihoods[label] = {}
        for word in vocab:
            count = word_counts[label].get(word, 0)
            likelihoods[label][word] = (count + 1) / (total_words[label] + vocab_size)
    return likelihoods


def score(text, label, priors, likelihoods, vocab):
    result = priors[label]
    for word in tokenize(text):
        if word in vocab:
            result *= likelihoods[label][word]
    return result


def predict(text, priors, likelihoods, vocab):
    best_label = None
    best_score = -1
    for label in priors:
        s = score(text, label, priors, likelihoods, vocab)
        if s > best_score:
            best_score = s
            best_label = label
    return best_label


def train_test_split(dataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    data = dataset[:]
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


def train(dataset):
    vocab = build_vocab(dataset)
    total_words, class_counts, word_counts = build_counts(dataset)
    priors      = compute_priors(class_counts)
    likelihoods = compute_likelihoods(word_counts, total_words, vocab)
    return priors, likelihoods, vocab


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("data/Fake.csv", "data/True.csv", sample_size=10000)
    print(f"Total samples: {len(dataset)}")

    train_data, test_data = train_test_split(dataset, test_ratio=0.2)
    print(f"Train: {len(train_data)}  |  Test: {len(test_data)}")

    print("\nTraining Naive Bayes...")
    priors, likelihoods, vocab = train(train_data)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Priors: { {k: round(v,3) for k,v in priors.items()} }")

    print("\nEvaluating on test set...")
    label_map = {"fake": 1, "true": 0}
    y_true = [label_map[label] for _, label in test_data]
    y_pred = [label_map[predict(text, priors, likelihoods, vocab)] for text, _ in test_data]

    ev = ClassificationEvaluator(y_true, y_pred)
    r  = ev.binary_report()

    print("\n" + "="*45)
    print("  RESULTS ON FAKE NEWS DATASET")
    print("="*45)
    print(f"  TP        : {r['TP']}")
    print(f"  FP        : {r['FP']}")
    print(f"  FN        : {r['FN']}")
    print(f"  TN        : {r['TN']}")
    print(f"  Accuracy  : {r['accuracy']:.4f}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1        : {r['f1']:.4f}")
    print("="*45)
