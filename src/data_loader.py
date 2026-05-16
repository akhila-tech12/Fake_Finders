"""
data_loader.py
==============
LIAR Dataset loader for fake news detection.

Dataset:
    LIAR: A Benchmark Dataset for Fake News Detection
    Wang, William Yang (2017)
    https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

TSV Column Schema:
    0  : ID
    1  : label  (pants-fire | false | barely-true | half-true | mostly-true | true)
    2  : statement
    3  : subject
    4  : speaker
    5  : speaker job title
    6  : state info
    7  : party affiliation
    8  : barely true counts
    9  : false counts
    10 : half true counts
    11 : mostly true counts
    12 : pants on fire counts
    13 : context / venue

Binary Label Mapping:
    FAKE (+1) : pants-fire, false, barely-true
    REAL (-1) : mostly-true, true
    SKIP      : half-true  (ambiguous — excluded from binary task)

Author : Akhila Pavithran, Rajana
Project: Fake Finders — NLP Master Project, University of Bamberg
"""

import os


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.tsv")
VALID_PATH = os.path.join(BASE_DIR, "data", "valid.tsv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "test.tsv")

FAKE_LABELS = frozenset({"pants-fire", "false", "barely-true"})
REAL_LABELS = frozenset({"mostly-true", "true"})

LABEL_NAMES = {+1: "FAKE", -1: "REAL"}


# ══════════════════════════════════════════════════════════════════════════════
# Label Mapping
# ══════════════════════════════════════════════════════════════════════════════

def map_label(raw_label: str):
    """
    Maps a raw LIAR label string to a binary integer label.

    Args:
        raw_label: One of the six LIAR label strings.

    Returns:
        +1  if fake
        -1  if real
        None if ambiguous (half-true) — caller should skip this example
    """
    label = raw_label.strip().lower()
    if label in FAKE_LABELS:
        return +1
    elif label in REAL_LABELS:
        return -1
    return None


# ══════════════════════════════════════════════════════════════════════════════
# File Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_split(filepath: str) -> list:
    """
    Loads a single LIAR .tsv split into a list of (text, label) tuples.

    Only binary-mappable examples are kept.
    half-true examples are silently skipped.

    Args:
        filepath: Absolute path to the .tsv file.

    Returns:
        List of (statement: str, label: int) tuples.

    Raises:
        FileNotFoundError: If the .tsv file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            f"Make sure train.tsv / valid.tsv / test.tsv are in the data/ folder."
        )

    dataset = []
    skipped = 0

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            raw_label = parts[1].strip()
            statement = parts[2].strip()
            label     = map_label(raw_label)

            if label is None:
                skipped += 1
                continue

            dataset.append((statement, label))

    return dataset


def load_dataset() -> tuple:
    """
    Loads all three LIAR splits (train, valid, test).

    Split Design:
        train → model learns weights from this
        valid → monitor performance during development
        test  → final evaluation only (never touched during training)

    Returns:
        Tuple of (train, valid, test) — each a list of (text, label) tuples.
    """
    print("\n" + "=" * 55)
    print("  LIAR DATASET LOADER")
    print("=" * 55)

    train = load_split(TRAIN_PATH)
    valid = load_split(VALID_PATH)
    test  = load_split(TEST_PATH)

    total = len(train) + len(valid) + len(test)

    print(f"  Train samples : {len(train):>5}")
    print(f"  Valid samples : {len(valid):>5}")
    print(f"  Test  samples : {len(test):>5}")
    print(f"  {'─' * 25}")
    print(f"  Total         : {total:>5}")

    _print_distribution("Train", train)
    _print_distribution("Valid",  valid)
    _print_distribution("Test",  test)

    print("=" * 55)

    return train, valid, test


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def _print_distribution(split_name: str, dataset: list) -> None:
    """Prints class distribution for a given split."""
    fake  = sum(1 for _, y in dataset if y == +1)
    real  = sum(1 for _, y in dataset if y == -1)
    total = len(dataset)
    print(f"\n  {split_name} Distribution:")
    print(f"    Fake (+1) : {fake:4d}  ({100 * fake / total:.1f}%)")
    print(f"    Real (-1) : {real:4d}  ({100 * real / total:.1f}%)")


def preview_dataset(dataset: list, split_name: str = "Dataset", limit: int = 5) -> None:
    """
    Prints a small sample of examples for sanity checking.

    Args:
        dataset   : List of (text, label) tuples.
        split_name: Label for display (e.g. 'Train', 'Test').
        limit     : Number of examples to show.
    """
    print(f"\n=== {split_name} Preview (first {limit} examples) ===")
    for text, label in dataset[:limit]:
        print(f"  [{LABEL_NAMES[label]}] {text[:110]}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point — smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train, valid, test = load_dataset()
    preview_dataset(train, split_name="Train")
    preview_dataset(test,  split_name="Test")