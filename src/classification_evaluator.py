"""
classification_evaluator.py
============================
Evaluation framework for binary and multi-class text classification.

Metrics implemented from scratch (no external libraries):
    Binary    : TP, FP, FN, TN, accuracy, precision, recall, F1
    Multiclass: per-class precision / recall / F1 + macro averages

Usage:
    from classification_evaluator import ClassificationEvaluator

    ev = ClassificationEvaluator(y_true, y_pred)
    report = ev.binary_report()
    print(report)

Author : Akhila Pavithran, Rajana
Project: Fake Finders — NLP Master Project, University of Bamberg
"""


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator Class
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationEvaluator:
    """
    Computes classification metrics for binary and multi-class tasks.

    All metrics are computed from scratch using only Python built-ins.

    Args:
        y_true : List of ground-truth labels.
        y_pred : List of predicted labels (same length as y_true).

    Raises:
        ValueError: If lists are empty or have mismatched lengths.
    """

    def __init__(self, y_true: list, y_pred: list) -> None:
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )
        if len(y_true) == 0:
            raise ValueError("Input lists must not be empty.")

        self.y_true    = y_true
        self.y_pred    = y_pred
        self.classes   = sorted(set(y_true) | set(y_pred))
        self._is_binary = set(self.classes).issubset({0, 1})


    # ──────────────────────────────────────────────────────────────────────────
    # Binary Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _binary_check(self) -> None:
        """Raises error if called on non-binary data."""
        if not self._is_binary:
            raise ValueError(
                "This method requires binary labels {0, 1}. "
                "Use multiclass methods instead."
            )

    def true_positives(self) -> int:
        """Correctly predicted positives."""
        self._binary_check()
        return sum(t == 1 and p == 1 for t, p in zip(self.y_true, self.y_pred))

    def false_positives(self) -> int:
        """Negatives incorrectly predicted as positive."""
        self._binary_check()
        return sum(t == 0 and p == 1 for t, p in zip(self.y_true, self.y_pred))

    def false_negatives(self) -> int:
        """Positives incorrectly predicted as negative."""
        self._binary_check()
        return sum(t == 1 and p == 0 for t, p in zip(self.y_true, self.y_pred))

    def true_negatives(self) -> int:
        """Correctly predicted negatives."""
        self._binary_check()
        return sum(t == 0 and p == 0 for t, p in zip(self.y_true, self.y_pred))

    def accuracy(self) -> float:
        """Fraction of all predictions that are correct."""
        return sum(t == p for t, p in zip(self.y_true, self.y_pred)) / len(self.y_true)

    def precision(self) -> float:
        """
        Of all predicted positives, how many are truly positive?
        Answers: 'When I predict FAKE, am I right?'
        """
        self._binary_check()
        tp = self.true_positives()
        fp = self.false_positives()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self) -> float:
        """
        Of all actual positives, how many did we catch?
        Answers: 'Did I find all the FAKE articles?'
        """
        self._binary_check()
        tp = self.true_positives()
        fn = self.false_negatives()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_score(self) -> float:
        """
        Harmonic mean of precision and recall.
        Answers: 'Are both precision and recall high?'
        Best value: 1.0  |  Worst value: 0.0
        """
        self._binary_check()
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def binary_report(self) -> dict:
        """
        Returns a full binary classification report as a dictionary.

        Returns:
            dict with keys: TP, FP, FN, TN, accuracy, precision, recall, f1
        """
        self._binary_check()
        return {
            "TP"       : self.true_positives(),
            "FP"       : self.false_positives(),
            "FN"       : self.false_negatives(),
            "TN"       : self.true_negatives(),
            "accuracy" : self.accuracy(),
            "precision": self.precision(),
            "recall"   : self.recall(),
            "f1"       : self.f1_score(),
        }


    # ──────────────────────────────────────────────────────────────────────────
    # Multi-class Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _class_tp(self, cls) -> int:
        return sum(t == cls and p == cls for t, p in zip(self.y_true, self.y_pred))

    def _class_fp(self, cls) -> int:
        return sum(t != cls and p == cls for t, p in zip(self.y_true, self.y_pred))

    def _class_fn(self, cls) -> int:
        return sum(t == cls and p != cls for t, p in zip(self.y_true, self.y_pred))

    def class_precision(self, cls) -> float:
        tp = self._class_tp(cls)
        fp = self._class_fp(cls)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def class_recall(self, cls) -> float:
        tp = self._class_tp(cls)
        fn = self._class_fn(cls)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def class_f1(self, cls) -> float:
        p = self.class_precision(cls)
        r = self.class_recall(cls)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def macro_precision(self) -> float:
        return sum(self.class_precision(c) for c in self.classes) / len(self.classes)

    def macro_recall(self) -> float:
        return sum(self.class_recall(c) for c in self.classes) / len(self.classes)

    def macro_f1(self) -> float:
        return sum(self.class_f1(c) for c in self.classes) / len(self.classes)

    def multiclass_report(self) -> dict:
        """
        Returns full multi-class classification report.

        Returns:
            dict with accuracy, per_class metrics, and macro averages.
        """
        report = {"accuracy": self.accuracy(), "per_class": {}}
        for cls in self.classes:
            report["per_class"][cls] = {
                "precision": self.class_precision(cls),
                "recall"   : self.class_recall(cls),
                "f1"       : self.class_f1(cls),
            }
        report["macro_avg"] = {
            "precision": self.macro_precision(),
            "recall"   : self.macro_recall(),
            "f1"       : self.macro_f1(),
        }
        return report


# ══════════════════════════════════════════════════════════════════════════════
# Pretty Printer
# ══════════════════════════════════════════════════════════════════════════════

def print_report(title: str, report: dict) -> None:
    """
    Pretty-prints a classification report dictionary.

    Args:
        title  : Display title for the report.
        report : Dict returned by binary_report() or multiclass_report().
    """
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")

    if "per_class" in report:
        print(f"  Accuracy   : {report['accuracy']:.4f}")
        print(f"  Per-class metrics:")
        for cls, m in report["per_class"].items():
            print(f"    Class {cls}: "
                  f"precision={m['precision']:.4f}  "
                  f"recall={m['recall']:.4f}  "
                  f"f1={m['f1']:.4f}")
        ma = report["macro_avg"]
        print(f"  Macro avg  : "
              f"precision={ma['precision']:.4f}  "
              f"recall={ma['recall']:.4f}  "
              f"f1={ma['f1']:.4f}")
    else:
        for k, v in report.items():
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.4f}")
            else:
                print(f"  {k:12s}: {v}")

    print(f"{'=' * 55}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point — run tests
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "#" * 55)
    print("  BINARY CLASSIFICATION TESTS")
    print("#" * 55)

    # Test 1 — Perfect
    ev = ClassificationEvaluator([1, 0, 1, 0], [1, 0, 1, 0])
    r  = ev.binary_report()
    assert r["accuracy"] == 1.0 and r["f1"] == 1.0
    print_report("TC1 — Perfect Classification", r)
    print("  ✓ Passed")

    # Test 2 — All wrong
    ev = ClassificationEvaluator([1, 1, 0, 0], [0, 0, 1, 1])
    r  = ev.binary_report()
    assert r["accuracy"] == 0.0 and r["f1"] == 0.0
    print_report("TC2 — All Wrong", r)
    print("  ✓ Passed")

    # Test 3 — Imbalanced trap
    ev = ClassificationEvaluator(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    r  = ev.binary_report()
    assert r["accuracy"] == 0.9 and r["f1"] == 0.0
    print_report("TC3 — Imbalanced Trap (90% accuracy, 0% F1)", r)
    print("  ✓ Passed")

    print("\n  All tests passed ✓")