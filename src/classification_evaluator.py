"""
Classification Evaluator
Supports binary and multi-class classification metrics.
No external libraries used.
"""


class ClassificationEvaluator:
    """
    Evaluates classification performance for binary and multi-class tasks.

    Binary metrics: TP, FP, FN, TN, accuracy, precision, recall, F1.
    Multi-class metrics: per-class precision/recall/F1 + macro averages.
    """

    def __init__(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if len(y_true) == 0:
            raise ValueError("Input lists must not be empty.")
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = sorted(set(y_true) | set(y_pred))
        self._is_binary = set(self.classes).issubset({0, 1})

    # ------------------------------------------------------------------
    # Binary helpers
    # ------------------------------------------------------------------

    def _binary_check(self):
        if not self._is_binary:
            raise ValueError(
                "This method is only available for binary classification. "
                "Use multi-class methods instead."
            )

    def true_positives(self):
        self._binary_check()
        return sum(t == 1 and p == 1 for t, p in zip(self.y_true, self.y_pred))

    def false_positives(self):
        self._binary_check()
        return sum(t == 0 and p == 1 for t, p in zip(self.y_true, self.y_pred))

    def false_negatives(self):
        self._binary_check()
        return sum(t == 1 and p == 0 for t, p in zip(self.y_true, self.y_pred))

    def true_negatives(self):
        self._binary_check()
        return sum(t == 0 and p == 0 for t, p in zip(self.y_true, self.y_pred))

    def accuracy(self):
        correct = sum(t == p for t, p in zip(self.y_true, self.y_pred))
        return correct / len(self.y_true)

    def precision(self):
        self._binary_check()
        tp = self.true_positives()
        fp = self.false_positives()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self):
        self._binary_check()
        tp = self.true_positives()
        fn = self.false_negatives()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_score(self):
        self._binary_check()
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def binary_report(self):
        self._binary_check()
        tp = self.true_positives()
        fp = self.false_positives()
        fn = self.false_negatives()
        tn = self.true_negatives()
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1_score(),
        }

    # ------------------------------------------------------------------
    # Multi-class helpers
    # ------------------------------------------------------------------

    def _class_tp(self, cls):
        return sum(t == cls and p == cls for t, p in zip(self.y_true, self.y_pred))

    def _class_fp(self, cls):
        return sum(t != cls and p == cls for t, p in zip(self.y_true, self.y_pred))

    def _class_fn(self, cls):
        return sum(t == cls and p != cls for t, p in zip(self.y_true, self.y_pred))

    def class_precision(self, cls):
        tp = self._class_tp(cls)
        fp = self._class_fp(cls)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def class_recall(self, cls):
        tp = self._class_tp(cls)
        fn = self._class_fn(cls)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def class_f1(self, cls):
        p = self.class_precision(cls)
        r = self.class_recall(cls)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def macro_precision(self):
        return sum(self.class_precision(c) for c in self.classes) / len(self.classes)

    def macro_recall(self):
        return sum(self.class_recall(c) for c in self.classes) / len(self.classes)

    def macro_f1(self):
        return sum(self.class_f1(c) for c in self.classes) / len(self.classes)

    def multiclass_report(self):
        report = {"accuracy": self.accuracy(), "per_class": {}}
        for cls in self.classes:
            report["per_class"][cls] = {
                "precision": self.class_precision(cls),
                "recall": self.class_recall(cls),
                "f1": self.class_f1(cls),
            }
        report["macro_avg"] = {
            "precision": self.macro_precision(),
            "recall": self.macro_recall(),
            "f1": self.macro_f1(),
        }
        return report


# ======================================================================
# Helper: pretty-print a dict report
# ======================================================================

def _print_report(title, report):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    if "per_class" in report:
        print(f"  Accuracy : {report['accuracy']:.4f}")
        print(f"  Per-class metrics:")
        for cls, metrics in report["per_class"].items():
            print(f"    Class {cls}: precision={metrics['precision']:.4f}  "
                  f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}")
        ma = report["macro_avg"]
        print(f"  Macro avg: precision={ma['precision']:.4f}  "
              f"recall={ma['recall']:.4f}  f1={ma['f1']:.4f}")
    else:
        for k, v in report.items():
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.4f}")
            else:
                print(f"  {k:12s}: {v}")


# ======================================================================
# Binary test cases (from assignment)
# ======================================================================

def test_binary():
    print("\n" + "#"*55)
    print("  BINARY CLASSIFICATION TESTS")
    print("#"*55)

    # ------------------------------------------------------------------
    # Test Case 1 — Perfect Classification
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator([1, 0, 1, 0], [1, 0, 1, 0])
    r = ev.binary_report()
    assert r["TP"] == 2,        f"TC1 TP: expected 2, got {r['TP']}"
    assert r["FP"] == 0,        f"TC1 FP: expected 0, got {r['FP']}"
    assert r["FN"] == 0,        f"TC1 FN: expected 0, got {r['FN']}"
    assert r["TN"] == 2,        f"TC1 TN: expected 2, got {r['TN']}"
    assert r["accuracy"]  == 1.0, f"TC1 accuracy:  expected 1.0, got {r['accuracy']}"
    assert r["precision"] == 1.0, f"TC1 precision: expected 1.0, got {r['precision']}"
    assert r["recall"]    == 1.0, f"TC1 recall:    expected 1.0, got {r['recall']}"
    assert r["f1"]        == 1.0, f"TC1 f1:        expected 1.0, got {r['f1']}"
    _print_report("TC1 — Perfect Classification", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # Test Case 2 — All Wrong
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator([1, 1, 0, 0], [0, 0, 1, 1])
    r = ev.binary_report()
    assert r["TP"] == 0,        f"TC2 TP: expected 0, got {r['TP']}"
    assert r["FP"] == 2,        f"TC2 FP: expected 2, got {r['FP']}"
    assert r["FN"] == 2,        f"TC2 FN: expected 2, got {r['FN']}"
    assert r["TN"] == 0,        f"TC2 TN: expected 0, got {r['TN']}"
    assert r["accuracy"]  == 0.0, f"TC2 accuracy:  expected 0.0, got {r['accuracy']}"
    assert r["precision"] == 0.0, f"TC2 precision: expected 0.0, got {r['precision']}"
    assert r["recall"]    == 0.0, f"TC2 recall:    expected 0.0, got {r['recall']}"
    assert r["f1"]        == 0.0, f"TC2 f1:        expected 0.0, got {r['f1']}"
    _print_report("TC2 — All Wrong", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # Test Case 3 — No Predicted Positives
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator([1, 0, 1, 0], [0, 0, 0, 0])
    r = ev.binary_report()
    assert r["TP"] == 0, f"TC3 TP: expected 0, got {r['TP']}"
    assert r["FP"] == 0, f"TC3 FP: expected 0, got {r['FP']}"
    assert r["FN"] == 2, f"TC3 FN: expected 2, got {r['FN']}"
    assert r["TN"] == 2, f"TC3 TN: expected 2, got {r['TN']}"
    assert r["precision"] == 0.0, f"TC3 precision: expected 0.0, got {r['precision']}"
    assert r["recall"]    == 0.0, f"TC3 recall:    expected 0.0, got {r['recall']}"
    assert r["f1"]        == 0.0, f"TC3 f1:        expected 0.0, got {r['f1']}"
    _print_report("TC3 — No Predicted Positives", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # Test Case 4 — No Actual Positives
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator([0, 0, 0, 0], [1, 0, 1, 0])
    r = ev.binary_report()
    assert r["TP"] == 0, f"TC4 TP: expected 0, got {r['TP']}"
    assert r["FP"] == 2, f"TC4 FP: expected 2, got {r['FP']}"
    assert r["FN"] == 0, f"TC4 FN: expected 0, got {r['FN']}"
    assert r["TN"] == 2, f"TC4 TN: expected 2, got {r['TN']}"
    assert r["recall"]    == 0.0, f"TC4 recall:    expected 0.0, got {r['recall']}"
    assert r["precision"] == 0.0, f"TC4 precision: expected 0.0, got {r['precision']}"
    assert r["f1"]        == 0.0, f"TC4 f1:        expected 0.0, got {r['f1']}"
    _print_report("TC4 — No Actual Positives", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # Test Case 5 — High Recall, Low Precision
    # The classifier catches almost every positive but also fires on many
    # negatives → many FP, few FN.
    # y_true = [1,1,1,1,0,0,0,0,0,0]
    # y_pred = [1,1,1,1,1,1,1,0,0,0]  <- predicts positive aggressively
    # TP=4, FP=3, FN=0, TN=3
    # precision = 4/(4+3) ≈ 0.57   recall = 4/(4+0) = 1.0
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator(
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    )
    r = ev.binary_report()
    assert r["recall"] == 1.0,              f"TC5 recall: expected 1.0, got {r['recall']}"
    assert r["precision"] < 0.65,           f"TC5 precision should be low, got {r['precision']}"
    _print_report("TC5 — High Recall, Low Precision", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # Test Case 6 — High Precision, Low Recall
    # The classifier only predicts positive when it is very sure → few FP,
    # but misses many positives → many FN.
    # y_true = [1,1,1,1,1,1,0,0,0,0]
    # y_pred = [1,0,0,0,0,0,0,0,0,0]  <- very conservative
    # TP=1, FP=0, FN=5, TN=4
    # precision = 1/1 = 1.0   recall = 1/6 ≈ 0.17
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    r = ev.binary_report()
    assert r["precision"] == 1.0,           f"TC6 precision: expected 1.0, got {r['precision']}"
    assert r["recall"] < 0.25,              f"TC6 recall should be low, got {r['recall']}"
    _print_report("TC6 — High Precision, Low Recall", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # Test Case 7 — Rare Positive Class, Classifier Always Predicts Negative
    # Classic imbalanced dataset trap: accuracy looks good but the model
    # detects nothing useful.
    # y_true = [1,0,0,0,0,0,0,0,0,0]  (1 positive out of 10)
    # y_pred = [0,0,0,0,0,0,0,0,0,0]  (always negative)
    # TP=0, FP=0, FN=1, TN=9
    # accuracy=0.9  precision=0.0  recall=0.0  f1=0.0
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    r = ev.binary_report()
    assert r["accuracy"]  == 0.9, f"TC7 accuracy:  expected 0.9, got {r['accuracy']}"
    assert r["precision"] == 0.0, f"TC7 precision: expected 0.0, got {r['precision']}"
    assert r["recall"]    == 0.0, f"TC7 recall:    expected 0.0, got {r['recall']}"
    assert r["f1"]        == 0.0, f"TC7 f1:        expected 0.0, got {r['f1']}"
    _print_report("TC7 — Rare Class / Always-Negative Classifier", r)
    print("  High accuracy despite detecting nothing — the imbalance trap!")
    print("  ✓ All assertions passed")


# ======================================================================
# Multi-class test cases
# ======================================================================

def test_multiclass():
    print("\n" + "#"*55)
    print("  MULTI-CLASS CLASSIFICATION TESTS")
    print("#"*55)

    # ------------------------------------------------------------------
    # MC Test 1 — Perfect 3-class classification
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator(
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2],
    )
    r = ev.multiclass_report()
    assert r["accuracy"] == 1.0,                 f"MC1 accuracy: expected 1.0, got {r['accuracy']}"
    assert r["macro_avg"]["precision"] == 1.0,   f"MC1 macro precision: expected 1.0"
    assert r["macro_avg"]["recall"]    == 1.0,   f"MC1 macro recall: expected 1.0"
    assert r["macro_avg"]["f1"]        == 1.0,   f"MC1 macro f1: expected 1.0"
    _print_report("MC1 — Perfect 3-class Classification", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # MC Test 2 — All wrong (3 classes, cyclic shift)
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator(
        [0, 0, 1, 1, 2, 2],
        [1, 1, 2, 2, 0, 0],
    )
    r = ev.multiclass_report()
    assert r["accuracy"] == 0.0,                 f"MC2 accuracy: expected 0.0, got {r['accuracy']}"
    assert r["macro_avg"]["precision"] == 0.0,   f"MC2 macro precision: expected 0.0"
    assert r["macro_avg"]["recall"]    == 0.0,   f"MC2 macro recall: expected 0.0"
    _print_report("MC2 — All Wrong (3-class cyclic shift)", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # MC Test 3 — Mixed performance across 3 classes
    # Class 0: perfect. Class 1: some errors. Class 2: never predicted.
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator(
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 0, 1, 1],
    )
    r = ev.multiclass_report()
    assert r["per_class"][2]["precision"] == 0.0, "MC3 class-2 precision: expected 0.0"
    assert r["per_class"][2]["recall"]    == 0.0, "MC3 class-2 recall: expected 0.0"
    assert r["per_class"][0]["recall"]    == 1.0, "MC3 class-0 recall: expected 1.0 (all 0s found)"
    _print_report("MC3 — Mixed 3-class Performance", r)
    print("  ✓ All assertions passed")

    # ------------------------------------------------------------------
    # MC Test 4 — 4-class classification with partial overlap
    # ------------------------------------------------------------------
    ev = ClassificationEvaluator(
        ["cat", "dog", "bird", "fish", "cat", "dog", "bird", "fish"],
        ["cat", "cat", "bird", "fish", "dog", "dog", "cat",  "fish"],
    )
    r = ev.multiclass_report()
    assert r["accuracy"] == 0.625,  f"MC4 accuracy: expected 0.625, got {r['accuracy']}"
    _print_report("MC4 — 4-class (cat/dog/bird/fish)", r)
    print("  ✓ All assertions passed")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    test_binary()
    test_multiclass()
    print("\n" + "="*55)
    print("  All tests passed ✓")
    print("="*55 + "\n")
