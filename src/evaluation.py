"""Binary metrics with explicit class ordering, including absent-class cases."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

CLASS_NAMES = ["Bearish", "Bullish"]


def classification_metrics(truth, predicted) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1], average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": float(precision), "recall_macro": float(recall),
        "f1_macro": float(f1), "n": int(len(truth)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            truth, predicted, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0),
        "class_support": {name: int(np.sum(np.asarray(truth) == i))
                          for i, name in enumerate(CLASS_NAMES)},
    }
