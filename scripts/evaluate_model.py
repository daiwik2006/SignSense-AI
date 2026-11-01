"""Evaluate the trained static classifier on the held-out test set.

Reads:
    data/processed/*  (test features/labels)
    models/sign_classifier.keras
    models/labels.json
    reports/training_history.csv  (training curves)

Writes to ``reports/``:
    evaluation_summary.json  (accuracy, precision, recall, F1 by split)
    classification_report.txt
    accuracy_history.png, loss_history.png, confusion_matrix.png

Usage:
    python scripts/evaluate_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sign_classifier.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(data_dir: Path, model_path: Path, labels_path: Path, report_dir: Path) -> int:
    if not model_path.exists():
        print(
            f"ERROR: model not found at {model_path}. Train first:\n"
            "  python scripts/preprocess_data.py\n"
            "  python scripts/train_static_model.py",
            file=sys.stderr,
        )
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    labels_data = load_json(labels_path)
    index_to_label = {
        int(k): v for k, v in labels_data["index_to_label"].items()
    }
    num_classes = len(index_to_label)
    labels = [index_to_label[i] for i in sorted(index_to_label)]

    print("Loading model...")
    model = keras.models.load_model(model_path)

    X_train = np.load(data_dir / "features_train.npy")
    y_train = np.load(data_dir / "labels_train.npy")
    X_val = np.load(data_dir / "features_val.npy")
    y_val = np.load(data_dir / "labels_val.npy")
    X_test = np.load(data_dir / "features_test.npy")
    y_test = np.load(data_dir / "labels_test.npy")

    def predict_labels(X: np.ndarray) -> np.ndarray:
        return np.argmax(model.predict(X, verbose=0), axis=1)

    y_pred_train = predict_labels(X_train)
    y_pred_val = predict_labels(X_val)
    y_pred_test = predict_labels(X_test)

    results = {}
    for split_name, y_true, y_pred in (
        ("train", y_train, y_pred_train),
        ("val", y_val, y_pred_val),
        ("test", y_test, y_pred_test),
    ):
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        results[split_name] = {
            "accuracy": float(acc),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "n": int(len(y_true)),
        }
        print(f"\n[{split_name.upper()}]")
        print(f"  accuracy          : {acc:.4f}")
        print(f"  precision (weighted): {precision:.4f}")
        print(f"  recall  (weighted): {recall:.4f}")
        print(f"  f1      (weighted): {f1:.4f}")

    # Per-class metrics on the test set.
    report = classification_report(
        y_test, y_pred_test, labels=list(range(num_classes)),
        target_names=labels, zero_division=0, output_dict=True,
    )
    print("\nPer-class metrics (test set):")
    for i in range(num_classes):
        r = report[labels[i]]
        print(f"  {labels[i]:<12}precision={r['precision']:.3f} "
              f"recall={r['recall']:.3f} f1={r['f1-score']:.3f} "
              f"support={r['support']}")

    with (report_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y_test, y_pred_test, labels=list(range(num_classes)),
                target_names=labels, zero_division=0,
            )
        )

    # Confusion matrix.
    cm = confusion_matrix(y_test, y_pred_test, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes), labels, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), labels)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Training history curves.
    history_path = report_dir / "training_history.csv"
    if history_path.exists():
        history = {}
        with history_path.open("r", encoding="utf-8") as f:
            cols = f.readline().strip().split(",")
            rows = [ln.strip().split(",") for ln in f if ln.strip()]
        for j, col in enumerate(cols):
            values = [float(r[j]) for r in rows]
            history[col] = values

        epochs = list(range(1, len(history.get("loss", [])) + 1))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, history["accuracy"], label="Train")
        ax.plot(epochs, history["val_accuracy"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(report_dir / "accuracy_history.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, history["loss"], label="Train")
        ax.plot(epochs, history["val_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(report_dir / "loss_history.png", dpi=150)
        plt.close(fig)

    summary = {
        "test_labels": labels,
        "test_class_counts": dict(Counter(y_test.tolist())),
        "results": results,
        "per_class_test": {
            labels[i]: report[labels[i]] for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
    }
    with (report_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote report files to {report_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args.data_dir, args.model, args.labels, args.report_dir))