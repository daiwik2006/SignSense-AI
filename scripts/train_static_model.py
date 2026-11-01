"""Train the static gesture classification model.

Reads the processed splits produced by ``scripts/preprocess_data.py``,
trains the classifier with early stopping + checkpointing, and saves:

- ``models/sign_classifier.keras`` (best validation-accuracy checkpoint)
- ``models/labels.json`` (label mapping, rewritten from the split metadata)
- ``reports/training_summary.json`` (loss/accuracy curves)

Usage:
    python scripts/train_static_model.py [--epochs 60] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras.callbacks import (  # noqa: E402
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from app.ml.model import build_static_classifier, compile_static_classifier  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sign_classifier.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_splits(data_dir: Path):
    def _load(name: str):
        return np.load(data_dir / f"{name}.npy")

    features_train = _load("features_train")
    labels_train = _load("labels_train")
    features_val = _load("features_val")
    labels_val = _load("labels_val")
    feature_dim = int(features_train.shape[1])
    return (
        features_train,
        labels_train,
        features_val,
        labels_val,
        feature_dim,
    )


def run(data_dir: Path, model_path: Path, labels_path: Path, report_dir: Path,
        epochs: int, batch_size: int) -> int:
    set_seeds()

    print(f"Loading processed splits from {data_dir}")
    X_train, y_train, X_val, y_val, feature_dim = load_splits(data_dir)

    num_classes = int(np.max(y_train)) + 1
    print(f"Feature dim: {feature_dim}")
    print(f"Classes: {num_classes}")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    model = build_static_classifier(feature_dim, num_classes)
    model = compile_static_classifier(model)
    model.summary()

    report_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        CSVLogger(str(report_dir / "training_history.csv"), append=False),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    summary = {
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": None,
        "best_val_loss": None,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    if history.history.get("val_accuracy"):
        summary["best_val_accuracy"] = float(max(history.history["val_accuracy"]))
        summary["best_val_loss"] = float(min(history.history["val_loss"]))

    # Re-write the label mapping (source of truth is split metadata).
    with (data_dir / "split_metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"label_to_index": meta["label_to_index"],
             "index_to_label": meta["index_to_label"]},
            f,
            indent=2,
        )

    with (report_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved labels  -> {labels_path}")
    print(f"Report        -> {report_dir / 'training_summary.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        run(
            args.data_dir, args.model, args.labels, args.report_dir,
            args.epochs, args.batch_size,
        )
    )