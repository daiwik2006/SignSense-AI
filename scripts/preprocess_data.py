"""Dataset preparation: validation, label encoding, stratified splitting.

Reads the raw landmark CSV produced by ``scripts/collect_data.py`` and writes:

- ``data/processed/features_train.npy``, ``labels_train.npy``
- ``data/processed/features_val.npy``, ``labels_val.npy``
- ``data/processed/features_test.npy``, ``labels_test.npy``
- ``data/processed/split_metadata.json`` (class mapping + sizes)
- ``models/labels.json`` (label encoding used at inference)

Malformed rows are reported and skipped; no data is silently mutated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from app.ml.preprocess import PreprocessConfig, Preprocessor  # noqa: E402
from app.utils.landmarks_io import FEATURE_COLS, load_landmarks, row_count  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "landmarks.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"

SEED = 42

logger = logging.getLogger(__name__)


def run(raw_path: Path, out_dir: Path, labels_path: Path) -> int:
    print(f"Loading raw data from {raw_path}")
    labels, handedness, timestamps, features = load_landmarks(raw_path)
    if not labels:
        print("ERROR: no valid samples found in the raw CSV.", file=sys.stderr)
        return 1

    print(f"Raw samples: {len(labels)}")

    # If original labels come pre-normalized they pass through normalize_dataset.
    # Apply normalization.
    from app.ml.detector import DetectedHand, Landmark

    preprocessor = Preprocessor(PreprocessConfig(max_hands=1))
    normalized: list[np.ndarray] = []
    kept_labels: list[str] = []
    malformed = 0
    for label, vec in zip(labels, features):
        if vec.size != FEATURE_COLS or not np.all(np.isfinite(vec)):
            malformed += 1
            continue
        hand = DetectedHand(
            landmarks=[
                Landmark(x=r[0], y=r[1], z=r[2])
                for r in vec.reshape(21, 3)
            ],
            handedness="Unknown",
            score=0.0,
        )
        feat = preprocessor.normalize_single_hand(hand)
        if feat is None:
            malformed += 1
            continue
        normalized.append(feat)
        kept_labels.append(label)

    if malformed:
        print(f"WARNING: skipped {malformed} malformed/degenerate samples.")

    if not kept_labels:
        print("ERROR: no usable samples survived validation.", file=sys.stderr)
        return 1

    labels_np = np.array(kept_labels)
    features_np = np.stack(normalized).astype(np.float32)
    print(f"Usable samples: {len(labels_np)}")

    # Minimum class coverage guard.
    class_counts = Counter(labels_np.tolist())
    min_per_class = min(class_counts.values())
    if len(class_counts) < 2:
        print("ERROR: need at least 2 distinct gesture labels.", file=sys.stderr)
        return 1
    if min_per_class < 3:
        print(
            f"ERROR: every label needs at least 3 samples "
            f"(smallest class has {min_per_class}).",
            file=sys.stderr,
        )
        return 1

    # Class distribution.
    print("\nClass distribution:")
    for label, count in class_counts.most_common():
        print(f"  {label:<16}{count:>6}")

    # Stratified split with reproducible seed.
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(labels_np))
    rng.shuffle(indices)
    labels_np = labels_np[indices]
    features_np = features_np[indices]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=labels_np,
        random_state=SEED,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.1875,  # 0.15 of total
        stratify=labels_np[train_idx],
        random_state=SEED,
    )

    features_train, labels_train = features_np[train_idx], labels_np[train_idx]
    features_val, labels_val = features_np[val_idx], labels_np[val_idx]
    features_test, labels_test = features_np[test_idx], labels_np[test_idx]

    unique_labels = sorted(set(labels_np.tolist()))
    mapping = {label: i for i, label in enumerate(unique_labels)}

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "features_train.npy", features_train)
    np.save(out_dir / "labels_train.npy", np.array([mapping[l] for l in labels_train]))
    np.save(out_dir / "features_val.npy", features_val)
    np.save(out_dir / "labels_val.npy", np.array([mapping[l] for l in labels_val]))
    np.save(out_dir / "features_test.npy", features_test)
    np.save(out_dir / "labels_test.npy", np.array([mapping[l] for l in labels_test]))

    metadata = {
        "seed": SEED,
        "num_classes": len(unique_labels),
        "label_to_index": mapping,
        "index_to_label": {str(i): l for i, l in enumerate(unique_labels)},
        "sizes": {
            "train": len(features_train),
            "val": len(features_val),
            "test": len(features_test),
            "total": len(features_np),
        },
        "class_distribution_train": dict(Counter(labels_train.tolist())),
        "example_feature_dim": int(features_np.shape[1]),
        "preprocessor": preprocessor.to_dict(),
    }
    with (out_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with labels_path.open("w", encoding="utf-8") as f:
        json.dump({"label_to_index": mapping, "index_to_label": {
            str(i): l for i, l in enumerate(unique_labels)}}, f, indent=2)

    print("\nSaved splits:")
    for name in ("train", "val", "test"):
        n = metadata["sizes"][name]
        frac = n / len(features_np) * 100
        print(f"  {name:<5}{n:>6}  ({frac:.1f}%)")
    print(f"\nLabel mapping -> {labels_path}")
    print(f"Split metadata -> {out_dir / 'split_metadata.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the landmark dataset.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Raw CSv path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output dir.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="labels.json path.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    sys.exit(run(args.raw, args.out, args.labels))