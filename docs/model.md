# Model and Preprocessing Documentation

## Overview

This document describes the static-gesture classification pipeline: how raw
MediaPipe landmarks are transformed into feature vectors, how the classifier
model is structured, and how predictions are smoothed into stable tokens.

---

## Feature Extraction Algorithm

For each hand detected by MediaPipe Hands, 21 landmarks are available with
`x`, `y`, and `z` coordinates normalized to [0, 1] relative to the camera
frame (MediaPipe's default).

### Single-hand feature vector (63 dimensions)

```
Step 1   Take the 21 landmarks as a (21, 3) array:
         L[i] = [x_i, y_i, z_i]   for i in 0..20

Step 2   Subtract the wrist landmark (index 0) from every row:
         R[i] = L[i] - L[0]

Step 3   Compute hand-scale S:
         S = ‖L[9] - L[0]‖   (Euclidean distance, wrist to middle-finger MCP)

Step 4   Guard against S ≈ 0 (degenerate hand / insufficient movement).
         Return None / skip that frame.

Step 5   Normalize:
         N[i] = R[i] / S

Step 6   Flatten row-major:
         features = N.reshape(63)
```

### Why this works

| Invariance achieved | How |
|----------------------|-----|
| **Translation** (hand position in frame) | Subtracting the wrist origin. |
| **Scale** (hand distance from camera) | Dividing by the wrist-to-MCP distance. |
| **Resolution** | All coordinates are MediaPipe's own 0–1 normalization, so the model never sees pixel coordinates. |

The normalized vector describes only the *shape* of the hand.

### Why keep z?

MediaPipe's `z` is the estimated landmark depth relative to the wrist. It
helps distinguish fingers that are curled (small z difference) from fingers
that are extended straight towards the camera. Tests verify that `use_z=True`
preserves z-derived features; `use_z=False` zeros the z channel.

---

## Handedness Representation (v1 Decision)

In v1, the model is trained on a **single hand** (the primary hand as
reported by MediaPipe). No mirroring or handedness-specific coordinate
transformation is applied. The assumption is:

- Each user performs signs with the same hand they used during data collection.
- Handedness is learned implicitly by the classifier.

This avoids accidental label leakage (the same physical gesture becomes two
different vectors depending on hand). It is deterministic and the same
function is called in both training and inference.

---

## Two-Hand Support (Architecture Note, Phase 15)

`PreprocessConfig.max_hands` controls feature-vector length:

- `max_hands = 1`: 63-dim vector (v1 model).
- `max_hands = 2`: 2 × (63 + 1) = 128-dim vector: each block contains the
  63-dim normalized features followed by a 1-bit presence flag. Hand ordering
  is always *Left, then Right* (regardless of detection order), ensuring
  deterministic input.

Missing hands are zero-filled with the presence flag set to 0. The v1 model
does **not** consume this 128-dim format.

---

## Model Architecture (v1 Static Classifier)

```
Input layer             shape=(63,)
Dense 128, BatchNorm, ReLU, Dropout(0.3)
Dense 64,  BatchNorm, ReLU, Dropout(0.3)
Dense [num_classes], Softmax
```

- Loss: sparse categorical cross-entropy.
- Optimizer: Adam (lr=1e-3, reduced to 1e-4 by ReduceLROnPlateau).
- Early stopping on validation loss (patience=10 epochs).
- Best model checkpoint saved by validation accuracy.

---

## Training Procedure

1. `scripts/preprocess_data.py` loads `data/raw/landmarks.csv`, validates
   rows, normalizes labels, stratified splits (70% train / 15% val / 15% test),
   writes `data/processed/` arrays + `models/labels.json`.
2. `scripts/train_static_model.py` trains with callbacks and saves
   `models/sign_classifier.keras`.
3. `scripts/evaluate_model.py` runs on the held-out test set and writes
   metrics and plots to `reports/`.

---

## Temporal Smoothing

See `app/ml/smoothing.py`. A rolling buffer of predictions is maintained. A
prediction is accepted as a stable token only when it dominates the buffer
above a configurable stability threshold. A cooldown prevents the same token
from being repeated immediately.

---

## Prediction Inference Pipeline

```
Webcam frame (BGR, 640×480)
  → convert to RGB
  → MediaPipe Hands (21 landmarks per hand)
  → Preprocessor.normalize_hands()
  → Neural network forward pass
  → argmax → class label
  → confidence check (skip if < threshold)
  → TemporalSmoothing.process()
  → accepted token or None
```

---

## Limitations

- Only static (pose-only) gestures are supported. Dynamic gestures (movement
  over time) are not yet part of the v1 model.
- The model recognizes only the gestures present in its trained vocabulary.
- Facial expression and body posture are not captured.
