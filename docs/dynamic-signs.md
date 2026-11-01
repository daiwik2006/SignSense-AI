# Dynamic (Sequence) Sign Recognition — Architecture Note

> **Status: NOT yet implemented as a model.** This document defines the
> architecture and data format for the optional second pipeline. The supported,
> working pipeline in this repository is the **static** gesture classifier
> (`scripts/train_static_model.py`).

## Motivation

Many real signs involve movement (wave, swipe, round, etc.). A static model
that classifies a single frame cannot represent these. The dynamic pipeline
classifies a *short sequence of frames* instead.

## Data Format

Each dynamic sample is a fixed-length sequence of normalized landmark
vectors. Recommended format: `.npz` files in `data/dynamic/`:

```
data/dynamic/
    label1/
        sample_001.npz   # {"seq": (T, 63) float32, "label": "label1"}
        sample_002.npz
    label2/
        ...
```

- `T` = number of frames in the sequence (e.g. 20). Keep it fixed for all
  samples so the model input is deterministic.
- Each 63-dim vector is produced by the exact same `Preprocessor`
  (`app/ml/preprocess.py`) used by the static pipeline.

## Proposed Model (LSTM/GRU)

```
Input            (T, 63)
MaskingLSTM(128) / GRU(128) or bidirectional
Dropout(0.3)
GRU/LSTM(64)
Dense(num_classes, softmax)
```

- Loss: sparse categorical cross-entropy.
- Sequences are variable-length-safe via a mask or padded + masked.
- The model is kept **fully separate** from the static classifier
  (`models/sign_dynamic_classifier.keras`) and uses its own label mapping.

## Sequence Normalization

Each frame is preprocessed identically to the static pipeline (wrist-relative,
scale-normalized). No additional temporal normalization is applied in v1;
longer sequences can later be resized by interpolation after data collection.

## Data Collection (experimental)

`python scripts/collect_dynamic_data.py --label WAVE --frames 20 --samples 30`

The collector shows a countdown, records the rolling window of the most
recent `--frames` landmark frames while a hand is visible, writes each
accepted window to `data/dynamic/<label>/`, and prints how many samples were
kept. It is a prototype; sequences shorter than `--frames` (hand dropping out
mid-window) are rejected.

## Training (not wired, documented design)

A future `scripts/train_dynamic_model.py` would:

1. Load `data/dynamic/*.npz`.
2. Apply the exact same feature normalization as collection/inference.
3. Pad/mask, label-encode, stratified split (same 70/15/15 policy).
4. Train the LSTM/GRU with early stopping + checkpointing.

## Inference (design)

```
webcam frames → rolling buffer of (T, 63) normalized vectors
→ LSTM/GRU forward pass → argmax → confidence filter → smoothing
```

The existing `TemporalSmoothing` class is reused unchanged.

## Why it is separated

- The static classifier is small, fast, and fully working.
- Sequence models add latency and require careful window bookkeeping.
- Keeping the two pipelines independent means a regression in one never
  affects the other.