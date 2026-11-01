"""Keras model definition for static gesture classification."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, ReLU, Input


def build_static_classifier(input_dim: int, num_classes: int) -> Sequential:
    """Build a small feed-forward network for static gesture classification.

    The architecture is intentionally compact so training and inference run
    comfortably on a CPU-only laptop.

    Parameters
    ----------
    input_dim:
        Length of the normalized landmark feature vector (63 for v1).
    num_classes:
        Number of trained gesture labels.
    """
    if input_dim < 1 or num_classes < 2:
        raise ValueError("input_dim must be >= 1 and num_classes must be >= 2.")

    model = Sequential(name="sign_static_classifier")
    model.add(Input(shape=(input_dim,), dtype="float32"))

    model.add(Dense(128, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Dense(64, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="softmax", name="output"))

    return model


def compile_static_classifier(model: Sequential, learning_rate: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model