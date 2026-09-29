"""Pure helpers for the preregistered temporal forward-holdout evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from complaint_intelligence.calibration import softmax
from complaint_intelligence.metrics import wilson_interval


def probability_logits(probabilities: np.ndarray) -> np.ndarray:
    """Convert a finite probability matrix to normalized log-probability scores."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty matrix with at least two classes")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = values.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("each probability row must have positive mass")
    normalized = values / row_sums
    return np.log(np.clip(normalized, 1e-12, 1.0))


def apply_probability_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply scalar temperature scaling without changing class rankings."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    return softmax(probability_logits(probabilities) / float(temperature))


def selective_policy_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    policy_enabled: bool,
) -> dict[str, Any]:
    """Evaluate a frozen confidence policy without selecting on evaluation labels."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or targets.ndim != 1 or len(probs) != len(targets):
        raise ValueError("probabilities and labels have incompatible shapes")
    if len(targets) == 0 or not np.isfinite(probs).all():
        raise ValueError("non-empty finite probabilities are required")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be between zero and one")

    predictions = probs.argmax(axis=1)
    accepted = (
        probs.max(axis=1) >= float(threshold)
        if policy_enabled
        else np.zeros(len(targets), dtype=bool)
    )
    accepted_count = int(accepted.sum())
    result: dict[str, Any] = {
        "policy_enabled": bool(policy_enabled),
        "threshold": float(threshold),
        "accepted_count": accepted_count,
        "review_count": int(len(targets) - accepted_count),
        "coverage": float(accepted_count / len(targets)),
        "accepted_accuracy": None,
        "accepted_accuracy_wilson_95": None,
    }
    if accepted_count:
        correct = predictions[accepted] == targets[accepted]
        low, high = wilson_interval(int(correct.sum()), accepted_count)
        result["accepted_accuracy"] = float(correct.mean())
        result["accepted_accuracy_wilson_95"] = {"low": low, "high": high}
    return result
