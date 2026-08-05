"""Post-hoc temperature calibration and confidence-based review routing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable row-wise softmax."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must be a two-dimensional array")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return multiclass negative log likelihood."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or targets.ndim != 1 or len(probs) != len(targets):
        raise ValueError("probabilities and labels have incompatible shapes")
    if len(targets) == 0:
        raise ValueError("at least one observation is required")
    selected = np.clip(probs[np.arange(len(targets)), targets], 1e-12, 1.0)
    return float(-np.log(selected).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    lower: float = 0.05,
    upper: float = 10.0,
) -> float:
    """Fit one scalar temperature on a held-out calibration split."""
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.ndim != 1 or len(values) != len(targets):
        raise ValueError("logits and labels have incompatible shapes")
    if len(targets) == 0 or not 0 < lower < upper:
        raise ValueError("invalid calibration inputs")

    result = minimize_scalar(
        lambda temperature: negative_log_likelihood(softmax(values / temperature), targets),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("temperature fitting failed")
    return float(np.clip(result.x, lower, upper))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> float:
    """Compute top-label expected calibration error with equal-width bins."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or targets.ndim != 1 or len(probs) != len(targets):
        raise ValueError("probabilities and labels have incompatible shapes")
    if len(targets) == 0 or n_bins <= 0:
        raise ValueError("non-empty inputs and positive n_bins are required")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidences[mask].mean())
            )
    return float(error)


def select_confidence_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    target_accuracy: float = 0.90,
    min_coverage: float = 0.0,
) -> dict[str, Any]:
    """Maximize calibration coverage subject to an accepted-accuracy target."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or targets.ndim != 1 or len(probs) != len(targets):
        raise ValueError("probabilities and labels have incompatible shapes")
    if len(targets) == 0 or not 0 <= target_accuracy <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("invalid selective-prediction inputs")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == targets
    required_count = math.ceil(min_coverage * len(targets))
    candidates: list[dict[str, Any]] = []
    for threshold in np.unique(confidences):
        accepted = confidences >= threshold
        count = int(accepted.sum())
        if count < required_count or count == 0:
            continue
        accuracy = float(correct[accepted].mean())
        if accuracy + 1e-12 >= target_accuracy:
            candidates.append(
                {
                    "threshold": float(threshold),
                    "accepted_count": count,
                    "coverage": float(count / len(targets)),
                    "accepted_accuracy": accuracy,
                    "target_met": True,
                }
            )
    if not candidates:
        return {
            "threshold": 1.0,
            "accepted_count": 0,
            "coverage": 0.0,
            "accepted_accuracy": None,
            "target_met": False,
        }
    return max(candidates, key=lambda item: (item["accepted_count"], -item["threshold"]))
