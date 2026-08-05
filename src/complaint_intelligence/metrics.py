"""Auditable classification metrics, intervals, and risk-coverage summaries."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    label_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe global and per-class classification metrics."""
    targets = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predicted.shape or targets.ndim != 1 or len(targets) == 0:
        raise ValueError("labels and predictions must be non-empty vectors of equal length")
    classes = np.unique(np.concatenate([targets, predicted]))
    names = [str(value) for value in classes] if label_names is None else list(label_names)
    if len(names) != len(classes):
        raise ValueError("label_names must match the observed class count")
    precision, recall, per_f1, support = precision_recall_fscore_support(
        targets,
        predicted,
        labels=classes,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(
            f1_score(targets, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predicted, labels=classes, average="weighted", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("invalid Wilson interval inputs")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    n_resamples: int = 1_000,
    seed: int = 53353,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return a deterministic percentile bootstrap interval for macro-F1."""
    targets = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predicted.shape or targets.ndim != 1 or len(targets) == 0:
        raise ValueError("labels and predictions must be non-empty vectors of equal length")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    classes = np.unique(np.concatenate([targets, predicted]))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, len(targets), size=len(targets))
        samples[index] = f1_score(
            targets[sample],
            predicted[sample],
            labels=classes,
            average="macro",
            zero_division=0,
        )
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(
            f1_score(targets, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1 - alpha)),
        "confidence": float(confidence),
        "resamples": int(n_resamples),
    }


def risk_coverage_curve(confidences: np.ndarray, correct: np.ndarray) -> list[dict[str, float]]:
    """Return empirical risk as progressively lower-confidence cases are deferred."""
    confidence_values = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correct, dtype=bool)
    if confidence_values.shape != correctness.shape or confidence_values.ndim != 1:
        raise ValueError("confidences and correct must be equal-length vectors")
    if len(confidence_values) == 0:
        return []
    order = np.argsort(-confidence_values, kind="stable")
    sorted_confidence = confidence_values[order]
    sorted_correct = correctness[order]
    result: list[dict[str, float]] = []
    for accepted_count in range(len(sorted_correct), 0, -1):
        accepted = sorted_correct[:accepted_count]
        result.append(
            {
                "coverage": float(accepted_count / len(sorted_correct)),
                "risk": float(1.0 - accepted.mean()),
                "threshold": float(sorted_confidence[accepted_count - 1]),
            }
        )
    return result
