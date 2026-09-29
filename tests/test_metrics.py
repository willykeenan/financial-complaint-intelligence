import numpy as np

from complaint_intelligence.metrics import (
    bootstrap_macro_f1,
    classification_metrics,
    paired_bootstrap_macro_f1_difference,
    risk_coverage_curve,
    wilson_interval,
)


def test_classification_metrics_have_expected_values() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    result = classification_metrics(labels, predictions, label_names=["a", "b"])
    assert result["accuracy"] == 0.75
    assert round(result["macro_f1"], 6) == round((2 / 3 + 0.8) / 2, 6)
    assert set(result["per_class"]) == {"a", "b"}


def test_wilson_interval_contains_empirical_rate() -> None:
    low, high = wilson_interval(successes=80, total=100)
    assert low < 0.8 < high


def test_bootstrap_interval_is_deterministic_and_bounded() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    predictions = np.array([0, 0, 1, 1, 1, 1, 2, 0, 2])
    first = bootstrap_macro_f1(labels, predictions, n_resamples=100, seed=53353)
    second = bootstrap_macro_f1(labels, predictions, n_resamples=100, seed=53353)
    assert first == second
    assert 0.0 <= first["low"] <= first["estimate"] <= first["high"] <= 1.0


def test_risk_coverage_curve_is_sorted_by_declining_coverage() -> None:
    confidences = np.array([0.9, 0.8, 0.7])
    correct = np.array([True, False, True])
    curve = risk_coverage_curve(confidences, correct)
    assert curve[0]["coverage"] == 1.0
    assert curve[-1]["coverage"] == 1 / 3
    assert all(curve[i]["coverage"] >= curve[i + 1]["coverage"] for i in range(2))


def test_paired_bootstrap_difference_is_deterministic_and_directional() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    better = labels.copy()
    worse = np.array([0, 1, 1, 1, 0, 0, 2, 0, 1])
    first = paired_bootstrap_macro_f1_difference(
        labels,
        better,
        worse,
        n_resamples=200,
        seed=53353,
    )
    second = paired_bootstrap_macro_f1_difference(
        labels,
        better,
        worse,
        n_resamples=200,
        seed=53353,
    )
    assert first == second
    assert first["estimate"] > 0
    assert first["fraction_above_zero"] > 0.9
