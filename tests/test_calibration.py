import numpy as np

from complaint_intelligence.calibration import (
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    select_confidence_threshold,
    softmax,
)
from complaint_intelligence.forward_holdout import (
    apply_probability_temperature,
    probability_logits,
    selective_policy_metrics,
)


def test_softmax_rows_sum_to_one() -> None:
    probs = softmax(np.array([[2.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))


def test_temperature_fitting_reduces_calibration_nll() -> None:
    logits = np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 0.0]])
    labels = np.array([0, 1, 1, 0])
    before = negative_log_likelihood(softmax(logits), labels)
    temperature = fit_temperature(logits, labels)
    after = negative_log_likelihood(softmax(logits / temperature), labels)
    assert 0.05 <= temperature <= 10.0
    assert after < before


def test_ece_and_selective_policy() -> None:
    probs = np.array(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.40, 0.60],
            [0.45, 0.55],
        ]
    )
    labels = np.array([0, 0, 0, 1])
    assert 0.0 <= expected_calibration_error(probs, labels, n_bins=4) <= 1.0
    policy = select_confidence_threshold(probs, labels, target_accuracy=0.9)
    assert policy["target_met"] is True
    assert policy["threshold"] == 0.9
    assert policy["accepted_count"] == 2
    assert policy["coverage"] == 0.5
    assert policy["accepted_accuracy"] == 1.0


def test_selective_policy_fails_closed_when_target_is_impossible() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2]])
    labels = np.array([1, 1])
    policy = select_confidence_threshold(probs, labels, target_accuracy=1.0)
    assert policy["target_met"] is False
    assert policy["accepted_count"] == 0
    assert policy["coverage"] == 0.0


def test_probability_temperature_one_preserves_normalized_probabilities() -> None:
    probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
    np.testing.assert_allclose(
        apply_probability_temperature(probabilities, 1.0),
        probabilities,
    )
    assert probability_logits(probabilities).shape == probabilities.shape


def test_forward_policy_is_disabled_when_calibration_target_failed() -> None:
    probabilities = np.array([[0.95, 0.05], [0.1, 0.9]])
    labels = np.array([0, 1])
    result = selective_policy_metrics(
        probabilities,
        labels,
        threshold=0.8,
        policy_enabled=False,
    )
    assert result["accepted_count"] == 0
    assert result["review_count"] == 2
    assert result["coverage"] == 0.0
    assert result["accepted_accuracy"] is None


def test_forward_policy_reports_wilson_interval_for_accepted_cases() -> None:
    probabilities = np.array([[0.95, 0.05], [0.8, 0.2], [0.45, 0.55]])
    labels = np.array([0, 1, 1])
    result = selective_policy_metrics(
        probabilities,
        labels,
        threshold=0.75,
        policy_enabled=True,
    )
    assert result["accepted_count"] == 2
    assert result["review_count"] == 1
    assert result["accepted_accuracy"] == 0.5
    assert result["accepted_accuracy_wilson_95"]["low"] < 0.5
    assert result["accepted_accuracy_wilson_95"]["high"] > 0.5
