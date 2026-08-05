import numpy as np

from complaint_intelligence.calibration import (
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    select_confidence_threshold,
    softmax,
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
