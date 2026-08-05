"""Offline unit tests for calibrated inference and fail-closed routing."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_DIR))

from inference import (  # noqa: E402
    ComplaintRouter,
    ConfigurationError,
    InputValidationError,
    ReviewPolicy,
    RuntimeConfig,
    normalize_complaint,
    parse_label_map,
    parse_review_policy,
    parse_temperature,
)


class FakeTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        del text
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeModel:
    def __init__(self, logits: list[float]) -> None:
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self._logits)


def make_router(*, threshold: float = 0.50, target_met: bool = True) -> ComplaintRouter:
    return ComplaintRouter(
        tokenizer=FakeTokenizer(),
        model=FakeModel([2.0, 1.0, 0.0]),
        temperature=2.0,
        review_policy=ReviewPolicy(threshold=threshold, target_met=target_met),
        labels=("Credit card", "Mortgage", "Debt collection"),
        device=torch.device("cpu"),
    )


class RuntimeConfigTests(unittest.TestCase):
    def test_model_id_is_required_without_loading_a_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "MODEL_ID is required"):
                RuntimeConfig.from_environment()

    def test_optional_revision_is_trimmed(self) -> None:
        with patch.dict(
            os.environ,
            {"MODEL_ID": " owner/model ", "MODEL_REVISION": " commit123 "},
            clear=True,
        ):
            config = RuntimeConfig.from_environment()
        self.assertEqual(config.model_id, "owner/model")
        self.assertEqual(config.revision, "commit123")


class ArtifactValidationTests(unittest.TestCase):
    def test_frozen_artifacts_parse(self) -> None:
        self.assertEqual(parse_temperature({"temperature": 1.5}), 1.5)
        self.assertEqual(
            parse_review_policy({"threshold": 0.8, "target_met": True}),
            ReviewPolicy(threshold=0.8, target_met=True),
        )
        self.assertEqual(
            parse_label_map({"0": "Credit card", "1": "Mortgage", "2": "Debt collection"}),
            ("Credit card", "Mortgage", "Debt collection"),
        )

    def test_invalid_calibration_artifacts_fail_closed(self) -> None:
        invalid_payloads = (
            lambda: parse_temperature({"temperature": float("nan")}),
            lambda: parse_review_policy({"threshold": 1.1, "target_met": True}),
            lambda: parse_review_policy({"threshold": 0.5, "target_met": "yes"}),
            lambda: parse_label_map({"1": "A", "2": "B", "3": "C"}),
        )
        for call in invalid_payloads:
            with self.subTest(call=call):
                with self.assertRaises(ConfigurationError):
                    call()


class PredictionTests(unittest.TestCase):
    def test_input_is_normalized_and_bounded(self) -> None:
        self.assertEqual(
            normalize_complaint("  A fictional complaint\nwith enough detail.  "),
            "A fictional complaint with enough detail.",
        )
        with self.assertRaises(InputValidationError):
            normalize_complaint("too short")

    def test_prediction_is_temperature_calibrated_and_returns_top_three(self) -> None:
        prediction = make_router().predict("A fictional complaint with enough detail to classify.")
        expected = torch.softmax(torch.tensor([2.0, 1.0, 0.0]) / 2.0, dim=0)[0].item()
        self.assertAlmostEqual(prediction.confidence, expected, places=6)
        self.assertEqual(prediction.predicted_product, "Credit card")
        self.assertEqual(len(prediction.top_predictions), 3)
        self.assertEqual(prediction.recommended_action, "model_assisted_route")
        self.assertFalse(hasattr(prediction, "text"))

    def test_below_threshold_routes_to_human_review(self) -> None:
        prediction = make_router(threshold=0.99).predict(
            "A fictional complaint with enough detail to classify."
        )
        self.assertEqual(prediction.recommended_action, "human_review")
        self.assertIn("below", prediction.review_reason)

    def test_unmet_policy_target_always_routes_to_human_review(self) -> None:
        prediction = make_router(threshold=0.0, target_met=False).predict(
            "A fictional complaint with enough detail to classify."
        )
        self.assertEqual(prediction.recommended_action, "human_review")
        self.assertIn("fails closed", prediction.review_reason)


if __name__ == "__main__":
    unittest.main()
