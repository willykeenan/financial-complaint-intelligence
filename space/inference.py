"""Calibrated complaint classification for the Hugging Face Space.

The module deliberately has no UI dependency. Model files are fetched only when
``ComplaintRouter.from_repository`` is called, which keeps imports and unit tests
offline.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MIN_INPUT_CHARS = 20
MAX_INPUT_CHARS = 6_000
MAX_MODEL_TOKENS = 192
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 10.0
MAX_SUPPORT_FILE_BYTES = 64 * 1024


class ConfigurationError(RuntimeError):
    """Raised when the Space or model repository contract is incomplete."""


class InputValidationError(ValueError):
    """Raised when submitted text is not suitable for inference."""


class PredictionError(RuntimeError):
    """Raised when inference cannot produce a safe, valid result."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Model repository settings read from the Space environment."""

    model_id: str
    revision: str | None = None

    @classmethod
    def from_environment(cls) -> RuntimeConfig:
        model_id = os.environ.get("MODEL_ID", "").strip()
        if not model_id:
            raise ConfigurationError(
                "MODEL_ID is required. Set it to the Hugging Face model repository ID."
            )
        revision = os.environ.get("MODEL_REVISION", "").strip() or None
        return cls(model_id=model_id, revision=revision)


@dataclass(frozen=True)
class ReviewPolicy:
    """Frozen selective-prediction policy loaded from the model repository."""

    threshold: float
    target_met: bool


@dataclass(frozen=True)
class ClassScore:
    """One calibrated class score."""

    label: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    """Privacy-safe inference result; submitted text is never included."""

    predicted_product: str
    confidence: float
    recommended_action: str
    review_reason: str
    review_threshold: float
    top_predictions: tuple[ClassScore, ...]


def normalize_complaint(text: Any) -> str:
    """Normalize and validate text without retaining or logging it."""
    if not isinstance(text, str):
        raise InputValidationError("Enter complaint context as text.")
    narrative = " ".join(text.split())
    if len(narrative) < MIN_INPUT_CHARS:
        raise InputValidationError(
            f"Enter at least {MIN_INPUT_CHARS} characters of complaint context."
        )
    if len(narrative) > MAX_INPUT_CHARS:
        raise InputValidationError(
            f"Keep complaint context to {MAX_INPUT_CHARS:,} characters or fewer."
        )
    return narrative


def parse_temperature(payload: Any) -> float:
    """Validate the scalar temperature produced by the frozen experiment."""
    if not isinstance(payload, Mapping):
        raise ConfigurationError("temperature.json must contain a JSON object.")
    value = payload.get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError("temperature.json must contain a numeric temperature.")
    temperature = float(value)
    if not math.isfinite(temperature) or not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ConfigurationError(
            f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}."
        )
    return temperature


def parse_review_policy(payload: Any) -> ReviewPolicy:
    """Validate the confidence threshold and fail-closed target flag."""
    if not isinstance(payload, Mapping):
        raise ConfigurationError("review_policy.json must contain a JSON object.")
    threshold_value = payload.get("threshold")
    target_met = payload.get("target_met")
    if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
        raise ConfigurationError("review_policy.json must contain a numeric threshold.")
    threshold = float(threshold_value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ConfigurationError("review threshold must be between 0 and 1.")
    if not isinstance(target_met, bool):
        raise ConfigurationError("review_policy.json must contain a boolean target_met flag.")
    return ReviewPolicy(threshold=threshold, target_met=target_met)


def parse_label_map(payload: Any) -> tuple[str, ...]:
    """Validate a contiguous, unique index-to-label mapping."""
    if not isinstance(payload, Mapping) or not payload:
        raise ConfigurationError("label_map.json must contain a non-empty JSON object.")
    try:
        indexed = {int(key): value for key, value in payload.items()}
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("label_map.json keys must be integer class IDs.") from exc
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise ConfigurationError("label_map.json class IDs must start at 0 and be contiguous.")
    labels = tuple(indexed[index] for index in expected)
    if len(labels) < 3:
        raise ConfigurationError("At least three classes are required for top-three scores.")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ConfigurationError("Every class label must be a non-empty string.")
    normalized = tuple(label.strip() for label in labels)
    if len(set(normalized)) != len(normalized):
        raise ConfigurationError("Class labels must be unique.")
    return normalized


def _read_support_json(config: RuntimeConfig, filename: str) -> Any:
    download_args: dict[str, Any] = {"repo_id": config.model_id, "filename": filename}
    if config.revision is not None:
        download_args["revision"] = config.revision
    path = Path(hf_hub_download(**download_args))
    try:
        if path.stat().st_size > MAX_SUPPORT_FILE_BYTES:
            raise ConfigurationError(f"{filename} is unexpectedly large.")
        return json.loads(path.read_text(encoding="utf-8"))
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"{filename} is not valid UTF-8 JSON.") from exc


def _configured_labels(model: Any) -> tuple[str, ...] | None:
    """Return model-config labels when they are complete, otherwise None."""
    raw = getattr(getattr(model, "config", None), "id2label", None)
    if not isinstance(raw, Mapping) or not raw:
        return None
    try:
        indexed = {int(key): value for key, value in raw.items()}
    except (TypeError, ValueError):
        return None
    if sorted(indexed) != list(range(len(indexed))):
        return None
    if any(not isinstance(indexed[index], str) for index in range(len(indexed))):
        return None
    return tuple(indexed[index].strip() for index in range(len(indexed)))


class ComplaintRouter:
    """Sequence-classification model with temperature calibration and review routing."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        temperature: float,
        review_policy: ReviewPolicy,
        labels: tuple[str, ...],
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.review_policy = review_policy
        self.labels = labels
        self.device = device

    @classmethod
    def from_repository(cls, config: RuntimeConfig) -> ComplaintRouter:
        """Load model, tokenizer, and frozen support artifacts from one repository."""
        temperature = parse_temperature(_read_support_json(config, "temperature.json"))
        review_policy = parse_review_policy(_read_support_json(config, "review_policy.json"))
        labels = parse_label_map(_read_support_json(config, "label_map.json"))

        load_args: dict[str, Any] = {
            "pretrained_model_name_or_path": config.model_id,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        if config.revision is not None:
            load_args["revision"] = config.revision
        tokenizer = AutoTokenizer.from_pretrained(**load_args)
        model = AutoModelForSequenceClassification.from_pretrained(**load_args)

        num_labels = getattr(getattr(model, "config", None), "num_labels", None)
        if num_labels != len(labels):
            raise ConfigurationError("The model output count does not match label_map.json.")
        model_labels = _configured_labels(model)
        if model_labels is not None and model_labels != labels:
            raise ConfigurationError("The model config labels do not match label_map.json.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            temperature=temperature,
            review_policy=review_policy,
            labels=labels,
            device=device,
        )

    def predict(self, text: Any) -> Prediction:
        narrative = normalize_complaint(text)
        try:
            encoded = self.tokenizer(
                narrative,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_MODEL_TOKENS,
            )
            model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**model_inputs).logits
        except Exception as exc:
            raise PredictionError("The model could not process this request.") from exc

        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
            raise PredictionError("The model returned an unexpected output shape.")
        if logits.shape[1] != len(self.labels):
            raise PredictionError("The model output no longer matches the configured labels.")
        if not bool(torch.isfinite(logits).all()):
            raise PredictionError("The model returned non-finite scores.")

        probabilities = torch.softmax(logits.float() / self.temperature, dim=-1)[0]
        if not bool(torch.isfinite(probabilities).all()):
            raise PredictionError("Calibrated class scores are not finite.")
        top_probabilities, top_indices = torch.topk(probabilities, k=3)
        top_predictions = tuple(
            ClassScore(label=self.labels[int(index)], probability=float(probability))
            for probability, index in zip(top_probabilities.cpu(), top_indices.cpu(), strict=True)
        )
        confidence = top_predictions[0].probability

        if self.review_policy.target_met and confidence >= self.review_policy.threshold:
            action = "model_assisted_route"
            reason = "The calibrated confidence meets the frozen review threshold."
        elif not self.review_policy.target_met:
            action = "human_review"
            reason = (
                "The frozen calibration policy did not meet its target, so this item fails closed."
            )
        else:
            action = "human_review"
            reason = "The calibrated confidence is below the frozen review threshold."

        return Prediction(
            predicted_product=top_predictions[0].label,
            confidence=confidence,
            recommended_action=action,
            review_reason=reason,
            review_threshold=self.review_policy.threshold,
            top_predictions=top_predictions,
        )
