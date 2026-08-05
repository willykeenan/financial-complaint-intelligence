"""Frozen experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass

PRODUCTS = (
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Credit card",
    "Checking or savings account",
    "Mortgage",
    "Student loan",
    "Money transfer, virtual currency, or money service",
    "Vehicle loan or lease",
)


@dataclass(frozen=True)
class ExperimentConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    samples_per_product: int = 500
    page_size: int = 100
    seed: int = 53353
    base_model: str = "distilbert/distilbert-base-uncased"
    base_model_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    max_length: int = 192
    epochs: int = 2
    learning_rate: float = 2e-5
    train_fraction: float = 0.70
    calibration_fraction: float = 0.15
    target_accepted_accuracy: float = 0.90
