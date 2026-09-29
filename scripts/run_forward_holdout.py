#!/usr/bin/env python3
"""Evaluate the frozen Q1 models on the preregistered 2024 Q2 holdout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from complaint_intelligence.calibration import (
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    select_confidence_threshold,
    softmax,
)
from complaint_intelligence.config import PRODUCTS, ExperimentConfig
from complaint_intelligence.forward_holdout import (
    apply_probability_temperature,
    probability_logits,
    selective_policy_metrics,
)
from complaint_intelligence.metrics import (
    bootstrap_macro_f1,
    classification_metrics,
    paired_bootstrap_macro_f1_difference,
    risk_coverage_curve,
)

PROTOCOL_ID = "fci.forward-holdout.2024q2.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _aligned_baseline_probabilities(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    classes = [str(value) for value in model.classes_]
    if set(classes) != set(PRODUCTS):
        raise ValueError("baseline classes do not match the frozen product set")
    raw = np.asarray(model.predict_proba([row["text"] for row in rows]), dtype=np.float64)
    indices = [classes.index(product) for product in PRODUCTS]
    aligned = raw[:, indices]
    return aligned / aligned.sum(axis=1, keepdims=True)


def _encode(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> TensorDataset:
    encoded = tokenizer(
        [row["text"] for row in rows],
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"])


def _transformer_logits(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    labels = [str(model.config.id2label[index]) for index in range(len(PRODUCTS))]
    if labels != list(PRODUCTS):
        raise ValueError("transformer label order does not match the frozen product order")
    model.to(device)
    model.eval()
    dataset = _encode(tokenizer, rows, ExperimentConfig())
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for input_ids, attention_mask in DataLoader(dataset, batch_size=batch_size):
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            output.append(logits.detach().cpu().numpy())
    return np.concatenate(output)


def _model_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels, predictions, label_names=PRODUCTS)
    metrics.update(
        {
            "macro_f1_interval": bootstrap_macro_f1(labels, predictions),
            "nll": negative_log_likelihood(probabilities, labels),
            "ece_15_bin": expected_calibration_error(probabilities, labels, n_bins=15),
        }
    )
    return metrics


def _aggregate_risk_coverage(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float]]:
    """Publish fixed coverage steps rather than a row-level confidence trace."""
    predictions = probabilities.argmax(axis=1)
    full_curve = risk_coverage_curve(
        probabilities.max(axis=1),
        predictions == labels,
    )
    count = len(full_curve)
    summary: list[dict[str, float]] = []
    for target_coverage in np.linspace(1.0, 0.05, 96):
        accepted_count = max(1, min(count, round(float(target_coverage) * count)))
        point = full_curve[count - accepted_count]
        summary.append(
            {
                "target_coverage": float(target_coverage),
                "coverage": float(point["coverage"]),
                "risk": float(point["risk"]),
                "threshold": float(point["threshold"]),
            }
        )
    return summary


def _reliability_points(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[list[float], list[float]]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_confidence: list[float] = []
    accuracy: list[float] = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (confidence >= lower) & (confidence <= upper)
            if index == 0
            else (confidence > lower) & (confidence <= upper)
        )
        if mask.any():
            mean_confidence.append(float(confidence[mask].mean()))
            accuracy.append(float(correct[mask].mean()))
    return mean_confidence, accuracy


def _plot_risk_coverage(
    baseline_probabilities: np.ndarray,
    transformer_probabilities: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for name, probabilities, color in (
        ("TF-IDF logistic baseline", baseline_probabilities, "#0f766e"),
        ("DistilBERT", transformer_probabilities, "#7c3aed"),
    ):
        predictions = probabilities.argmax(axis=1)
        curve = risk_coverage_curve(probabilities.max(axis=1), predictions == labels)
        axis.plot(
            [point["coverage"] for point in curve],
            [point["risk"] for point in curve],
            label=name,
            color=color,
            linewidth=2,
        )
    axis.axhline(0.10, color="#64748b", linestyle="--", linewidth=1, label="10% risk")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="coverage (fraction accepted)",
        ylabel="empirical error among accepted cases",
        title="Forward-holdout risk vs. coverage — 2024 Q2",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_reliability(
    baseline_probabilities: np.ndarray,
    transformer_probabilities: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="#64748b", label="perfect calibration")
    for name, probabilities, color in (
        ("TF-IDF logistic baseline", baseline_probabilities, "#0f766e"),
        ("DistilBERT", transformer_probabilities, "#7c3aed"),
    ):
        confidence, accuracy = _reliability_points(probabilities, labels)
        axis.plot(confidence, accuracy, "o-", color=color, label=name)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="mean confidence",
        ylabel="empirical accuracy",
        title="Forward-holdout reliability — 2024 Q2",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scikit-learn", "scipy", "torch", "transformers")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=Path("data/forward_holdout/holdout.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/forward_holdout_manifest.json"),
    )
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/baseline.joblib"))
    parser.add_argument(
        "--transformer-model",
        type=Path,
        default=Path("artifacts/model"),
    )
    parser.add_argument(
        "--transformer-temperature",
        type=Path,
        default=Path("artifacts/temperature.json"),
    )
    parser.add_argument(
        "--transformer-policy",
        type=Path,
        default=Path("artifacts/review_policy.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/forward_holdout_metrics.json"),
    )
    parser.add_argument(
        "--risk-coverage-plot",
        type=Path,
        default=Path("artifacts/forward_holdout_risk_coverage.png"),
    )
    parser.add_argument(
        "--reliability-plot",
        type=Path,
        default=Path("artifacts/forward_holdout_reliability.png"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    calibration_rows = _read_jsonl(args.reference_data_dir / "calibration.jsonl")
    holdout_rows = _read_jsonl(args.holdout)
    if not holdout_rows:
        raise ValueError("forward holdout is empty")
    holdout_labels = np.array([PRODUCTS.index(row["label"]) for row in holdout_rows])
    calibration_labels = np.array([PRODUCTS.index(row["label"]) for row in calibration_rows])

    print("calibrating frozen TF-IDF logistic baseline on the Q1 calibration split")
    baseline = joblib.load(args.baseline)
    calibration_raw = _aligned_baseline_probabilities(baseline, calibration_rows)
    baseline_temperature = fit_temperature(
        probability_logits(calibration_raw),
        calibration_labels,
    )
    calibration_probabilities = apply_probability_temperature(
        calibration_raw,
        baseline_temperature,
    )
    baseline_policy = select_confidence_threshold(
        calibration_probabilities,
        calibration_labels,
        target_accuracy=ExperimentConfig.target_accepted_accuracy,
    )
    holdout_baseline_probabilities = apply_probability_temperature(
        _aligned_baseline_probabilities(baseline, holdout_rows),
        baseline_temperature,
    )

    print(f"running frozen DistilBERT inference on {device}")
    transformer_temperature_artifact = _read_json(args.transformer_temperature)
    transformer_policy = _read_json(args.transformer_policy)
    transformer_temperature = float(transformer_temperature_artifact["temperature"])
    holdout_transformer_probabilities = softmax(
        _transformer_logits(
            args.transformer_model,
            holdout_rows,
            device=device,
            batch_size=args.batch_size,
        )
        / transformer_temperature
    )

    baseline_predictions = holdout_baseline_probabilities.argmax(axis=1)
    transformer_predictions = holdout_transformer_probabilities.argmax(axis=1)
    paired_difference = paired_bootstrap_macro_f1_difference(
        holdout_labels,
        baseline_predictions,
        transformer_predictions,
    )
    baseline_selective = selective_policy_metrics(
        holdout_baseline_probabilities,
        holdout_labels,
        threshold=float(baseline_policy["threshold"]),
        policy_enabled=bool(baseline_policy["target_met"]),
    )
    transformer_selective = selective_policy_metrics(
        holdout_transformer_probabilities,
        holdout_labels,
        threshold=float(transformer_policy["threshold"]),
        policy_enabled=bool(transformer_policy["target_met"]),
    )
    interval = baseline_selective["accepted_accuracy_wilson_95"]
    baseline_transfer_supported = bool(
        baseline_policy["target_met"]
        and baseline_selective["accepted_accuracy"] is not None
        and float(baseline_selective["accepted_accuracy"]) >= 0.90
        and interval is not None
        and float(interval["low"]) >= 0.85
        and float(baseline_selective["coverage"]) >= 0.25
    )
    primary_supported = bool(
        float(paired_difference["estimate"]) > 0 and float(paired_difference["low"]) > 0
    )

    manifest = _read_json(args.manifest)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("manifest protocol does not match the frozen evaluation")
    if int(manifest["holdout"]["count"]) != len(holdout_rows):
        raise ValueError("manifest count does not match the local holdout")

    model_weight_path = args.transformer_model / "model.safetensors"
    results = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "evaluation_status": "completed_forward_holdout",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "protocol_commit": _git_commit(),
        "data": {
            "window": {"start": "2024-04-01", "end": "2024-06-30"},
            "count": len(holdout_rows),
            "class_counts": dict(sorted(Counter(row["label"] for row in holdout_rows).items())),
            "manifest_sha256": _file_sha256(args.manifest),
            "local_content_sha256": manifest["holdout"]["local_content_sha256"],
        },
        "provenance": {
            "baseline_joblib_sha256": _file_sha256(args.baseline),
            "transformer_model_safetensors_sha256": _file_sha256(model_weight_path),
            "transformer_temperature_sha256": _file_sha256(args.transformer_temperature),
            "transformer_policy_sha256": _file_sha256(args.transformer_policy),
            "runtime_versions": _runtime_versions(),
            "inference_device": device,
        },
        "baseline": {
            **_model_metrics(holdout_labels, holdout_baseline_probabilities),
            "q1_calibration": {
                "temperature": baseline_temperature,
                "nll": negative_log_likelihood(calibration_probabilities, calibration_labels),
                "ece_15_bin": expected_calibration_error(
                    calibration_probabilities,
                    calibration_labels,
                    n_bins=15,
                ),
                "review_policy": baseline_policy,
            },
            "q2_selective_policy": baseline_selective,
            "risk_coverage_q2": _aggregate_risk_coverage(
                holdout_baseline_probabilities,
                holdout_labels,
            ),
        },
        "transformer": {
            **_model_metrics(holdout_labels, holdout_transformer_probabilities),
            "frozen_q1_temperature": transformer_temperature,
            "frozen_q1_review_policy": transformer_policy,
            "q2_selective_policy": transformer_selective,
            "risk_coverage_q2": _aggregate_risk_coverage(
                holdout_transformer_probabilities,
                holdout_labels,
            ),
        },
        "comparison": {
            "paired_macro_f1_difference_baseline_minus_transformer": paired_difference,
            "primary_baseline_advantage_supported": primary_supported,
            "baseline_selective_transfer_supported": baseline_transfer_supported,
        },
        "boundaries": [
            "No Q2 row was used for training, calibration, threshold selection, or tuning.",
            "Raw narratives, complaint IDs, issues, and row-level predictions are not published.",
            "This portfolio experiment is not a production or consequential decision system.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_risk_coverage(
        holdout_baseline_probabilities,
        holdout_transformer_probabilities,
        holdout_labels,
        args.risk_coverage_plot,
    )
    _plot_reliability(
        holdout_baseline_probabilities,
        holdout_transformer_probabilities,
        holdout_labels,
        args.reliability_plot,
    )
    print(
        "FORWARD_EVALUATION_OK "
        f"records={len(holdout_rows)} "
        f"baseline_macro_f1={results['baseline']['macro_f1']:.4f} "
        f"transformer_macro_f1={results['transformer']['macro_f1']:.4f} "
        f"delta={paired_difference['estimate']:+.4f} "
        f"primary_supported={str(primary_supported).lower()} "
        f"selective_transfer_supported={str(baseline_transfer_supported).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
