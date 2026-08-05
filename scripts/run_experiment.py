#!/usr/bin/env python3
"""Run the frozen baseline, transformer, calibration, and locked test evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
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
from complaint_intelligence.metrics import (
    bootstrap_macro_f1,
    classification_metrics,
    risk_coverage_curve,
    wilson_interval,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _baseline(seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=60_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=4.0,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )


def _encode(tokenizer: Any, rows: list[dict[str, Any]], config: ExperimentConfig) -> TensorDataset:
    encoded = tokenizer(
        [row["text"] for row in rows],
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([PRODUCTS.index(row["label"]) for row in rows], dtype=torch.long)
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def _logits(model: Any, dataset: TensorDataset, device: str, batch_size: int) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for input_ids, attention_mask, _ in DataLoader(dataset, batch_size=batch_size):
            result = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            output.append(result.detach().cpu().numpy())
    return np.concatenate(output)


def _train_transformer(
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: ExperimentConfig,
    output_dir: Path,
    batch_size: int,
) -> tuple[Any, Any, np.ndarray, np.ndarray, dict[str, Any]]:
    id2label = {index: label for index, label in enumerate(PRODUCTS)}
    label2id = {label: index for index, label in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        num_labels=len(PRODUCTS),
        id2label=id2label,
        label2id=label2id,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    train_dataset = _encode(tokenizer, train_rows, config)
    calibration_dataset = _encode(tokenizer, calibration_rows, config)
    test_dataset = _encode(tokenizer, test_rows, config)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for step, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            result = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(result.loss.detach().cpu()))
            if step % 25 == 0 or step == len(loader):
                progress = f"epoch={epoch + 1}/{config.epochs} step={step}/{len(loader)}"
                print(f"{progress} loss={np.mean(epoch_losses):.4f}")
        losses.append(float(np.mean(epoch_losses)))
    duration = time.perf_counter() - started
    calibration_logits = _logits(model, calibration_dataset, device, batch_size * 2)
    test_logits = _logits(model, test_dataset, device, batch_size * 2)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return (
        model,
        tokenizer,
        calibration_logits,
        test_logits,
        {
            "device": device,
            "epoch_losses": losses,
            "training_seconds": duration,
            "batch_size": batch_size,
        },
    )


def _plot_confusion(labels: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(PRODUCTS)))
    display_labels = [
        {
            "Credit reporting or other personal consumer reports": "Credit reporting",
            "Debt collection": "Debt collection",
            "Credit card": "Credit card",
            "Checking or savings account": "Checking / savings",
            "Mortgage": "Mortgage",
            "Student loan": "Student loan",
            "Money transfer, virtual currency, or money service": "Money transfer",
            "Vehicle loan or lease": "Vehicle loan / lease",
        }[label]
        for label in PRODUCTS
    ]
    figure, axis = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(matrix, display_labels=display_labels).plot(
        ax=axis,
        colorbar=False,
        cmap="Blues",
        xticks_rotation=30,
    )
    axis.set_title("DistilBERT confusion matrix — locked temporal test")
    axis.set_xlabel("Predicted product")
    axis.set_ylabel("Actual product")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_reliability(probabilities: np.ndarray, labels: np.ndarray, path: Path) -> None:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0, 1, 11)
    centers: list[float] = []
    accuracies: list[float] = []
    mean_confidences: list[float] = []
    for index in range(10):
        mask = (
            (confidence >= edges[index]) & (confidence <= edges[index + 1])
            if index == 0
            else ((confidence > edges[index]) & (confidence <= edges[index + 1]))
        )
        if mask.any():
            centers.append(float((edges[index] + edges[index + 1]) / 2))
            accuracies.append(float(correct[mask].mean()))
            mean_confidences.append(float(confidence[mask].mean()))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="#64748b", label="perfect calibration")
    axis.plot(mean_confidences, accuracies, "o-", color="#0f766e", label="model")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence", ylabel="empirical accuracy")
    axis.set_title("Calibrated reliability on temporal test holdout")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = ExperimentConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    train_rows = _read_jsonl(args.data_dir / "train.jsonl")
    calibration_rows = _read_jsonl(args.data_dir / "calibration.jsonl")
    test_rows = _read_jsonl(args.data_dir / "test.jsonl")
    test_labels = np.array([PRODUCTS.index(row["label"]) for row in test_rows])

    print("training fixed TF-IDF + logistic-regression baseline")
    baseline = _baseline(config.seed)
    baseline.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
    baseline_predictions_text = baseline.predict([row["text"] for row in test_rows])
    baseline_predictions = np.array([PRODUCTS.index(label) for label in baseline_predictions_text])
    baseline_metrics = classification_metrics(
        test_labels, baseline_predictions, label_names=PRODUCTS
    )
    baseline_metrics["macro_f1_interval"] = bootstrap_macro_f1(test_labels, baseline_predictions)
    joblib.dump(baseline, args.artifacts / "baseline.joblib", compress=3)

    print("training fixed two-epoch DistilBERT")
    _, _, calibration_logits, test_logits, training = _train_transformer(
        train_rows,
        calibration_rows,
        test_rows,
        config,
        args.artifacts / "model",
        args.batch_size,
    )
    calibration_labels = np.array([PRODUCTS.index(row["label"]) for row in calibration_rows])
    temperature = fit_temperature(calibration_logits, calibration_labels)
    calibration_probabilities = softmax(calibration_logits / temperature)
    test_probabilities = softmax(test_logits / temperature)
    review_policy = select_confidence_threshold(
        calibration_probabilities,
        calibration_labels,
        target_accuracy=config.target_accepted_accuracy,
    )
    test_predictions = test_probabilities.argmax(axis=1)
    transformer_metrics = classification_metrics(
        test_labels, test_predictions, label_names=PRODUCTS
    )
    transformer_metrics.update(
        {
            "macro_f1_interval": bootstrap_macro_f1(test_labels, test_predictions),
            "nll": negative_log_likelihood(test_probabilities, test_labels),
            "ece_15_bin": expected_calibration_error(test_probabilities, test_labels, n_bins=15),
        }
    )
    accepted = test_probabilities.max(axis=1) >= float(review_policy["threshold"])
    if review_policy["target_met"] and accepted.any():
        accepted_correct = test_predictions[accepted] == test_labels[accepted]
        interval = wilson_interval(int(accepted_correct.sum()), int(accepted.sum()))
        selective_test = {
            "accepted_count": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float(accepted_correct.mean()),
            "accepted_accuracy_wilson_95": {"low": interval[0], "high": interval[1]},
        }
    else:
        selective_test = {
            "accepted_count": 0,
            "coverage": 0.0,
            "accepted_accuracy": None,
            "accepted_accuracy_wilson_95": None,
        }

    improvement = float(transformer_metrics["macro_f1"] - baseline_metrics["macro_f1"])
    results = {
        "schema_version": 1,
        "experiment_status": "completed_locked_test",
        "sample_counts": {
            "train": len(train_rows),
            "calibration": len(calibration_rows),
            "test": len(test_rows),
        },
        "primary_hypothesis": {
            "required_macro_f1_improvement": 0.02,
            "observed_macro_f1_improvement": improvement,
            "supported": improvement >= 0.02,
        },
        "baseline": baseline_metrics,
        "transformer": transformer_metrics,
        "calibration": {
            "temperature": temperature,
            "calibration_split_nll": negative_log_likelihood(
                calibration_probabilities, calibration_labels
            ),
            "calibration_split_ece_15_bin": expected_calibration_error(
                calibration_probabilities, calibration_labels, n_bins=15
            ),
        },
        "review_policy_calibration": review_policy,
        "review_policy_test": selective_test,
        "training": training,
        "risk_coverage_test": risk_coverage_curve(
            test_probabilities.max(axis=1), test_predictions == test_labels
        ),
    }
    (args.artifacts / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.artifacts / "temperature.json").write_text(
        json.dumps({"temperature": temperature}, indent=2) + "\n", encoding="utf-8"
    )
    (args.artifacts / "review_policy.json").write_text(
        json.dumps(review_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.artifacts / "label_map.json").write_text(
        json.dumps({str(index): label for index, label in enumerate(PRODUCTS)}, indent=2) + "\n",
        encoding="utf-8",
    )
    for filename in ("temperature.json", "review_policy.json", "label_map.json"):
        (args.artifacts / "model" / filename).write_bytes((args.artifacts / filename).read_bytes())
    _plot_confusion(test_labels, test_predictions, args.artifacts / "confusion_matrix.png")
    _plot_reliability(test_probabilities, test_labels, args.artifacts / "reliability.png")
    print(
        "EXPERIMENT_OK "
        f"baseline_macro_f1={baseline_metrics['macro_f1']:.4f} "
        f"transformer_macro_f1={transformer_metrics['macro_f1']:.4f} "
        f"delta={improvement:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
