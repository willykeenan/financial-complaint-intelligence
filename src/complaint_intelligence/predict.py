"""Inference wrapper for calibrated predictions and human-review routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .calibration import softmax


def _support_file(model_id_or_path: str, filename: str) -> Path:
    local = Path(model_id_or_path) / filename
    if local.exists():
        return local
    return Path(hf_hub_download(repo_id=model_id_or_path, filename=filename))


class ComplaintRouter:
    """Load a classifier plus its frozen calibration and review policy."""

    def __init__(self, model_id_or_path: str, *, device: str | None = None) -> None:
        self.model_id_or_path = model_id_or_path
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.temperature = float(
            json.loads(_support_file(model_id_or_path, "temperature.json").read_text())[
                "temperature"
            ]
        )
        self.policy = json.loads(_support_file(model_id_or_path, "review_policy.json").read_text())

    def predict(self, text: str) -> dict[str, Any]:
        narrative = " ".join(str(text).split())
        if len(narrative) < 20:
            raise ValueError("Enter at least 20 characters of complaint context.")
        encoded = self.tokenizer(
            narrative,
            return_tensors="pt",
            truncation=True,
            max_length=192,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits.detach().cpu().numpy()
        probabilities = softmax(logits / self.temperature)[0]
        order = np.argsort(-probabilities)
        top = [
            {
                "label": self.model.config.id2label[int(index)],
                "probability": float(probabilities[index]),
            }
            for index in order[:3]
        ]
        confidence = top[0]["probability"]
        threshold = float(self.policy["threshold"])
        action = (
            "auto_route"
            if self.policy.get("target_met") and confidence >= threshold
            else "human_review"
        )
        return {
            "predicted_product": top[0]["label"],
            "confidence": confidence,
            "recommended_action": action,
            "review_threshold": threshold,
            "top_predictions": top,
            "disclaimer": "Portfolio demonstration only; not for consequential decisions.",
        }
