#!/usr/bin/env python3
"""Fail-closed verifier for publishable forward-holdout aggregate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

PROTOCOL_ID = "fci.forward-holdout.2024q2.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_KEYS = {"text", "complaint_id", "issue", "row_predictions", "predictions"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _walk(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"row-level or sensitive key found: {key}")
            _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            _walk(nested)
    elif isinstance(value, str) and "/Users/" in value:
        raise ValueError("absolute local path found in public evidence")


def _finite_rate(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a finite rate")
    return number


def _verify_policy(policy: dict[str, Any], count: int, name: str) -> None:
    accepted = int(policy["accepted_count"])
    reviewed = int(policy["review_count"])
    if accepted < 0 or reviewed < 0 or accepted + reviewed != count:
        raise ValueError(f"{name} policy counts do not match holdout")
    coverage = _finite_rate(policy["coverage"], f"{name} coverage")
    if not math.isclose(coverage, accepted / count, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{name} coverage is inconsistent")
    if accepted == 0:
        if policy["accepted_accuracy"] is not None:
            raise ValueError(f"{name} empty policy reports accepted accuracy")
    else:
        _finite_rate(policy["accepted_accuracy"], f"{name} accepted accuracy")
        interval = policy["accepted_accuracy_wilson_95"]
        low = _finite_rate(interval["low"], f"{name} interval low")
        high = _finite_rate(interval["high"], f"{name} interval high")
        if low > high:
            raise ValueError(f"{name} interval is reversed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/forward_holdout_manifest.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/forward_holdout_metrics.json"),
    )
    args = parser.parse_args()
    manifest = _load(args.manifest)
    metrics = _load(args.metrics)
    _walk(manifest)
    _walk(metrics)

    if manifest.get("protocol_id") != PROTOCOL_ID or metrics.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol IDs do not match the frozen v1 evaluation")
    privacy = manifest["privacy"]
    if any(
        privacy[key]
        for key in (
            "raw_narratives_published",
            "complaint_ids_published",
            "row_level_predictions_published",
        )
    ):
        raise ValueError("manifest reports publication of private or row-level material")

    count = int(manifest["holdout"]["count"])
    if count <= 0 or count != int(metrics["data"]["count"]):
        raise ValueError("manifest and metrics holdout counts do not match")
    if sum(int(value) for value in manifest["holdout"]["class_counts"].values()) != count:
        raise ValueError("manifest class counts do not sum to the holdout")
    if metrics["data"]["class_counts"] != manifest["holdout"]["class_counts"]:
        raise ValueError("metrics and manifest class counts differ")

    for name in ("baseline", "transformer"):
        model = metrics[name]
        _finite_rate(model["accuracy"], f"{name} accuracy")
        _finite_rate(model["macro_f1"], f"{name} macro-F1")
        _finite_rate(model["ece_15_bin"], f"{name} ECE")
        if sum(int(item["support"]) for item in model["per_class"].values()) != count:
            raise ValueError(f"{name} class supports do not sum to the holdout")
        _verify_policy(model["q2_selective_policy"], count, name)
        curve = model["risk_coverage_q2"]
        if len(curve) != 96 or not math.isclose(
            float(curve[0]["target_coverage"]), 1.0, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"{name} risk-coverage summary has the wrong grid")
        if not math.isclose(float(curve[-1]["target_coverage"]), 0.05, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{name} risk-coverage summary has the wrong lower bound")

    paired = metrics["comparison"]["paired_macro_f1_difference_baseline_minus_transformer"]
    observed = float(metrics["baseline"]["macro_f1"]) - float(metrics["transformer"]["macro_f1"])
    if not math.isclose(float(paired["estimate"]), observed, rel_tol=0, abs_tol=1e-12):
        raise ValueError("paired macro-F1 estimate is inconsistent")
    primary = bool(float(paired["estimate"]) > 0 and float(paired["low"]) > 0)
    if primary != bool(metrics["comparison"]["primary_baseline_advantage_supported"]):
        raise ValueError("primary decision does not follow the frozen rule")

    hashes = [
        manifest["source"]["full_snapshot_sha256"],
        manifest["holdout"]["local_content_sha256"],
        metrics["data"]["manifest_sha256"],
        metrics["data"]["local_content_sha256"],
        metrics["provenance"]["baseline_joblib_sha256"],
        metrics["provenance"]["transformer_model_safetensors_sha256"],
        metrics["provenance"]["transformer_temperature_sha256"],
        metrics["provenance"]["transformer_policy_sha256"],
    ]
    if any(not SHA256.fullmatch(str(value)) for value in hashes):
        raise ValueError("missing or malformed SHA-256 provenance")

    digest = hashlib.sha256(args.metrics.read_bytes()).hexdigest()
    print(f"FORWARD_HOLDOUT_EVIDENCE_OK sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
