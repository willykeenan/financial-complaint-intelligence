"""Deterministic preparation helpers for public CFPB complaint narratives."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize narrative text for exact duplicate detection."""
    normalized = unicodedata.normalize("NFKC", str(text))
    return _WHITESPACE.sub(" ", normalized).strip().lower()


def narrative_hash(text: str) -> str:
    """Return a non-reversible content identifier for a normalized narrative."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def deduplicate_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate narratives and fail closed on cross-label conflicts.

    Same-label repeats keep the earliest record. If identical normalized text is
    associated with multiple labels, the entire group is excluded to avoid
    injecting contradictory supervision.
    """
    rows = [dict(record) for record in records]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("text") or not row.get("label"):
            continue
        digest = narrative_hash(str(row["text"]))
        row["narrative_hash"] = digest
        groups[digest].append(row)

    kept: list[dict[str, Any]] = []
    duplicate_count = 0
    conflict_count = 0
    for group in groups.values():
        labels = {str(row["label"]) for row in group}
        if len(labels) != 1:
            conflict_count += len(group)
            continue
        ordered = sorted(
            group,
            key=lambda row: (str(row.get("date_received", "")), str(row.get("complaint_id", ""))),
        )
        kept.append(ordered[0])
        duplicate_count += len(ordered) - 1

    kept.sort(
        key=lambda row: (
            str(row.get("date_received", "")),
            str(row.get("complaint_id", "")),
            str(row.get("label", "")),
        )
    )
    stats = {
        "input_count": len(rows),
        "output_count": len(kept),
        "duplicate_count": duplicate_count,
        "conflict_count": conflict_count,
    }
    return kept, stats


def temporal_split(
    records: Iterable[Mapping[str, Any]],
    *,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Split chronologically within each class with no future-to-past leakage."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between zero and one")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave a test split")

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        row = dict(record)
        by_label[str(row["label"])].append(row)

    result: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "calibration": [],
        "test": [],
    }
    for label, rows in sorted(by_label.items()):
        ordered = sorted(
            rows,
            key=lambda row: (str(row.get("date_received", "")), str(row.get("complaint_id", ""))),
        )
        if len(ordered) < 3:
            raise ValueError(f"label {label!r} needs at least three records")
        train_end = max(1, int(len(ordered) * train_fraction))
        calibration_size = max(1, int(len(ordered) * calibration_fraction))
        calibration_end = train_end + calibration_size
        if calibration_end >= len(ordered):
            calibration_end = len(ordered) - 1
            train_end = min(train_end, calibration_end - 1)
        result["train"].extend(ordered[:train_end])
        result["calibration"].extend(ordered[train_end:calibration_end])
        result["test"].extend(ordered[calibration_end:])

    for rows in result.values():
        rows.sort(
            key=lambda row: (str(row["label"]), str(row["date_received"]), str(row["complaint_id"]))
        )
    return result


def evenly_spaced_offsets(total: int, page_size: int = 100, pages: int = 5) -> list[int]:
    """Return deterministic offsets spanning a result window."""
    if total < 0 or page_size <= 0 or pages <= 0:
        raise ValueError("total must be non-negative and page_size/pages positive")
    if total <= page_size:
        return [0] if total else []
    maximum = total - page_size
    count = min(pages, math.ceil(total / page_size))
    if count == 1:
        return [0]
    return sorted({round(index * maximum / (count - 1)) for index in range(count)})
