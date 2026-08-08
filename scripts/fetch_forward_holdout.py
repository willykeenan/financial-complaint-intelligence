#!/usr/bin/env python3
"""Acquire the frozen 2024 Q2 holdout without publishing complaint text."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from complaint_intelligence.config import PRODUCTS, ExperimentConfig
from complaint_intelligence.data import deduplicate_records, narrative_hash

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
FULL_SNAPSHOT_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
PROTOCOL_ID = "fci.forward-holdout.2024q2.v1"
HOLDOUT_START = "2024-04-01"
HOLDOUT_END = "2024-06-30"
SAMPLES_PER_PRODUCT = 500


def _sample_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def _download_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        return
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "15",
        "--max-time",
        "3600",
        "--continue-at",
        "-",
        FULL_SNAPSHOT_URL,
        "--output",
        str(path),
    ]
    subprocess.run(command, check=True)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"downloaded CFPB snapshot is not a valid ZIP: {path}")


def _sample_snapshot(
    snapshot: Path,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    columns = [
        "Product",
        "Consumer complaint narrative",
        "Date received",
        "Complaint ID",
        "Issue",
    ]
    by_product: dict[str, list[dict[str, Any]]] = {product: [] for product in PRODUCTS}
    with zipfile.ZipFile(snapshot) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("expected exactly one CSV in the CFPB snapshot")
        with archive.open(members[0]) as handle:
            chunks = pd.read_csv(
                handle,
                usecols=columns,
                dtype=str,
                keep_default_na=False,
                chunksize=100_000,
                low_memory=False,
            )
            for chunk in chunks:
                mask = (
                    chunk["Product"].isin(PRODUCTS)
                    & (chunk["Consumer complaint narrative"] != "")
                    & (chunk["Date received"] >= HOLDOUT_START)
                    & (chunk["Date received"] <= HOLDOUT_END)
                )
                selected = chunk.loc[mask, columns]
                for product, narrative, received, complaint_id, issue in selected.itertuples(
                    index=False,
                    name=None,
                ):
                    by_product[str(product)].append(
                        {
                            "complaint_id": str(complaint_id),
                            "date_received": str(received),
                            "issue": str(issue),
                            "label": str(product),
                            "text": str(narrative),
                        }
                    )

    sampled: list[dict[str, Any]] = []
    exported_counts: dict[str, int] = {}
    for product in PRODUCTS:
        rows = by_product[product]
        rows.sort(key=lambda row: (row["date_received"], row["complaint_id"]))
        exported_counts[product] = len(rows)
        product_sample = _sample_evenly(rows, config.samples_per_product)
        sampled.extend(product_sample)
        print(f"sampled {len(product_sample):4d} / {len(rows):6d} exported {product}")
    return sampled, exported_counts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _reference_hashes(reference_data_dir: Path) -> tuple[set[str], dict[str, str]]:
    hashes: set[str] = set()
    file_hashes: dict[str, str] = {}
    for split in ("train", "calibration", "test"):
        path = reference_data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing original Q1 split: {path}")
        file_hashes[split] = _file_sha256(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    hashes.add(narrative_hash(str(row["text"])))
    return hashes, file_hashes


def _snapshot_metadata() -> dict[str, Any]:
    response = requests.head(
        FULL_SNAPSHOT_URL,
        headers={"User-Agent": "financial-complaint-intelligence/0.1"},
        timeout=30,
    )
    if response.status_code != 200:
        return {}
    return {
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "content_length": response.headers.get("Content-Length"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/forward_holdout"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/forward_holdout_manifest.json"),
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        start_date=HOLDOUT_START,
        end_date=HOLDOUT_END,
        samples_per_product=SAMPLES_PER_PRODUCT,
    )
    reference_hashes, reference_file_hashes = _reference_hashes(args.reference_data_dir)

    snapshot = args.snapshot or args.output / "cache" / "complaints.csv.zip"
    _download_snapshot(snapshot)
    rows, exported_counts = _sample_snapshot(snapshot, config)

    deduplicated, dedup_stats = deduplicate_records(rows)
    holdout = [row for row in deduplicated if str(row["narrative_hash"]) not in reference_hashes]
    overlap_count = len(deduplicated) - len(holdout)
    holdout.sort(
        key=lambda row: (str(row["label"]), str(row["date_received"]), str(row["complaint_id"]))
    )
    output_path = args.output / "holdout.jsonl"
    content_sha = _write_jsonl(output_path, holdout)
    snapshot_meta = _snapshot_metadata()
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": "CFPB Consumer Complaint Database",
            "api": API_URL,
            "full_snapshot_url": FULL_SNAPSHOT_URL,
            "full_snapshot_sha256": _file_sha256(snapshot),
            "snapshot_etag": snapshot_meta.get("etag"),
            "snapshot_last_modified": snapshot_meta.get("last_modified"),
            "snapshot_content_length": snapshot_meta.get("content_length"),
            "window": {"start": HOLDOUT_START, "end": HOLDOUT_END},
            "has_narrative": True,
        },
        "sampling": {
            "method": (
                "local quarter/product/narrative filter over the official full CSV snapshot, "
                "then deterministic evenly spaced rows within each sorted product"
            ),
            "requested_per_product": SAMPLES_PER_PRODUCT,
            "exported_counts": dict(sorted(exported_counts.items())),
            "source_revision_note": "The full daily snapshot can change as source data is revised.",
        },
        "deduplication": {
            **dedup_stats,
            "cross_period_overlap_excluded": overlap_count,
            "holdout_count": len(holdout),
        },
        "holdout": {
            "count": len(holdout),
            "class_counts": dict(sorted(Counter(row["label"] for row in holdout).items())),
            "local_content_sha256": content_sha,
        },
        "reference_q1_split_sha256": reference_file_hashes,
        "privacy": {
            "raw_narratives_published": False,
            "complaint_ids_published": False,
            "row_level_predictions_published": False,
            "note": "Raw Q1/Q2 rows and the full CSV snapshot remain local and gitignored.",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FORWARD_DATA_OK records={len(holdout)} sha256={content_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
