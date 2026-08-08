#!/usr/bin/env python3
"""Acquire the frozen 2024 Q2 holdout without publishing complaint text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from complaint_intelligence.config import PRODUCTS, ExperimentConfig
from complaint_intelligence.data import deduplicate_records, narrative_hash

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
PROTOCOL_ID = "fci.forward-holdout.2024q2.v1"
HOLDOUT_START = "2024-04-01"
HOLDOUT_END = "2024-06-30"
SAMPLES_PER_PRODUCT = 500


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _sample_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def _fetch_product(
    product: str,
    config: ExperimentConfig,
    cache_dir: Path,
) -> tuple[str, list[dict[str, Any]], int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    export_path = cache_dir / f"{_slug(product)}.csv"
    command = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "15",
        "--max-time",
        "240",
        "--get",
        API_URL,
        "--data-urlencode",
        f"date_received_min={config.start_date}",
        "--data-urlencode",
        f"date_received_max={config.end_date}",
        "--data-urlencode",
        "field=all",
        "--data-urlencode",
        "has_narrative=true",
        "--data-urlencode",
        f"product={product}",
        "--data-urlencode",
        "format=csv",
        "--data-urlencode",
        "no_aggs=true",
        "--output",
        str(export_path),
    ]
    subprocess.run(command, check=True)
    rows: list[dict[str, Any]] = []
    with export_path.open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            narrative = source.get("Consumer complaint narrative")
            if source.get("Product") != product or not narrative:
                continue
            received = str(source.get("Date received", ""))
            if not HOLDOUT_START <= received <= HOLDOUT_END:
                continue
            rows.append(
                {
                    "complaint_id": str(source.get("Complaint ID", "")),
                    "date_received": received,
                    "issue": str(source.get("Issue", "")),
                    "label": product,
                    "text": narrative,
                }
            )
    rows.sort(key=lambda row: (row["date_received"], row["complaint_id"]))
    return product, _sample_evenly(rows, config.samples_per_product), len(rows)


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


def _api_metadata(config: ExperimentConfig) -> dict[str, Any]:
    response = requests.get(
        API_URL,
        params={
            "date_received_min": config.start_date,
            "date_received_max": config.end_date,
            "has_narrative": "true",
            "size": 1,
            "no_aggs": "true",
        },
        headers={"User-Agent": "financial-complaint-intelligence/0.1"},
        timeout=30,
    )
    if response.status_code != 200:
        return {}
    return response.json().get("_meta", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/forward_holdout"))
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

    rows: list[dict[str, Any]] = []
    exported_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_fetch_product, product, config, args.output / "cache"): product
            for product in PRODUCTS
        }
        for future in as_completed(futures):
            product, sampled, exported_count = future.result()
            exported_counts[product] = exported_count
            rows.extend(sampled)
            print(f"sampled {len(sampled):4d} / {exported_count:6d} exported {product}")

    deduplicated, dedup_stats = deduplicate_records(rows)
    holdout = [row for row in deduplicated if str(row["narrative_hash"]) not in reference_hashes]
    overlap_count = len(deduplicated) - len(holdout)
    holdout.sort(
        key=lambda row: (str(row["label"]), str(row["date_received"]), str(row["complaint_id"]))
    )
    output_path = args.output / "holdout.jsonl"
    content_sha = _write_jsonl(output_path, holdout)
    api_meta = _api_metadata(config)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": "CFPB Consumer Complaint Database",
            "api": API_URL,
            "api_last_updated": api_meta.get("last_updated"),
            "window": {"start": HOLDOUT_START, "end": HOLDOUT_END},
            "has_narrative": True,
        },
        "sampling": {
            "method": "deterministic evenly spaced rows within each filtered CSV export",
            "requested_per_product": SAMPLES_PER_PRODUCT,
            "exported_counts": dict(sorted(exported_counts.items())),
            "export_limit_note": "Filtered CFPB CSV exports can change as source data is revised.",
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
            "note": "Raw Q1/Q2 rows and downloaded CSV files remain local and gitignored.",
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
