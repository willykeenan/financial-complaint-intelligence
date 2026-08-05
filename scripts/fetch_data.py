#!/usr/bin/env python3
"""Fetch a bounded public CFPB sample without redistributing complaint text."""

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
from complaint_intelligence.data import deduplicate_records, temporal_split

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"


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
        "180",
        "--get",
        API_URL,
        "--data-urlencode",
        f"date_received_min={config.start_date}",
        "--data-urlencode",
        f"date_received_max={config.end_date}",
        "--data-urlencode",
        "has_narrative=true",
        "--data-urlencode",
        f"product={product}",
        "--data-urlencode",
        "format=csv",
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
            rows.append(
                {
                    "complaint_id": str(source.get("Complaint ID", "")),
                    "date_received": str(source.get("Date received", "")),
                    "issue": str(source.get("Issue", "")),
                    "label": product,
                    "text": narrative,
                }
            )
    rows.sort(key=lambda row: (row["date_received"], row["complaint_id"]))
    exported_count = len(rows)
    return product, _sample_evenly(rows, config.samples_per_product), exported_count


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


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
        headers={"User-Agent": "curl/8.7.1"},
        timeout=30,
    )
    if response.status_code != 200:
        return {}
    return response.json().get("_meta", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument(
        "--samples-per-product",
        type=int,
        default=ExperimentConfig.samples_per_product,
    )
    args = parser.parse_args()
    config = ExperimentConfig(samples_per_product=args.samples_per_product)
    rows: list[dict[str, Any]] = []
    exported_counts: dict[str, int] = {}
    cache_dir = args.output / "cache"
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_fetch_product, product, config, cache_dir): product for product in PRODUCTS
        }
        for future in as_completed(futures):
            product, sampled, exported_count = future.result()
            exported_counts[product] = exported_count
            rows.extend(sampled)
            print(f"sampled {len(sampled):4d} / {exported_count:6d} exported {product}")

    deduplicated, dedup_stats = deduplicate_records(rows)
    split = temporal_split(
        deduplicated,
        train_fraction=config.train_fraction,
        calibration_fraction=config.calibration_fraction,
    )
    hashes: dict[str, str] = {}
    for name, split_rows in split.items():
        hashes[name] = _write_jsonl(args.output / "processed" / f"{name}.jsonl", split_rows)

    api_meta = _api_metadata(config)
    manifest = {
        "schema_version": 1,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": "CFPB Consumer Complaint Database",
            "api": API_URL,
            "license": "CC0-1.0",
            "api_last_updated": api_meta.get("last_updated"),
            "window": {"start": config.start_date, "end": config.end_date},
            "has_narrative": True,
        },
        "sampling": {
            "method": "deterministic evenly spaced rows within each filtered CSV export",
            "requested_per_product": config.samples_per_product,
            "exported_counts": dict(sorted(exported_counts.items())),
            "export_limit_note": "CFPB filtered CSV exports may be capped at 100,000 rows.",
        },
        "deduplication": dedup_stats,
        "splits": {
            name: {
                "count": len(split_rows),
                "class_counts": dict(sorted(Counter(row["label"] for row in split_rows).items())),
                "local_content_sha256": hashes[name],
            }
            for name, split_rows in split.items()
        },
        "privacy": {
            "raw_narratives_published": False,
            "complaint_ids_published": False,
            "note": "Local text and CSV files are gitignored; rerun against the source API.",
        },
    }
    artifact_path = Path("artifacts/data_manifest.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DATA_OK {sum(len(value) for value in split.values())} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
