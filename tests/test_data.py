from complaint_intelligence.data import (
    deduplicate_records,
    evenly_spaced_offsets,
    normalize_text,
    temporal_split,
)


def _record(label: str, day: int, text: str, complaint_id: str) -> dict[str, str]:
    return {
        "label": label,
        "date_received": f"2024-01-{day:02d}",
        "text": text,
        "complaint_id": complaint_id,
    }


def test_normalize_text_is_stable() -> None:
    assert normalize_text("  Fees\n\tWERE   charged. ") == "fees were charged."


def test_dedup_keeps_earliest_and_drops_label_conflicts() -> None:
    records = [
        _record("card", 2, "Duplicate narrative", "2"),
        _record("card", 1, " duplicate   narrative ", "1"),
        _record("mortgage", 3, "Conflicting text", "3"),
        _record("loan", 4, "conflicting text", "4"),
        _record("loan", 5, "Unique narrative", "5"),
    ]
    kept, stats = deduplicate_records(records)
    assert [row["complaint_id"] for row in kept] == ["1", "5"]
    assert stats == {
        "input_count": 5,
        "output_count": 2,
        "duplicate_count": 1,
        "conflict_count": 2,
    }


def test_temporal_split_is_per_label_and_ordered() -> None:
    records = [
        _record(label, day, f"{label} narrative {day}", f"{label}-{day}")
        for label in ("card", "mortgage")
        for day in range(1, 11)
    ]
    split = temporal_split(records, train_fraction=0.7, calibration_fraction=0.15)
    assert {name: len(rows) for name, rows in split.items()} == {
        "train": 14,
        "calibration": 2,
        "test": 4,
    }
    for label in ("card", "mortgage"):
        train_days = [r["date_received"] for r in split["train"] if r["label"] == label]
        cal_days = [r["date_received"] for r in split["calibration"] if r["label"] == label]
        test_days = [r["date_received"] for r in split["test"] if r["label"] == label]
        assert max(train_days) < min(cal_days) < min(test_days)


def test_even_offsets_cover_window_without_overflow() -> None:
    assert evenly_spaced_offsets(total=430, page_size=100, pages=5) == [0, 82, 165, 248, 330]
    assert evenly_spaced_offsets(total=80, page_size=100, pages=5) == [0]
