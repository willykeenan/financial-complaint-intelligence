"""Immutable, shell-free PowerSwarm lane verifier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LANES = {
    "data-pipeline": "tests/test_data.py",
    "calibration-policy": "tests/test_calibration.py",
    "evaluation-metrics": "tests/test_metrics.py",
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in LANES:
        print("usage: verify_lane.py <data-pipeline|calibration-policy|evaluation-metrics>")
        return 2
    lane = sys.argv[1]
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", LANES[lane]],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    print(f"LANE_OK {lane}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
