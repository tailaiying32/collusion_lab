"""CLI wrapper for CollusionLab offline steganography analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.metrics.steganography import (  # noqa: E402
    analyze_log,
    analyze_rows,
    build_feature_rows,
    load_log,
    resolve_log_path,
    score_features,
    write_feature_csv,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a CollusionLab log for offline steganography signals."
    )
    parser.add_argument("path", help="Run directory or log.jsonl path")
    parser.add_argument("--min-feature-count", type=int, default=2)
    parser.add_argument("--csv", help="Optional CSV path for top feature scores")
    args = parser.parse_args(argv)

    summary = analyze_log(args.path, min_feature_count=args.min_feature_count)
    if args.csv:
        write_feature_csv(summary, args.csv)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
