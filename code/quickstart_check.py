#!/usr/bin/env python3
"""Smoke-check that released tables and a sample trace match the manifests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _count_csv_rows(path: Path) -> int:
    with path.open() as f:
        return max(sum(1 for _ in f) - 1, 0)


def _table_stem(entry: dict) -> str:
    family, size = entry["family"], entry["size"]
    dataset = entry.get("dataset")
    if dataset and dataset not in ("ao", "AO"):
        # e.g. ultrafeedback_binarized -> ultrafeedback filenames in data/
        short = dataset.replace("_binarized", "")
        return f"{family}_{size}_{short}"
    return f"{family}_{size}"


def main() -> int:
    errors: list[str] = []

    manifest = json.loads((ROOT / "manifest.json").read_text())
    for entry in manifest["tables"]:
        stem = _table_stem(entry)
        for kind, key in (("raw", "n_rows_raw"), ("blackbox", "n_rows_blackbox")):
            path = ROOT / "data" / f"{stem}_{kind}.csv"
            if not path.is_file():
                errors.append(f"missing {path.relative_to(ROOT)}")
                continue
            n = _count_csv_rows(path)
            expected = entry[key]
            if n != expected:
                errors.append(f"{path.name}: {n} rows, expected {expected}")

    for name in ("all_checkpoints_raw.csv", "all_checkpoints_blackbox.csv"):
        path = ROOT / "data" / name
        if not path.is_file():
            errors.append(f"missing data/{name}")

    sample = ROOT / "traces/qwen3_0.6b/gp/generous/best_found.csv.gz"
    if not sample.is_file():
        errors.append(f"missing {sample.relative_to(ROOT)}")
    else:
        try:
            import pandas as pd

            df = pd.read_csv(sample, compression="gzip")
            if df.empty:
                errors.append(f"{sample.name} is empty")
            needed = {"search_objective", "optimizer", "seed"}
            missing = needed - set(df.columns)
            if missing:
                errors.append(f"{sample.name} missing columns: {sorted(missing)}")
        except ImportError:
            print("note: pandas not installed; skipped trace column check", file=sys.stderr)

    if errors:
        print("quickstart_check FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    n_tables = len(manifest["tables"])
    print(f"OK: {n_tables} per-size tables match manifest.json")
    print(f"OK: sample trace present at {sample.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
