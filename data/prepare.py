"""High-throughput dataset loader

Reads raw text/JSON/CSV files from an input directory and writes a unified
Parquet file using PyArrow at ≥ 50 MB/s. The script is content-agnostic: any
file containing one JSON or raw text record per line is accepted.

Usage:
    python data/prepare.py --input ./raw --output ./processed/data.parquet

Throughput is logged. If the achieved speed is < 50 MB/s exit code = 1.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MB = 1 << 20
TARGET_MB_S = 50


def collect_lines(files: List[Path]) -> List[str]:
    lines: List[str] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    return lines


def parse_lines(lines: List[str]) -> pa.Table:
    texts: List[str] = []
    for ln in lines:
        if ln.startswith("{") and ln.endswith("}"):
            try:
                obj = json.loads(ln)
                texts.append(obj.get("text", json.dumps(obj)))
            except json.JSONDecodeError:
                texts.append(ln)
        else:
            texts.append(ln)
    arr = pa.array(texts, pa.string())
    return pa.Table.from_arrays([arr], names=["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory with raw files")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file")
    args = parser.parse_args()

    in_dir = Path(args.input)
    files = [p for p in in_dir.rglob("*.*") if p.is_file()]
    if not files:
        raise SystemExit("No files found in input directory")

    start_time = time.perf_counter()
    lines = collect_lines(files)
    table = parse_lines(lines)
    pq.write_table(table, args.output, compression="zstd")
    elapsed = time.perf_counter() - start_time
    size_mb = sum(p.stat().st_size for p in files) / MB
    throughput = size_mb / elapsed if elapsed > 0 else 0
    print(f"Processed {size_mb:.1f} MB in {elapsed:.2f} s → {throughput:.1f} MB/s")

    if throughput < TARGET_MB_S:
        print("Throughput below target", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
