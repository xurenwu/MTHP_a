#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mthp.config import load_config
from mthp.data.interactions import _read_lst


def discover(configured: str | None, train_file: str, patterns: list[str]) -> str | None:
    if configured and configured != "auto" and Path(configured).exists():
        return configured
    train_dir = Path(train_file).resolve().parent
    roots = [train_dir, train_dir.parent]
    matches = []
    for root in roots:
        for pattern in patterns:
            matches.extend(sorted(root.glob(pattern)))
    return str(matches[0]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_file = cfg["data"]["train_file"]
    test_file = cfg["data"]["test_file"]
    train = _read_lst(train_file)
    test = _read_lst(test_file)
    matrix_file = discover(cfg["graph"]["matrix_file"], train_file, ["item_matrix.npy", "item_matrix_top*.npy", "item_matrix*.npz"])
    degree_file = discover(cfg["graph"]["degree_file"], train_file, ["in_degree.npy", "*degree*.npy"])
    report = {
        "train_file": train_file,
        "test_file": test_file,
        "train_rows": len(train),
        "test_rows": len(test),
        "raw_users": len({x[0] for x in train + test}),
        "train_items": len({x[1] for x in train}),
        "all_items": len({x[1] for x in train + test}),
        "min_item_id": min(x[1] for x in train + test),
        "max_item_id": max(x[1] for x in train + test),
        "min_user_id": min(x[0] for x in train + test),
        "max_user_id": max(x[0] for x in train + test),
        "matrix_file": matrix_file,
        "degree_file": degree_file,
    }
    if matrix_file:
        if matrix_file.endswith(".npy"):
            report["matrix_shape"] = list(np.load(matrix_file, mmap_mode="r").shape)
        else:
            import scipy.sparse as sp
            report["matrix_shape"] = list(sp.load_npz(matrix_file).shape)
    if degree_file:
        report["degree_shape"] = list(np.load(degree_file, mmap_mode="r").shape)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
