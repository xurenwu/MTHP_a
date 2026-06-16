#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from mthp.config import load_config
from mthp.data.interactions import _read_lst


def _matrix(path: str):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, mmap_mode="r"), False
    if p.suffix == ".npz":
        return sp.load_npz(p).tocsr(), True
    raise ValueError(f"Unsupported matrix format: {p}")


def _row_sum(matrix, sparse: bool, block: int = 1024) -> np.ndarray:
    if sparse:
        return np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float64)
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], block):
        stop = min(start + block, matrix.shape[0])
        result[start:stop] = np.asarray(matrix[start:stop]).sum(axis=1)
    return result


def _diagonal_stats(matrix, sparse: bool) -> dict:
    diag = matrix.diagonal() if sparse else np.asarray(matrix).diagonal()
    diag = np.asarray(diag)
    return {
        "diagonal_nonzero": int(np.count_nonzero(diag)),
        "diagonal_zero": int(diag.size - np.count_nonzero(diag)),
        "diagonal_min": float(diag.min()) if diag.size else 0.0,
        "diagonal_max": float(diag.max()) if diag.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--compare-degree", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    paths = {
        "train": cfg["data"].get("train_file"),
        "valid": cfg["data"].get("valid_file"),
        "test": cfg["data"].get("test_file"),
        "test_new": cfg["data"].get("test_new_file"),
    }
    splits = {name: _read_lst(path) if path else [] for name, path in paths.items()}
    all_rows = [row for rows in splits.values() for row in rows]
    if not all_rows:
        raise ValueError("No interactions found")

    matrix_file = cfg["graph"].get("matrix_file")
    report = {
        "files": paths,
        "rows": {name: len(rows) for name, rows in splits.items()},
        "raw_users": len({x[0] for x in all_rows}),
        "train_users": len({x[0] for x in splits["train"]}),
        "all_items": len({x[1] for x in all_rows}),
        "train_items": len({x[1] for x in splits["train"]}),
        "min_item_id": min(x[1] for x in all_rows),
        "max_item_id": max(x[1] for x in all_rows),
        "min_user_id": min(x[0] for x in all_rows),
        "max_user_id": max(x[0] for x in all_rows),
        "matrix_file": matrix_file,
        "degree_file": cfg["graph"].get("degree_file"),
        "degree_source": cfg["graph"].get("degree_source"),
        "self_loop_mode": cfg["graph"].get("self_loop_mode"),
    }

    if matrix_file and Path(matrix_file).exists():
        matrix, sparse = _matrix(matrix_file)
        report["matrix_shape"] = list(matrix.shape)
        report["matrix_dtype"] = str(matrix.dtype)
        report["matrix_sparse"] = sparse
        report.update(_diagonal_stats(matrix, sparse))

        degree_file = cfg["graph"].get("degree_file")
        if args.compare_degree or (degree_file and Path(degree_file).exists()):
            matrix_degree = _row_sum(matrix, sparse)
            report["matrix_degree_min"] = float(matrix_degree.min())
            report["matrix_degree_mean"] = float(matrix_degree.mean())
            report["matrix_degree_max"] = float(matrix_degree.max())
            if degree_file and Path(degree_file).exists():
                supplied = np.asarray(np.load(degree_file, mmap_mode="r"), dtype=np.float64).reshape(-1)
                supplied = supplied[: matrix.shape[0]]
                report["degree_shape"] = list(supplied.shape)
                if supplied.shape[0] == matrix_degree.shape[0]:
                    diff = np.abs(supplied - matrix_degree)
                    report["degree_abs_diff_mean"] = float(diff.mean())
                    report["degree_abs_diff_max"] = float(diff.max())
                    if np.std(supplied) > 0 and np.std(matrix_degree) > 0:
                        report["degree_pearson"] = float(np.corrcoef(supplied, matrix_degree)[0, 1])
                    report["degree_exact_match"] = bool(np.allclose(supplied, matrix_degree))

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
