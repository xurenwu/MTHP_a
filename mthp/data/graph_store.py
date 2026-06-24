from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


class ItemGraphStore:
    """Memory-safe access to an item interaction matrix.

    ``degree_source`` controls normalization:
    - ``local``: normalize every sliced user-history subgraph by its own degree;
    - ``file``: use a supplied global graph-degree vector;
    - ``matrix``: compute global row degrees from the complete matrix.

    ``self_loop_mode`` is one of ``auto`` (ensure diagonal >= 1 without
    double-counting), ``add`` (always add one), or ``none``.
    """

    def __init__(
        self,
        matrix_file: str | Path,
        degree_file: str | Path | None = None,
        matrix_is_normalized: bool = False,
        self_loop_mode: str = "auto",
        degree_source: str = "local",
        cache_size: int = 20000,
    ) -> None:
        self.matrix_file = Path(matrix_file)
        self.matrix_is_normalized = bool(matrix_is_normalized)
        self.self_loop_mode = str(self_loop_mode)
        self.degree_source = str(degree_source)
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[Tuple[int, ...], np.ndarray] = OrderedDict()

        if self.self_loop_mode not in {"auto", "add", "none"}:
            raise ValueError("self_loop_mode must be auto, add, or none")
        if self.degree_source not in {"local", "file", "matrix"}:
            raise ValueError("degree_source must be local, file, or matrix")

        if self.matrix_file.suffix == ".npy":
            self.matrix = np.load(self.matrix_file, mmap_mode="r")
            self.is_sparse = False
        elif self.matrix_file.suffix == ".npz":
            self.matrix = sp.load_npz(self.matrix_file).tocsr()
            self.is_sparse = True
        else:
            raise ValueError("item matrix must be .npy or scipy sparse .npz")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("item matrix must be square")
        self.num_items = int(self.matrix.shape[0])

        self.degree_file = Path(degree_file) if degree_file else None
        self.degree: np.ndarray | None = None
        if not self.matrix_is_normalized and self.degree_source == "file":
            if self.degree_file is None or not self.degree_file.exists():
                raise FileNotFoundError(
                    "degree_source='file' requires an existing graph degree_file"
                )
            degree = np.asarray(np.load(self.degree_file, mmap_mode="r"), dtype=np.float32).reshape(-1)
            if degree.shape[0] < self.num_items:
                raise ValueError("degree vector is shorter than item matrix")
            self.degree = degree[: self.num_items]
        elif not self.matrix_is_normalized and self.degree_source == "matrix":
            self.degree = self._load_or_compute_matrix_degree()

    def _load_or_compute_matrix_degree(self) -> np.ndarray:
        if self.degree_file is not None and self.degree_file.exists():
            degree = np.asarray(np.load(self.degree_file, mmap_mode="r"), dtype=np.float32).reshape(-1)
            if degree.shape[0] < self.num_items:
                raise ValueError("degree vector is shorter than item matrix")
            return degree[: self.num_items]

        if self.is_sparse:
            degree = np.asarray(self.matrix.sum(axis=1)).reshape(-1).astype(np.float32)
        else:
            degree = np.zeros(self.num_items, dtype=np.float64)
            block = 1024
            for start in range(0, self.num_items, block):
                stop = min(start + block, self.num_items)
                degree[start:stop] = np.asarray(self.matrix[start:stop]).sum(axis=1)
            degree = degree.astype(np.float32)

        if self.degree_file is not None:
            self.degree_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.degree_file, degree)
        return degree

    def _raw_slice(self, ids: np.ndarray) -> np.ndarray:
        if self.is_sparse:
            return self.matrix[ids][:, ids].toarray().astype(np.float32, copy=False)
        return np.asarray(self.matrix[np.ix_(ids, ids)], dtype=np.float32)

    def _apply_self_loops(self, adjacency: np.ndarray) -> np.ndarray:
        a = adjacency.copy()
        if self.self_loop_mode == "add":
            a += np.eye(a.shape[0], dtype=np.float32)
        elif self.self_loop_mode == "auto":
            diagonal = np.diag(a).copy()
            np.fill_diagonal(a, np.maximum(diagonal, 1.0))
        return a

    def _normalize(self, adjacency: np.ndarray, ids: np.ndarray) -> np.ndarray:
        if self.matrix_is_normalized:
            return adjacency.astype(np.float32, copy=False)

        a = self._apply_self_loops(adjacency)
        if self.degree_source == "local":
            degree = a.sum(axis=1)
        else:
            if self.degree is None:
                raise RuntimeError("global graph degree was not initialized")
            degree = self.degree[ids].astype(np.float32, copy=True)
            if self.self_loop_mode == "add":
                degree += 1.0
            elif self.self_loop_mode == "auto":
                raw_diag = np.diag(adjacency)
                degree += (raw_diag < 1.0).astype(np.float32)

        inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        valid = degree > 0
        inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
        return inv_sqrt[:, None] * a * inv_sqrt[None, :]

    def slice(self, item_ids: Sequence[int]) -> np.ndarray:
        key = tuple(int(x) for x in item_ids)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        ids = np.asarray(key, dtype=np.int64)
        if ids.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if ids.min() < 0 or ids.max() >= self.num_items:
            raise IndexError("history item is outside the item matrix")
        value = self._normalize(self._raw_slice(ids), ids)
        if self.cache_size > 0:
            self.cache[key] = value
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return value

    def batch_slice(self, history_items: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        items_np = history_items.detach().cpu().numpy()
        mask_np = history_mask.detach().cpu().numpy().astype(bool)
        batch, length = items_np.shape
        output = np.zeros((batch, length, length), dtype=np.float32)
        for row in range(batch):
            positions = np.flatnonzero(mask_np[row])
            ids = items_np[row, positions]
            if ids.size:
                sub = self.slice(ids.tolist())
                output[row][np.ix_(positions, positions)] = sub
        return torch.from_numpy(output)

    @staticmethod
    def build_from_user_sequences(
        sequences: Dict[int, Sequence[int]], num_items: int, output_file: str | Path
    ) -> Path:
        rows: List[int] = []
        cols: List[int] = []
        for items in sequences.values():
            unique = np.unique(np.asarray(items, dtype=np.int64))
            if unique.size == 0:
                continue
            rr = np.repeat(unique, unique.size)
            cc = np.tile(unique, unique.size)
            rows.extend(rr.tolist())
            cols.extend(cc.tolist())
        data = np.ones(len(rows), dtype=np.float32)
        matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_items, num_items)).tocsr()
        matrix.data[:] = 1.0
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path, matrix)
        np.save(path.with_name(path.stem + "_graph_degree.npy"), np.asarray(matrix.sum(axis=1)).reshape(-1))
        return path
