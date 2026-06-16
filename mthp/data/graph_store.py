from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


class ItemGraphStore:
    """Memory-safe access to the provided item interaction matrix.

    Dense .npy files are memory-mapped. Sparse .npz files are loaded as CSR.
    User-specific LxL slices are normalized on demand and cached.
    """

    def __init__(
        self,
        matrix_file: str | Path,
        degree_file: str | Path | None = None,
        matrix_is_normalized: bool = False,
        add_self_loops: bool = True,
        cache_size: int = 20000,
    ) -> None:
        self.matrix_file = Path(matrix_file)
        self.matrix_is_normalized = matrix_is_normalized
        self.add_self_loops = add_self_loops
        self.cache_size = cache_size
        self.cache: OrderedDict[Tuple[int, ...], np.ndarray] = OrderedDict()

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

        self.degree = None
        if degree_file is not None and Path(degree_file).exists():
            degree = np.asarray(np.load(degree_file), dtype=np.float32).reshape(-1)
            if degree.shape[0] < self.num_items:
                raise ValueError("degree vector is shorter than item matrix")
            self.degree = degree[: self.num_items]

    def _raw_slice(self, ids: np.ndarray) -> np.ndarray:
        if self.is_sparse:
            return self.matrix[ids][:, ids].toarray().astype(np.float32, copy=False)
        return np.asarray(self.matrix[np.ix_(ids, ids)], dtype=np.float32)

    def _normalize(self, adjacency: np.ndarray, ids: np.ndarray) -> np.ndarray:
        if self.matrix_is_normalized:
            return adjacency
        a = adjacency.copy()
        if self.add_self_loops:
            a += np.eye(a.shape[0], dtype=np.float32)
        if self.degree is not None:
            degree = self.degree[ids].astype(np.float32, copy=True)
            if self.add_self_loops:
                degree += 1.0
        else:
            degree = a.sum(axis=1)
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
        np.save(path.with_name(path.stem + "_degree.npy"), np.asarray(matrix.sum(axis=1)).reshape(-1))
        return path
