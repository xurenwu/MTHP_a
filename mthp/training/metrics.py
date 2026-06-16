from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def rank_metrics(ranks: Sequence[int], ks: Iterable[int]) -> Dict[str, float]:
    ranks_np = np.asarray(ranks, dtype=np.int64)
    result: Dict[str, float] = {}
    if ranks_np.size == 0:
        for k in ks:
            result[f"recall@{k}"] = 0.0
            result[f"mrr@{k}"] = 0.0
            result[f"ndcg@{k}"] = 0.0
        return result
    for k in ks:
        hit = ranks_np <= k
        result[f"recall@{k}"] = float(hit.mean())
        result[f"mrr@{k}"] = float(np.where(hit, 1.0 / ranks_np, 0.0).mean())
        result[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks_np + 1), 0.0).mean())
    return result
