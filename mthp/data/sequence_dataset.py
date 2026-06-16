from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .interactions import SequenceExample


class NegativeSampler:
    def __init__(
        self,
        popularity: np.ndarray,
        power: float = 0.75,
        exclude_seen: bool = False,
        seed: int = 2026,
    ) -> None:
        weights = np.power(popularity, power)
        observed = weights > 0
        if not observed.any():
            raise ValueError("Training data contains no observed items")
        self.candidates = np.flatnonzero(observed)
        self.probabilities = weights[self.candidates]
        self.probabilities /= self.probabilities.sum()
        self.exclude_seen = exclude_seen
        self.rng = np.random.default_rng(seed)

    def sample(self, target: int, seen: Sequence[int], size: int) -> np.ndarray:
        forbidden = {target}
        if self.exclude_seen:
            forbidden.update(seen)
        result: List[int] = []
        while len(result) < size:
            draw = self.rng.choice(self.candidates, size=max(size * 2, 16), p=self.probabilities)
            for item in draw.tolist():
                if item not in forbidden:
                    result.append(item)
                    if len(result) == size:
                        break
        return np.asarray(result, dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[SequenceExample],
        max_history: int,
        pad_item_id: int,
        negative_sampler: NegativeSampler | None = None,
        negative_samples: int = 0,
    ) -> None:
        self.examples = list(examples)
        self.max_history = max_history
        self.pad_item_id = pad_item_id
        self.negative_sampler = negative_sampler
        self.negative_samples = negative_samples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ex = self.examples[index]
        hist_items = list(ex.history_items[-self.max_history :])
        hist_times = list(ex.history_times[-self.max_history :])
        length = len(hist_items)

        items = np.full(self.max_history, self.pad_item_id, dtype=np.int64)
        times = np.zeros(self.max_history, dtype=np.float32)
        mask = np.zeros(self.max_history, dtype=np.float32)
        if length:
            items[-length:] = np.asarray(hist_items, dtype=np.int64)
            times[-length:] = np.asarray(hist_times, dtype=np.float32)
            mask[-length:] = 1.0

        result: Dict[str, Any] = {
            "user": np.int64(ex.user),
            "target_item": np.int64(ex.target_item),
            "target_time": np.float32(ex.target_time),
            "history_items": items,
            "history_times": times,
            "history_mask": mask,
            "seen_items": ex.seen_items,
        }
        if self.negative_sampler is not None and self.negative_samples > 0:
            result["negative_items"] = self.negative_sampler.sample(
                ex.target_item, ex.seen_items, self.negative_samples
            )
        return result


def collate_sequence_batch(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = [
        "user",
        "target_item",
        "target_time",
        "history_items",
        "history_times",
        "history_mask",
    ]
    if "negative_items" in rows[0]:
        tensor_keys.append("negative_items")
    batch: Dict[str, Any] = {}
    for key in tensor_keys:
        values = np.stack([row[key] for row in rows])
        batch[key] = torch.from_numpy(values)
    batch["seen_items"] = [row["seen_items"] for row in rows]
    return batch
