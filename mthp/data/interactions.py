from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Interaction:
    user: int
    item: int
    timestamp: float
    raw_user: int
    raw_item: int


@dataclass(frozen=True)
class SequenceExample:
    user: int
    target_item: int
    target_time: float
    history_items: Tuple[int, ...]
    history_times: Tuple[float, ...]
    seen_items: Tuple[int, ...]


def _read_lst(path: str | Path) -> List[Tuple[int, int, float]]:
    rows: List[Tuple[int, int, float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no} must contain user item timestamp")
            rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    if not rows:
        raise ValueError(f"No interactions found in {path}")
    return rows


class InteractionCorpus:
    """Loads the provided .lst files without changing their interaction content.

    User and item namespaces are separated. Item IDs preserve matrix indices whenever
    the supplied matrix shape makes that possible (0-based or 1-based IDs).
    """

    def __init__(
        self,
        train_file: str | Path,
        test_file: str | Path,
        matrix_num_items: Optional[int] = None,
        item_id_mode: str = "auto",
        validation_items_per_user: int = 1,
    ) -> None:
        train_raw = _read_lst(train_file)
        test_raw = _read_lst(test_file)
        all_raw = train_raw + test_raw

        raw_users = sorted({r[0] for r in all_raw})
        self.raw_user_to_id = {raw: idx for idx, raw in enumerate(raw_users)}
        self.id_to_raw_user = raw_users

        raw_items = sorted({r[1] for r in all_raw})
        self.item_offset, self.num_items = self._infer_item_layout(
            raw_items, matrix_num_items, item_id_mode
        )
        self.raw_item_to_id = {raw: self._map_item(raw) for raw in raw_items}
        self.id_to_raw_item = [idx + self.item_offset for idx in range(self.num_items)]

        self.train_interactions = self._map_rows(train_raw)
        self.test_interactions = self._map_rows(test_raw)
        self.num_users = len(raw_users)
        self.validation_items_per_user = validation_items_per_user

        self.train_by_user = self._group(self.train_interactions)
        self.test_by_user = self._group(self.test_interactions)
        self.train_core_by_user, self.validation_examples = self._split_validation()

    @staticmethod
    def _infer_item_layout(
        raw_items: Sequence[int], matrix_num_items: Optional[int], item_id_mode: str
    ) -> Tuple[int, int]:
        min_item, max_item = min(raw_items), max(raw_items)
        if item_id_mode not in {"auto", "zero_based", "one_based"}:
            raise ValueError("item_id_mode must be auto, zero_based, or one_based")
        if item_id_mode == "zero_based":
            size = matrix_num_items or (max_item + 1)
            return 0, size
        if item_id_mode == "one_based":
            size = matrix_num_items or max_item
            return 1, size
        if matrix_num_items is not None:
            if min_item >= 0 and max_item < matrix_num_items:
                return 0, matrix_num_items
            if min_item >= 1 and max_item <= matrix_num_items:
                return 1, matrix_num_items
            raise ValueError(
                f"Raw item IDs [{min_item}, {max_item}] do not fit matrix size {matrix_num_items}"
            )
        if min_item == 0:
            return 0, max_item + 1
        if min_item == 1:
            return 1, max_item
        raise ValueError(
            "Cannot align non-contiguous item IDs without a matrix shape; provide matrix_file or item_id_mode"
        )

    def _map_item(self, raw_item: int) -> int:
        item = raw_item - self.item_offset
        if not 0 <= item < self.num_items:
            raise IndexError(f"Mapped item {item} outside [0, {self.num_items})")
        return item

    def _map_rows(self, rows: Iterable[Tuple[int, int, float]]) -> List[Interaction]:
        mapped = [
            Interaction(
                user=self.raw_user_to_id[u],
                item=self._map_item(i),
                timestamp=t,
                raw_user=u,
                raw_item=i,
            )
            for u, i, t in rows
        ]
        return mapped

    def _group(self, rows: Sequence[Interaction]) -> Dict[int, List[Interaction]]:
        grouped: Dict[int, List[Interaction]] = {u: [] for u in range(len(self.raw_user_to_id))}
        for row in rows:
            grouped[row.user].append(row)
        for user in grouped:
            grouped[user].sort(key=lambda x: x.timestamp)
        return grouped

    def _split_validation(self) -> Tuple[Dict[int, List[Interaction]], List[SequenceExample]]:
        core: Dict[int, List[Interaction]] = {}
        examples: List[SequenceExample] = []
        n_val = max(0, int(self.validation_items_per_user))
        for user, seq in self.train_by_user.items():
            if n_val == 0 or len(seq) <= n_val:
                core[user] = list(seq)
                continue
            split = len(seq) - n_val
            core[user] = list(seq[:split])
            running = list(seq[:split])
            for event in seq[split:]:
                examples.append(self._make_example(user, event, running))
                running.append(event)
        return core, examples

    @staticmethod
    def _make_example(
        user: int, target: Interaction, history: Sequence[Interaction]
    ) -> SequenceExample:
        return SequenceExample(
            user=user,
            target_item=target.item,
            target_time=target.timestamp,
            history_items=tuple(x.item for x in history),
            history_times=tuple(x.timestamp for x in history),
            seen_items=tuple(sorted({x.item for x in history})),
        )

    def training_examples(self, min_history: int = 1) -> List[SequenceExample]:
        examples: List[SequenceExample] = []
        for user, seq in self.train_core_by_user.items():
            for idx in range(min_history, len(seq)):
                examples.append(self._make_example(user, seq[idx], seq[:idx]))
        return examples

    def test_examples(self, new_items_only: bool = False) -> List[SequenceExample]:
        examples: List[SequenceExample] = []
        for user in range(self.num_users):
            history = list(self.train_by_user.get(user, []))
            seen = {x.item for x in history}
            for event in self.test_by_user.get(user, []):
                include = (event.item not in seen) if new_items_only else True
                if include:
                    examples.append(self._make_example(user, event, history))
                history.append(event)
                seen.add(event.item)
        return examples

    def item_popularity(self) -> np.ndarray:
        counts = np.zeros(self.num_items, dtype=np.float64)
        for seq in self.train_core_by_user.values():
            for event in seq:
                counts[event.item] += 1.0
        return counts

    def train_user_item_sets(self) -> Dict[int, set[int]]:
        return {
            user: {event.item for event in seq}
            for user, seq in self.train_core_by_user.items()
        }
