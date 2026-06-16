from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULTS: Dict[str, Any] = {
    "seed": 2026,
    "device": "cuda",
    "output_dir": "outputs/phase_a",
    "data": {
        "train_file": None,
        "test_file": None,
        "test_new_file": None,
        "timestamp_unit": "hour",
        "max_history": 3,
        "validation_items_per_user": 1,
        "min_train_history": 1,
        "item_id_mode": "auto",
    },
    "graph": {
        "matrix_file": None,
        "degree_file": None,
        "matrix_is_normalized": False,
        "add_self_loops": True,
        "degree_source": "file_or_matrix",
        "build_if_missing": True,
        "cache_size": 20000,
    },
    "model": {
        "embedding_dim": 256,
        "shcn_layers": 4,
        "shcn_heads": 1,
        "shcn_dropout": 0.1,
        "graph_mix_coeff": 0.1,
        "shcn_residual": False,
        "user_fusion": "add",
        "similarity": "cosine",
        "granularities": ["hour", "day", "week"],
        "granularity_weights": "learnable_global",
        "decay_mode": "shared_user",
        "decay_init": 1.0,
        "positive_intensity": False,
    },
    "train": {
        "epochs": 200,
        "batch_size": 512,
        "eval_batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "negative_samples": 5,
        "negative_power": 0.75,
        "exclude_seen_negatives": False,
        "num_workers": 0,
        "patience": 20,
        "monitor": "recall@20",
        "gradient_clip": 5.0,
        "amp": False,
    },
    "evaluation": {
        "ks": [10, 20, 30],
        "full_ranking": True,
        "item_chunk_size": 4096,
        "mask_seen_items": False,
        "evaluate_new_items": True,
    },
}


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg["_config_path"] = str(path.resolve())
    return cfg
