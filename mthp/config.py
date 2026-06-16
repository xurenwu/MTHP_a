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
        "valid_file": None,
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
        "self_loop_mode": "auto",
        "degree_source": "local",
        "build_if_missing": True,
        "rebuild_from_train": False,
        "strict_no_leakage": False,
        "cache_size": 20000,
    },
    "model": {
        "embedding_dim": 256,
        "shcn_layers": 1,
        "shcn_heads": 1,
        "shcn_dropout": 0.1,
        "graph_mix_coeff": 0.1,
        "shcn_residual": False,
        "shcn_output_projection": False,
        "user_fusion": "structural_only",
        "similarity": "cosine",
        "granularities": ["hour", "day", "week"],
        "granularity_weights": "learnable_global",
        "granularity_hidden_dim": 256,
        "decay_mode": "shared_user",
        "decay_init": 1.0,
        "decay_parameterization": "softplus",
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
        "checkpoint_policy": "best",
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


def _normalize_legacy_options(cfg: Dict[str, Any], raw: Dict[str, Any]) -> None:
    graph_raw = raw.get("graph", {}) if isinstance(raw.get("graph", {}), dict) else {}
    if "add_self_loops" in graph_raw and "self_loop_mode" not in graph_raw:
        cfg["graph"]["self_loop_mode"] = "add" if graph_raw["add_self_loops"] else "none"
    policy = str(cfg["train"]["checkpoint_policy"])
    if policy not in {"best", "last"}:
        raise ValueError("checkpoint_policy must be best or last")


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, raw)
    _normalize_legacy_options(cfg, raw)
    cfg["_config_path"] = str(path.resolve())
    return cfg
