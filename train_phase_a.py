from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mthp.config import load_config
from mthp.data import InteractionCorpus, ItemGraphStore, SequenceDataset
from mthp.data.sequence_dataset import NegativeSampler
from mthp.models import MTHPHC
from mthp.training import ExperimentRunner
from mthp.utils import ensure_dir, resolve_device, save_json, set_seed, setup_logging


def discover_graph_file(configured: str | None, train_file: str, patterns: list[str]) -> str | None:
    if configured and configured != "auto":
        return configured if Path(configured).exists() else configured
    train_dir = Path(train_file).resolve().parent
    matches: list[Path] = []
    for root in (train_dir, train_dir.parent):
        for pattern in patterns:
            matches.extend(sorted(root.glob(pattern)))
    unique = sorted({p.resolve() for p in matches})
    if len(unique) > 1:
        raise RuntimeError(
            "Automatic graph discovery is ambiguous. Set graph.matrix_file/degree_file explicitly: "
            + ", ".join(str(p) for p in unique)
        )
    return str(unique[0]) if unique else None


def matrix_shape(path: str | None) -> int | None:
    if not path or not Path(path).exists():
        return None
    p = Path(path)
    if p.suffix == ".npy":
        return int(np.load(p, mmap_mode="r").shape[0])
    if p.suffix == ".npz":
        import scipy.sparse as sp

        return int(sp.load_npz(p).shape[0])
    raise ValueError("Unsupported item matrix format")


def prepare_graph(cfg: Dict[str, Any], corpus: InteractionCorpus) -> str:
    graph_cfg = cfg["graph"]
    matrix_file = graph_cfg.get("matrix_file")
    rebuild = bool(graph_cfg.get("rebuild_from_train", False))
    strict = bool(graph_cfg.get("strict_no_leakage", False))

    internal_validation = (
        not corpus.has_explicit_validation
        and int(cfg["data"]["validation_items_per_user"]) > 0
    )
    if strict and internal_validation and not rebuild:
        raise ValueError(
            "strict_no_leakage=true with an internal validation split requires "
            "graph.rebuild_from_train=true, because a prebuilt graph may contain validation edges"
        )

    if matrix_file and Path(matrix_file).exists() and not rebuild:
        return str(matrix_file)
    if not rebuild and not graph_cfg["build_if_missing"]:
        raise FileNotFoundError(f"Missing item matrix: {matrix_file}")

    cache_root = ensure_dir(Path(cfg["output_dir"]) / Path(cfg["_config_path"]).stem / "graph_cache")
    output = cache_root / "training_only_item_graph.npz"
    sequences = {
        user: [event.item for event in seq]
        for user, seq in corpus.train_core_by_user.items()
    }
    logging.warning("Building a training-only sparse item graph at %s", output)
    ItemGraphStore.build_from_user_sequences(sequences, corpus.num_items, output)
    graph_cfg["matrix_file"] = str(output)
    graph_cfg["degree_file"] = str(output.with_name(output.stem + "_graph_degree.npy"))
    return str(output)


def run(config_path: str, seed_override: int | None = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = seed_override
    seed = int(cfg["seed"])
    set_seed(seed)
    run_dir = ensure_dir(Path(cfg["output_dir"]) / Path(config_path).stem / f"seed_{seed}")
    setup_logging(run_dir)

    cfg["graph"]["matrix_file"] = discover_graph_file(
        cfg["graph"].get("matrix_file"),
        cfg["data"]["train_file"],
        ["item_matrix.npy", "item_matrix_top*.npy", "item_matrix*.npz"],
    )
    if cfg["graph"].get("degree_source") == "file":
        cfg["graph"]["degree_file"] = discover_graph_file(
            cfg["graph"].get("degree_file"),
            cfg["data"]["train_file"],
            ["*graph_degree*.npy", "*degree*.npy"],
        )

    graph_items = matrix_shape(cfg["graph"]["matrix_file"])
    corpus = InteractionCorpus(
        train_file=cfg["data"]["train_file"],
        valid_file=cfg["data"].get("valid_file"),
        test_file=cfg["data"]["test_file"],
        test_new_file=cfg["data"].get("test_new_file"),
        matrix_num_items=graph_items,
        item_id_mode=cfg["data"]["item_id_mode"],
        validation_items_per_user=int(cfg["data"]["validation_items_per_user"]),
    )
    matrix_file = prepare_graph(cfg, corpus)
    graph = ItemGraphStore(
        matrix_file=matrix_file,
        degree_file=cfg["graph"].get("degree_file"),
        matrix_is_normalized=bool(cfg["graph"]["matrix_is_normalized"]),
        self_loop_mode=str(cfg["graph"]["self_loop_mode"]),
        degree_source=str(cfg["graph"]["degree_source"]),
        cache_size=int(cfg["graph"]["cache_size"]),
    )
    if graph.num_items != corpus.num_items:
        raise ValueError("Item matrix size and corpus item count are inconsistent")

    sampler = NegativeSampler(
        corpus.item_popularity(),
        power=float(cfg["train"]["negative_power"]),
        exclude_seen=bool(cfg["train"]["exclude_seen_negatives"]),
        seed=seed,
    )
    common = {
        "max_history": int(cfg["data"]["max_history"]),
        "pad_item_id": corpus.num_items,
    }
    train_ds = SequenceDataset(
        corpus.training_examples(int(cfg["data"]["min_train_history"])),
        negative_sampler=sampler,
        negative_samples=int(cfg["train"]["negative_samples"]),
        **common,
    )
    valid_ds = SequenceDataset(corpus.validation_examples, **common)
    test_ds = SequenceDataset(corpus.test_examples(False), **common)
    new_ds = (
        SequenceDataset(corpus.new_test_examples(), **common)
        if cfg["evaluation"]["evaluate_new_items"]
        else None
    )

    model = MTHPHC(
        num_users=corpus.num_users,
        num_items=corpus.num_items,
        timestamp_unit=cfg["data"]["timestamp_unit"],
        config=cfg["model"],
    )
    device = resolve_device(cfg["device"])
    logging.info(
        "dataset users=%d items=%d train=%d valid=%d test=%d new_test=%d device=%s",
        corpus.num_users,
        corpus.num_items,
        len(train_ds),
        len(valid_ds),
        len(test_ds),
        len(new_ds) if new_ds else 0,
        device,
    )
    summary = {
        **corpus.split_statistics(),
        "item_offset": corpus.item_offset,
        "train_examples": len(train_ds),
        "valid_examples": len(valid_ds),
        "test_examples": len(test_ds),
        "new_test_examples": len(new_ds) if new_ds else 0,
        "matrix_file": matrix_file,
        "degree_source": cfg["graph"]["degree_source"],
        "self_loop_mode": cfg["graph"]["self_loop_mode"],
    }
    save_json(summary, run_dir / "dataset_summary.json")
    save_json(cfg, run_dir / "resolved_config.json")

    runner = ExperimentRunner(cfg, model, graph, train_ds, valid_ds, test_ds, new_ds, device, run_dir)
    return runner.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MTHP-HC")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    result = run(args.config, args.seed)
    print(result)


if __name__ == "__main__":
    main()
