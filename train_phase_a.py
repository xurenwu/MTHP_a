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
    if configured and configured != "auto" and Path(configured).exists():
        return configured
    train_dir = Path(train_file).resolve().parent
    matches = []
    for root in (train_dir, train_dir.parent):
        for pattern in patterns:
            matches.extend(sorted(root.glob(pattern)))
    return str(matches[0]) if matches else (configured if configured != "auto" else None)

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
    matrix_file = graph_cfg["matrix_file"]
    if matrix_file and Path(matrix_file).exists():
        return matrix_file
    if not graph_cfg["build_if_missing"]:
        raise FileNotFoundError(f"Missing item matrix: {matrix_file}")
    output = Path(matrix_file or cfg["output_dir"]) 
    if output.suffix not in {".npz", ".npy"}:
        output = output / "derived_item_matrix.npz"
    elif output.suffix == ".npy":
        output = output.with_suffix(".npz")
    sequences = {
        user: [event.item for event in seq]
        for user, seq in corpus.train_core_by_user.items()
    }
    logging.warning("Provided item matrix not found; building a training-only sparse cache at %s", output)
    ItemGraphStore.build_from_user_sequences(sequences, corpus.num_items, output)
    graph_cfg["matrix_file"] = str(output)
    if not graph_cfg.get("degree_file"):
        graph_cfg["degree_file"] = str(output.with_name(output.stem + "_degree.npy"))
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
        cfg["graph"].get("matrix_file"), cfg["data"]["train_file"],
        ["item_matrix.npy", "item_matrix_top*.npy", "item_matrix*.npz"],
    )
    cfg["graph"]["degree_file"] = discover_graph_file(
        cfg["graph"].get("degree_file"), cfg["data"]["train_file"],
        ["in_degree.npy", "*degree*.npy"],
    )
    graph_items = matrix_shape(cfg["graph"]["matrix_file"])
    corpus = InteractionCorpus(
        train_file=cfg["data"]["train_file"],
        test_file=cfg["data"]["test_file"],
        matrix_num_items=graph_items,
        item_id_mode=cfg["data"]["item_id_mode"],
        validation_items_per_user=int(cfg["data"]["validation_items_per_user"]),
    )
    matrix_file = prepare_graph(cfg, corpus)
    graph = ItemGraphStore(
        matrix_file=matrix_file,
        degree_file=cfg["graph"]["degree_file"],
        matrix_is_normalized=bool(cfg["graph"]["matrix_is_normalized"]),
        add_self_loops=bool(cfg["graph"]["add_self_loops"]),
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
    new_ds = SequenceDataset(corpus.test_examples(True), **common) if cfg["evaluation"]["evaluate_new_items"] else None

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
    save_json(
        {
            "num_users": corpus.num_users,
            "num_items": corpus.num_items,
            "item_offset": corpus.item_offset,
            "train_examples": len(train_ds),
            "valid_examples": len(valid_ds),
            "test_examples": len(test_ds),
        },
        run_dir / "dataset_summary.json",
    )
    runner = ExperimentRunner(cfg, model, graph, train_ds, valid_ds, test_ds, new_ds, device, run_dir)
    return runner.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the clean Phase-A MTHP-HC reproduction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    result = run(args.config, args.seed)
    print(result)


if __name__ == "__main__":
    main()
