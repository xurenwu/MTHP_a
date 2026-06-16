from pathlib import Path

import numpy as np
import torch

from mthp.data.graph_store import ItemGraphStore
from mthp.data.interactions import InteractionCorpus
from mthp.data.sequence_dataset import NegativeSampler, SequenceDataset, collate_sequence_batch
from mthp.models import MTHPHC


def _write(path: Path, rows):
    path.write_text("\n".join(f"{u} {i} {t}" for u, i, t in rows) + "\n", encoding="utf-8")


def test_end_to_end_forward(tmp_path: Path):
    train = tmp_path / "train.lst"
    test = tmp_path / "test.lst"
    _write(train, [(10, 0, 1), (10, 1, 2), (10, 2, 3), (11, 1, 1), (11, 3, 2), (11, 4, 3)])
    _write(test, [(10, 3, 4), (11, 0, 4)])
    matrix = np.eye(5, dtype=np.float32)
    matrix[0, 1] = matrix[1, 0] = 1
    matrix[1, 2] = matrix[2, 1] = 1
    matrix_path = tmp_path / "item_matrix.npy"
    degree_path = tmp_path / "degree.npy"
    np.save(matrix_path, matrix)
    np.save(degree_path, matrix.sum(axis=1))

    corpus = InteractionCorpus(train, test, matrix_num_items=5, validation_items_per_user=1)
    graph = ItemGraphStore(matrix_path, degree_path)
    sampler = NegativeSampler(corpus.item_popularity(), seed=7)
    ds = SequenceDataset(
        corpus.training_examples(), max_history=3, pad_item_id=5,
        negative_sampler=sampler, negative_samples=2,
    )
    batch = collate_sequence_batch([ds[0], ds[1]])
    adjacency = graph.batch_slice(batch["history_items"], batch["history_mask"])
    cfg = {
        "embedding_dim": 16,
        "shcn_layers": 2,
        "shcn_heads": 1,
        "shcn_dropout": 0.0,
        "graph_mix_coeff": 0.1,
        "shcn_residual": False,
        "user_fusion": "add",
        "similarity": "cosine",
        "granularities": ["hour", "day", "week"],
        "granularity_weights": "learnable_global",
        "decay_mode": "shared_user",
        "decay_init": 1.0,
        "positive_intensity": False,
    }
    model = MTHPHC(corpus.num_users, corpus.num_items, "hour", cfg)
    pos, neg = model(
        batch["user"], batch["target_item"], batch["negative_items"],
        batch["target_time"], batch["history_items"], batch["history_times"],
        batch["history_mask"], adjacency,
    )
    assert pos.shape == (2,)
    assert neg.shape == (2, 2)
    loss = model.bpr_loss(pos, neg)
    loss.backward()
    assert model.shcn.layers[0].q_proj.weight.grad is not None
    scores = model.score_items(
        batch["user"], torch.arange(5), batch["target_time"], batch["history_items"],
        batch["history_times"], batch["history_mask"], adjacency,
    )
    assert scores.shape == (2, 5)


def test_one_based_item_alignment(tmp_path: Path):
    train = tmp_path / "train.lst"
    test = tmp_path / "test.lst"
    _write(train, [(100, 1, 1), (100, 2, 2)])
    _write(test, [(100, 3, 3)])
    corpus = InteractionCorpus(train, test, matrix_num_items=3, item_id_mode="auto", validation_items_per_user=0)
    assert corpus.item_offset == 1
    assert corpus.train_interactions[0].item == 0
    assert corpus.test_interactions[0].item == 2
