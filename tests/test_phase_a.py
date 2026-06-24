from pathlib import Path

import numpy as np
import torch

from mthp.data.graph_store import ItemGraphStore
from mthp.data.interactions import InteractionCorpus
from mthp.data.sequence_dataset import NegativeSampler, SequenceDataset, collate_sequence_batch
from mthp.models import MTHPHC


def _write(path: Path, rows):
    path.write_text("\n".join(f"{u} {i} {t}" for u, i, t in rows) + "\n", encoding="utf-8")


def _model_cfg(**overrides):
    cfg = {
        "embedding_dim": 16,
        "shcn_layers": 1,
        "shcn_heads": 1,
        "shcn_dropout": 0.0,
        "graph_mix_coeff": 0.1,
        "shcn_residual": False,
        "shcn_output_projection": False,
        "user_fusion": "structural_only",
        "similarity": "cosine",
        "granularities": ["hour", "day", "week"],
        "granularity_weights": "learnable_global",
        "granularity_hidden_dim": 8,
        "decay_mode": "shared_user",
        "decay_init": 1.0,
        "decay_parameterization": "softplus",
        "positive_intensity": False,
    }
    cfg.update(overrides)
    return cfg


def test_end_to_end_forward(tmp_path: Path):
    train = tmp_path / "train.lst"
    test = tmp_path / "test.lst"
    _write(train, [(10, 0, 1), (10, 1, 2), (10, 2, 3), (11, 1, 1), (11, 3, 2), (11, 4, 3)])
    _write(test, [(10, 3, 4), (11, 0, 4)])
    matrix = np.eye(5, dtype=np.float32)
    matrix[0, 1] = matrix[1, 0] = 1
    matrix[1, 2] = matrix[2, 1] = 1
    matrix_path = tmp_path / "item_matrix.npy"
    np.save(matrix_path, matrix)

    corpus = InteractionCorpus(train, test, matrix_num_items=5, validation_items_per_user=0)
    graph = ItemGraphStore(matrix_path, degree_source="local", self_loop_mode="auto")
    sampler = NegativeSampler(corpus.item_popularity(), seed=7)
    ds = SequenceDataset(
        corpus.training_examples(min_history=0), max_history=3, pad_item_id=5,
        negative_sampler=sampler, negative_samples=2,
    )
    batch = collate_sequence_batch([ds[1], ds[2]])
    adjacency = graph.batch_slice(batch["history_items"], batch["history_mask"])
    model = MTHPHC(corpus.num_users, corpus.num_items, "hour", _model_cfg())
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
    assert model.raw_decay.weight.grad is not None


def test_one_based_item_alignment(tmp_path: Path):
    train = tmp_path / "train.lst"
    test = tmp_path / "test.lst"
    _write(train, [(100, 1, 1), (100, 2, 2)])
    _write(test, [(100, 3, 3)])
    corpus = InteractionCorpus(train, test, matrix_num_items=3, item_id_mode="auto", validation_items_per_user=0)
    assert corpus.item_offset == 1
    assert corpus.train_interactions[0].item == 0
    assert corpus.test_interactions[0].item == 2


def test_explicit_validation_and_new_test_files(tmp_path: Path):
    train = tmp_path / "train.lst"
    valid = tmp_path / "valid.lst"
    test = tmp_path / "test.lst"
    new = tmp_path / "new.lst"
    _write(train, [(10, 0, 1), (10, 1, 2)])
    _write(valid, [(10, 2, 3)])
    _write(test, [(10, 1, 4)])
    _write(new, [(10, 3, 5)])
    corpus = InteractionCorpus(
        train, test, valid_file=valid, test_new_file=new,
        matrix_num_items=4, validation_items_per_user=1,
    )
    assert corpus.has_explicit_validation
    assert corpus.validation_examples[0].history_items == (0, 1)
    assert corpus.test_examples()[0].history_items == (0, 1, 2)
    assert corpus.new_test_examples()[0].target_item == 3


def test_auto_self_loop_does_not_double_diagonal(tmp_path: Path):
    matrix = np.array([[1, 1], [1, 0]], dtype=np.float32)
    path = tmp_path / "graph.npy"
    np.save(path, matrix)
    store = ItemGraphStore(path, degree_source="local", self_loop_mode="auto")
    raw = store._apply_self_loops(matrix)
    assert raw[0, 0] == 1.0
    assert raw[1, 1] == 1.0


def test_user_adaptive_granularity_and_positive_decay():
    cfg = _model_cfg(
        granularity_weights="user_adaptive",
        decay_mode="user_granularity",
        positive_intensity=True,
        user_fusion="add",
    )
    model = MTHPHC(2, 5, "hour", cfg)
    users = torch.tensor([0, 1])
    hist = torch.tensor([[0, 1, 5], [2, 3, 4]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    adjacency = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    user_repr, _ = model.encode_context(users, hist, mask, adjacency)
    theta = model._theta(user_repr)
    decay = model._decay(users)
    assert theta.shape == (2, 3)
    assert torch.allclose(theta.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(decay > 0)


def test_structural_only_falls_back_for_empty_history():
    model = MTHPHC(1, 3, "hour", _model_cfg(user_fusion="structural_only"))
    users = torch.tensor([0])
    hist = torch.tensor([[3, 3, 3]])
    mask = torch.zeros((1, 3))
    adjacency = torch.zeros((1, 3, 3))
    user_repr, _ = model.encode_context(users, hist, mask, adjacency)
    assert torch.isfinite(user_repr).all()
    assert torch.linalg.norm(user_repr, dim=-1).item() > 0.99
