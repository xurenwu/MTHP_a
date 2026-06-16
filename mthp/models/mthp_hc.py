from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shcn import StructuralHypergraphConvolutionNetwork


_TIME_SCALE_IN_HOURS = {"hour": 1.0, "day": 24.0, "week": 168.0}
_TIMESTAMP_TO_HOURS = {"second": 1.0 / 3600.0, "minute": 1.0 / 60.0, "hour": 1.0, "day": 24.0}


class MTHPHC(nn.Module):
    """Clean implementation of the MTHP-HC method described in the paper.

    Phase A intentionally keeps the paper's signed cosine Hawkes score and BPR objective.
    `positive_intensity=True` is provided only as a controlled compatibility switch for Phase B.
    """

    def __init__(self, num_users: int, num_items: int, timestamp_unit: str, config: Dict) -> None:
        super().__init__()
        dim = int(config["embedding_dim"])
        self.num_users = num_users
        self.num_items = num_items
        self.pad_item_id = num_items
        self.timestamp_unit = timestamp_unit
        if timestamp_unit not in _TIMESTAMP_TO_HOURS:
            raise ValueError(f"Unsupported timestamp_unit: {timestamp_unit}")
        self.user_fusion = config["user_fusion"]
        self.similarity = config["similarity"]
        self.granularities: List[str] = list(config["granularities"])
        for name in self.granularities:
            if name not in _TIME_SCALE_IN_HOURS:
                raise ValueError(f"Unsupported granularity: {name}")
        self.granularity_weights = config["granularity_weights"]
        self.decay_mode = config["decay_mode"]
        self.positive_intensity = bool(config.get("positive_intensity", False))

        self.user_embedding = nn.Embedding(num_users, dim)
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=self.pad_item_id)
        bound = 0.5 / dim
        nn.init.uniform_(self.user_embedding.weight, -bound, bound)
        nn.init.uniform_(self.item_embedding.weight, -bound, bound)
        with torch.no_grad():
            self.item_embedding.weight[self.pad_item_id].zero_()

        self.shcn = StructuralHypergraphConvolutionNetwork(
            dim=dim,
            layers=int(config["shcn_layers"]),
            heads=int(config["shcn_heads"]),
            dropout=float(config["shcn_dropout"]),
            graph_mix_coeff=float(config["graph_mix_coeff"]),
            residual=bool(config["shcn_residual"]),
        )

        decay_init = float(config["decay_init"])
        if self.decay_mode == "shared_user":
            self.raw_decay = nn.Embedding(num_users, 1)
        elif self.decay_mode == "user_granularity":
            self.raw_decay = nn.Embedding(num_users, len(self.granularities))
        else:
            raise ValueError("decay_mode must be shared_user or user_granularity")
        nn.init.constant_(self.raw_decay.weight, decay_init)

        if self.granularity_weights == "learnable_global":
            self.theta_logits = nn.Parameter(torch.zeros(len(self.granularities)))
        elif self.granularity_weights == "uniform":
            self.register_buffer("theta_logits", torch.zeros(len(self.granularities)))
        else:
            raise ValueError("granularity_weights must be learnable_global or uniform")

    def encode_user(
        self,
        users: torch.Tensor,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_embeddings = self.item_embedding(history_items)
        structural = self.shcn(history_embeddings, adjacency, history_mask)
        identity = F.normalize(self.user_embedding(users), p=2, dim=-1, eps=1e-12)
        if self.user_fusion == "add":
            user_repr = F.normalize(identity + structural, p=2, dim=-1, eps=1e-12)
        elif self.user_fusion == "structural_only":
            user_repr = structural
        elif self.user_fusion == "identity_only":
            user_repr = identity
        else:
            raise ValueError("user_fusion must be add, structural_only, or identity_only")
        return user_repr, history_embeddings

    def _similarity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.similarity == "cosine":
            return torch.sum(
                F.normalize(left, p=2, dim=-1, eps=1e-12)
                * F.normalize(right, p=2, dim=-1, eps=1e-12),
                dim=-1,
            )
        if self.similarity == "dot":
            return torch.sum(left * right, dim=-1)
        raise ValueError("similarity must be cosine or dot")

    def _decay(self, users: torch.Tensor) -> torch.Tensor:
        # Paper/legacy behavior: user-specific positive decay, enforced by clamping.
        return self.raw_decay(users).clamp_min(1e-6)

    def _time_deltas(
        self, target_times: torch.Tensor, history_times: torch.Tensor
    ) -> torch.Tensor:
        base_hours = torch.abs(target_times.unsqueeze(1) - history_times)
        base_hours = base_hours * _TIMESTAMP_TO_HOURS[self.timestamp_unit]
        deltas = [base_hours / _TIME_SCALE_IN_HOURS[g] for g in self.granularities]
        return torch.stack(deltas, dim=1)  # [B, G, L]

    def _score_candidates_from_repr(
        self,
        users: torch.Tensor,
        user_repr: torch.Tensor,
        history_embeddings: torch.Tensor,
        history_times: torch.Tensor,
        history_mask: torch.Tensor,
        target_times: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        user_norm = F.normalize(user_repr, p=2, dim=-1, eps=1e-12)
        history_norm = F.normalize(history_embeddings, p=2, dim=-1, eps=1e-12)
        if candidate_items.dim() == 1:
            candidates = self.item_embedding(candidate_items)  # [C,D]
            candidate_norm = F.normalize(candidates, p=2, dim=-1, eps=1e-12)
            if self.similarity == "cosine":
                mu = torch.matmul(user_norm, candidate_norm.transpose(0, 1))
                alpha = torch.einsum("bld,cd->blc", history_norm, candidate_norm)
            elif self.similarity == "dot":
                mu = torch.matmul(user_repr, candidates.transpose(0, 1))
                alpha = torch.einsum("bld,cd->blc", history_embeddings, candidates)
            else:
                raise ValueError("similarity must be cosine or dot")
        elif candidate_items.dim() == 2:
            candidates = self.item_embedding(candidate_items)  # [B,C,D]
            candidate_norm = F.normalize(candidates, p=2, dim=-1, eps=1e-12)
            if self.similarity == "cosine":
                mu = torch.einsum("bd,bcd->bc", user_norm, candidate_norm)
                alpha = torch.einsum("bld,bcd->blc", history_norm, candidate_norm)
            elif self.similarity == "dot":
                mu = torch.einsum("bd,bcd->bc", user_repr, candidates)
                alpha = torch.einsum("bld,bcd->blc", history_embeddings, candidates)
            else:
                raise ValueError("similarity must be cosine or dot")
        else:
            raise ValueError("candidate_items must have shape [C] or [B,C]")

        if self.positive_intensity:
            mu = F.softplus(mu)
            alpha = F.softplus(alpha)

        deltas = self._time_deltas(target_times, history_times)  # [B,G,L]
        decay = self._decay(users)
        if decay.size(1) == 1:
            decay = decay.expand(-1, len(self.granularities))
        kernel = torch.exp(-decay.unsqueeze(-1) * deltas) * history_mask.unsqueeze(1)
        excitation = torch.einsum("bgl,blc->bgc", kernel, alpha)
        branch_scores = mu.unsqueeze(1) + excitation
        theta = torch.softmax(self.theta_logits, dim=0)
        return torch.einsum("g,bgc->bc", theta, branch_scores)

    def encode_context(
        self,
        users: torch.Tensor,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_user(users, history_items, history_mask, adjacency)

    def score_from_context(
        self,
        users: torch.Tensor,
        user_repr: torch.Tensor,
        history_embeddings: torch.Tensor,
        candidate_items: torch.Tensor,
        target_times: torch.Tensor,
        history_times: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._score_candidates_from_repr(
            users, user_repr, history_embeddings, history_times, history_mask, target_times, candidate_items
        )

    def forward(
        self,
        users: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor,
        target_times: torch.Tensor,
        history_items: torch.Tensor,
        history_times: torch.Tensor,
        history_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_repr, history_embeddings = self.encode_user(users, history_items, history_mask, adjacency)
        positive = self._score_candidates_from_repr(
            users,
            user_repr,
            history_embeddings,
            history_times,
            history_mask,
            target_times,
            target_items.unsqueeze(1),
        ).squeeze(1)
        negative = self._score_candidates_from_repr(
            users,
            user_repr,
            history_embeddings,
            history_times,
            history_mask,
            target_times,
            negative_items,
        )
        return positive, negative

    def score_items(
        self,
        users: torch.Tensor,
        candidate_items: torch.Tensor,
        target_times: torch.Tensor,
        history_items: torch.Tensor,
        history_times: torch.Tensor,
        history_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        user_repr, history_embeddings = self.encode_user(users, history_items, history_mask, adjacency)
        return self._score_candidates_from_repr(
            users,
            user_repr,
            history_embeddings,
            history_times,
            history_mask,
            target_times,
            candidate_items,
        )

    @staticmethod
    def bpr_loss(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return F.softplus(negative - positive.unsqueeze(1)).mean()
