from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shcn import StructuralHypergraphConvolutionNetwork


_TIME_SCALE_IN_HOURS = {"hour": 1.0, "day": 24.0, "week": 168.0}
_TIMESTAMP_TO_HOURS = {"second": 1.0 / 3600.0, "minute": 1.0 / 60.0, "hour": 1.0, "day": 24.0}


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value)) if value < 20.0 else value


class MTHPHC(nn.Module):
    """Unified MTHP-HC implementation for reproduction and ASOC extensions."""

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
        self.decay_parameterization = config.get("decay_parameterization", "softplus")
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
            output_projection=bool(config.get("shcn_output_projection", False)),
        )

        decay_init = float(config["decay_init"])
        decay_dim = 1 if self.decay_mode == "shared_user" else len(self.granularities)
        if self.decay_mode not in {"shared_user", "user_granularity"}:
            raise ValueError("decay_mode must be shared_user or user_granularity")
        self.raw_decay = nn.Embedding(num_users, decay_dim)
        if self.decay_parameterization == "softplus":
            nn.init.constant_(self.raw_decay.weight, _inverse_softplus(decay_init))
        elif self.decay_parameterization == "clamp":
            nn.init.constant_(self.raw_decay.weight, decay_init)
        else:
            raise ValueError("decay_parameterization must be softplus or clamp")

        if self.granularity_weights == "learnable_global":
            self.theta_logits = nn.Parameter(torch.zeros(len(self.granularities)))
            self.theta_network = None
        elif self.granularity_weights == "uniform":
            self.register_buffer("theta_logits", torch.zeros(len(self.granularities)))
            self.theta_network = None
        elif self.granularity_weights == "user_adaptive":
            hidden = int(config.get("granularity_hidden_dim", dim))
            self.theta_logits = None
            self.theta_network = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, len(self.granularities)),
            )
        else:
            raise ValueError(
                "granularity_weights must be learnable_global, uniform, or user_adaptive"
            )

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
        has_history = history_mask.sum(dim=1, keepdim=True) > 0

        if self.user_fusion == "add":
            user_repr = F.normalize(identity + structural, p=2, dim=-1, eps=1e-12)
        elif self.user_fusion == "structural_only":
            user_repr = torch.where(has_history, structural, identity)
        elif self.user_fusion == "identity_only":
            user_repr = identity
        else:
            raise ValueError("user_fusion must be add, structural_only, or identity_only")
        return user_repr, history_embeddings

    def _decay(self, users: torch.Tensor) -> torch.Tensor:
        raw = self.raw_decay(users)
        if self.decay_parameterization == "softplus":
            return F.softplus(raw) + 1e-8
        return raw.clamp_min(1e-6)

    def _time_deltas(
        self, target_times: torch.Tensor, history_times: torch.Tensor
    ) -> torch.Tensor:
        base_hours = torch.abs(target_times.unsqueeze(1) - history_times)
        base_hours = base_hours * _TIMESTAMP_TO_HOURS[self.timestamp_unit]
        deltas = [base_hours / _TIME_SCALE_IN_HOURS[g] for g in self.granularities]
        return torch.stack(deltas, dim=1)

    def _theta(self, user_repr: torch.Tensor) -> torch.Tensor:
        if self.granularity_weights in {"learnable_global", "uniform"}:
            return torch.softmax(self.theta_logits, dim=0)
        if self.theta_network is None:
            raise RuntimeError("user-adaptive theta network is not initialized")
        return torch.softmax(self.theta_network(user_repr), dim=-1)

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
            candidates = self.item_embedding(candidate_items)
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
            candidates = self.item_embedding(candidate_items)
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

        deltas = self._time_deltas(target_times, history_times)
        decay = self._decay(users)
        if decay.size(1) == 1:
            decay = decay.expand(-1, len(self.granularities))
        kernel = torch.exp(-decay.unsqueeze(-1) * deltas) * history_mask.unsqueeze(1)
        excitation = torch.einsum("bgl,blc->bgc", kernel, alpha)
        branch_scores = mu.unsqueeze(1) + excitation
        theta = self._theta(user_repr)
        if theta.dim() == 1:
            return torch.einsum("g,bgc->bc", theta, branch_scores)
        return torch.einsum("bg,bgc->bc", theta, branch_scores)

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
            users, user_repr, history_embeddings, history_times,
            history_mask, target_times, candidate_items,
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
            users, user_repr, history_embeddings, history_times, history_mask,
            target_times, target_items.unsqueeze(1),
        ).squeeze(1)
        negative = self._score_candidates_from_repr(
            users, user_repr, history_embeddings, history_times, history_mask,
            target_times, negative_items,
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
            users, user_repr, history_embeddings, history_times, history_mask,
            target_times, candidate_items,
        )

    @staticmethod
    def bpr_loss(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return F.softplus(negative - positive.unsqueeze(1)).mean()
