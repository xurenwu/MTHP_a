from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable if iterable is not None else range(0)

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **kwargs):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

from mthp.data import ItemGraphStore, SequenceDataset, collate_sequence_batch
from mthp.models import MTHPHC
from mthp.training.metrics import rank_metrics
from mthp.utils import ensure_dir, save_json


class ExperimentRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        model: MTHPHC,
        graph_store: ItemGraphStore,
        train_dataset: SequenceDataset,
        valid_dataset: SequenceDataset,
        test_dataset: SequenceDataset,
        new_test_dataset: Optional[SequenceDataset],
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        self.cfg = config
        self.model = model.to(device)
        self.graph_store = graph_store
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.new_test_dataset = new_test_dataset
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        train_cfg = config["train"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda")

    def _loader(self, dataset: SequenceDataset, train: bool) -> DataLoader:
        batch_size = self.cfg["train"]["batch_size" if train else "eval_batch_size"]
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=train,
            num_workers=int(self.cfg["train"]["num_workers"]),
            collate_fn=collate_sequence_batch,
            pin_memory=self.device.type == "cuda",
        )

    def _move(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        adjacency = self.graph_store.batch_slice(batch["history_items"], batch["history_mask"])
        moved = {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        moved["adjacency"] = adjacency.to(self.device, non_blocking=True)
        return moved

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        amp_enabled = bool(self.cfg["train"]["amp"]) and self.device.type == "cuda"
        loader = self._loader(self.train_dataset, train=True)
        progress = tqdm(
            loader,
            total=len(loader),
            desc="train",
            leave=False,
            dynamic_ncols=True,
        )
        for step, raw in enumerate(progress, start=1):
            batch = self._move(raw)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                positive, negative = self.model(
                    users=batch["user"],
                    target_items=batch["target_item"],
                    negative_items=batch["negative_items"],
                    target_times=batch["target_time"],
                    history_items=batch["history_items"],
                    history_times=batch["history_times"],
                    history_mask=batch["history_mask"],
                    adjacency=batch["adjacency"],
                )
                loss = self.model.bpr_loss(positive, negative)
            self.scaler.scale(loss).backward()
            clip = float(self.cfg["train"]["gradient_clip"])
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = int(batch["user"].size(0))
            total_loss += float(loss.detach().cpu()) * batch_size
            count += batch_size
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", samples=count)
        return total_loss / max(count, 1)

    @torch.no_grad()
    def evaluate(self, dataset: SequenceDataset) -> Dict[str, float]:
        self.model.eval()
        ranks: List[int] = []
        item_chunk = int(self.cfg["evaluation"]["item_chunk_size"])
        mask_seen = bool(self.cfg["evaluation"]["mask_seen_items"])
        for raw in self._loader(dataset, train=False):
            batch = self._move(raw)
            user_repr, history_embeddings = self.model.encode_context(
                batch["user"], batch["history_items"], batch["history_mask"], batch["adjacency"]
            )
            score_chunks: List[torch.Tensor] = []
            for start in range(0, self.model.num_items, item_chunk):
                candidates = torch.arange(
                    start, min(start + item_chunk, self.model.num_items), device=self.device
                )
                scores = self.model.score_from_context(
                    users=batch["user"],
                    user_repr=user_repr,
                    history_embeddings=history_embeddings,
                    candidate_items=candidates,
                    target_times=batch["target_time"],
                    history_times=batch["history_times"],
                    history_mask=batch["history_mask"],
                )
                score_chunks.append(scores.cpu())
            all_scores = torch.cat(score_chunks, dim=1)
            if mask_seen:
                for row, seen in enumerate(raw["seen_items"]):
                    target = int(raw["target_item"][row])
                    target_score = all_scores[row, target].clone()
                    if seen:
                        all_scores[row, list(seen)] = -torch.inf
                    all_scores[row, target] = target_score
            targets = raw["target_item"].long()
            target_scores = all_scores.gather(1, targets.unsqueeze(1))
            batch_ranks = 1 + (all_scores > target_scores).sum(dim=1)
            ranks.extend(batch_ranks.tolist())
        return rank_metrics(ranks, self.cfg["evaluation"]["ks"])

    def fit(self) -> Dict[str, Any]:
        train_cfg = self.cfg["train"]
        monitor = str(train_cfg["monitor"])
        patience = int(train_cfg["patience"])
        best_value = -float("inf")
        best_epoch = -1
        stale = 0
        history: List[Dict[str, Any]] = []
        checkpoint = self.output_dir / "best.pt"
        epochs = int(train_cfg["epochs"])

        logging.info("Starting training for %d epochs", epochs)
        epoch_progress = tqdm(range(1, epochs + 1), desc="epoch", dynamic_ncols=True)

        for epoch in epoch_progress:
            started = time.time()
            logging.info("Epoch %d/%d started", epoch, epochs)
            loss = self.train_epoch()
            logging.info("Epoch %d/%d finished training, running validation", epoch, epochs)
            valid = self.evaluate(self.valid_dataset) if len(self.valid_dataset) else {}
            value = valid.get(monitor, -loss)
            record = {"epoch": epoch, "loss": loss, "seconds": time.time() - started, **valid}
            history.append(record)
            log_message = f"epoch={epoch} loss={loss:.6f} valid={valid}"
            logging.info(log_message)
            epoch_progress.set_postfix(loss=f"{loss:.4f}", best=f"{best_value:.4f}", monitor=f"{value:.4f}")
            if value > best_value:
                best_value = value
                best_epoch = epoch
                stale = 0
                torch.save(
                    {"model": self.model.state_dict(), "epoch": epoch, "config": self.cfg},
                    checkpoint,
                )
            else:
                stale += 1
                if patience > 0 and stale >= patience:
                    logging.info("Early stopping at epoch %d", epoch)
                    break

        epoch_progress.close()

        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
        test = self.evaluate(self.test_dataset)
        new_test = self.evaluate(self.new_test_dataset) if self.new_test_dataset and len(self.new_test_dataset) else None
        result = {
            "best_epoch": best_epoch,
            "best_validation": best_value,
            "test": test,
            "new_item_test": new_test,
            "history": history,
        }
        save_json(result, self.output_dir / "results.json")
        return result
