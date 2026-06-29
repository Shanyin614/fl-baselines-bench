"""Federated client abstraction shared by all baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.flbench.core.types import ClientMeta
from src.flbench.utils.state_dict import StateDict, clone_state, state_delta

SplitName = Literal["train", "val", "test"]


@dataclass
class LocalTrainResult:
    new_state: StateDict
    delta: StateDict
    num_samples: int
    train_loss: float


class FLClient:
    """One federated participant with train/val/test subsets."""

    def __init__(
        self,
        meta: ClientMeta,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        device: torch.device,
    ) -> None:
        self.id = int(meta.client_id)
        self.group_id = None if meta.group_id is None else int(meta.group_id)
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_set = Subset(train_dataset, list(meta.train_indices))
        self.val_set = Subset(train_dataset, list(meta.val_indices))
        self.test_set = Subset(test_dataset, list(meta.test_indices))
        self.num_train = len(self.train_set)
        self.num_val = len(self.val_set)
        self.num_test = len(self.test_set)

    def _loader(self, split: SplitName, shuffle: bool = False) -> DataLoader:
        if split == "train":
            dataset = self.train_set
        elif split == "val":
            dataset = self.val_set
        elif split == "test":
            dataset = self.test_set
        else:
            raise ValueError(f"unknown split: {split}")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    def train(
        self,
        model_state: StateDict,
        model_fn: Callable[[], nn.Module],
        epochs: int,
        lr: float,
        optimizer_name: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        proximal_mu: float = 0.0,
    ) -> LocalTrainResult:
        """Train locally from a provided model state and return new_state and delta.

        `proximal_mu > 0` turns this into a FedProx-style local objective, but
        FeSEM and FedAvg use the default `0.0`.
        """
        base_state = clone_state(model_state)
        model = model_fn().to(self.device)
        model.load_state_dict(base_state)
        model.train()

        if optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"unsupported optimizer: {optimizer_name}")

        # Keep a device copy of the reference model for an optional FedProx penalty.
        prox_ref = {k: v.to(self.device) for k, v in base_state.items()} if proximal_mu > 0 else None

        total_loss = 0.0
        total_n = 0
        for _ in range(int(epochs)):
            for x, y in self._loader("train", shuffle=True):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                if proximal_mu > 0 and prox_ref is not None:
                    prox_term = torch.zeros((), device=self.device)
                    for name, param in model.named_parameters():
                        prox_term = prox_term + torch.sum((param - prox_ref[name]) ** 2)
                    loss = loss + 0.5 * proximal_mu * prox_term
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * int(y.numel())
                total_n += int(y.numel())

        new_state = clone_state(model.state_dict())
        delta = state_delta(new_state, base_state)
        return LocalTrainResult(
            new_state=new_state,
            delta=delta,
            num_samples=self.num_train,
            train_loss=total_loss / max(total_n, 1),
        )

    @torch.no_grad()
    def loss_on_model(
        self,
        model_state: StateDict,
        model_fn: Callable[[], nn.Module],
        split: SplitName = "val",
    ) -> float:
        model = model_fn().to(self.device)
        model.load_state_dict(clone_state(model_state))
        model.eval()
        total_loss = 0.0
        total_n = 0
        for x, y in self._loader(split, shuffle=False):
            x = x.to(self.device)
            y = y.to(self.device)
            total_loss += float(F.cross_entropy(model(x), y, reduction="sum").cpu())
            total_n += int(y.numel())
        return total_loss / max(total_n, 1)

    @torch.no_grad()
    def evaluate_state(
        self,
        model_state: StateDict,
        model_fn: Callable[[], nn.Module],
        split: SplitName = "test",
    ) -> dict[str, object]:
        model = model_fn().to(self.device)
        model.load_state_dict(clone_state(model_state))
        model.eval()
        total_loss = 0.0
        total_n = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        for x, y in self._loader(split, shuffle=False):
            x = x.to(self.device)
            y_dev = y.to(self.device)
            logits = model(x)
            total_loss += float(F.cross_entropy(logits, y_dev, reduction="sum").cpu())
            total_n += int(y.numel())
            pred = logits.argmax(1).detach().cpu().numpy().astype(int).tolist()
            y_true.extend(y.numpy().astype(int).tolist())
            y_pred.extend(pred)
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_pred_np = np.asarray(y_pred, dtype=np.int64)
        correct = int((y_true_np == y_pred_np).sum())
        return {
            "client_id": self.id,
            "group_id": self.group_id,
            "loss": total_loss / max(total_n, 1),
            "acc": correct / max(total_n, 1),
            "num_samples": total_n,
            "y_true": y_true_np,
            "y_pred": y_pred_np,
        }
