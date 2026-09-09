"""The training loop both NarraBERT models share.

The two models genuinely differ — one is a nine-output regression head, the
other two binary heads over marked spans — and so do their losses, metrics, and
early-stopping criteria. What they share is the mechanics: the optimizer and
warmup schedule, the per-epoch step, and the patience loop around them. Those
live here; everything model-specific is passed in.
"""

import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_annotation.io import REPO_ROOT  # noqa: F401  — re-exported for the trainers


def select_device() -> torch.device:
    """CUDA, then Apple Silicon, then CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def make_optimizer_scheduler(model: nn.Module, steps_per_epoch: int, n_epochs: int,
                             lr: float, weight_decay: float):
    """AdamW with linear decay after a 10% warmup."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = steps_per_epoch * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, scheduler, device,
                loss_fn: Callable) -> float:
    """One pass over `loader`. `loss_fn(outputs, labels)` supplies the objective."""
    model.train()
    total = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(ids, mask), labs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def train_with_early_stopping(
    build_model: Callable[[], nn.Module],
    train_ds: Dataset,
    val_ds: Dataset,
    device,
    *,
    loss_fn: Callable,
    score_fn: Callable[[nn.Module, DataLoader], tuple[float, str]],
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, int]:
    """Train until `score_fn` stops improving, then restore the best weights.

    `score_fn` returns `(criterion, message)`; lower criterion is better, and
    the message is printed alongside the training loss. Which criterion to use
    is the caller's decision — the Likert model early-stops on validation MAE,
    the event model on validation loss.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model().to(device)
    optimizer, scheduler = make_optimizer_scheduler(
        model, len(train_loader), max_epochs, lr, weight_decay
    )

    best, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
        criterion, message = score_fn(model, val_loader)
        print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  {message}")

        if criterion < best:
            best, best_epoch, waited = criterion, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  early stop at epoch {epoch}  (best epoch {best_epoch}, {best:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def train_fixed_epochs(
    build_model: Callable[[], nn.Module],
    train_ds: Dataset,
    device,
    *,
    loss_fn: Callable,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> nn.Module:
    """Train on everything for a fixed number of epochs, with no held-out set.

    Used for the final model after cross-validation has settled the epoch count.
    """
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = build_model().to(device)
    optimizer, scheduler = make_optimizer_scheduler(
        model, len(loader), n_epochs, lr, weight_decay
    )
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(model, loader, optimizer, scheduler, device, loss_fn)
        print(f"  epoch {epoch:2d}  train_loss={loss:.4f}")
    return model
