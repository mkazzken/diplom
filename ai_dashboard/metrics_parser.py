"""Normalize metrics.json / history.json from Stefan experiments."""
from __future__ import annotations

from typing import Any


def flat_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key.lower(): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def training_losses(history: dict[str, Any]) -> tuple[list[float], list[float]]:
    train = list(history.get('train_loss') or [])
    val = list(history.get('val_loss') or [])
    return train, val
