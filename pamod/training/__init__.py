"""Module provides training methods."""

from pamod.training._metrics import (
    MetricEvaluator,
    brier_loss_multi,
    final_metrics,
    get_probs,
)
from pamod.training._mlptrainer import MLPTrainer
from pamod.training._trainer import Trainer

__all__ = [
    "brier_loss_multi",
    "get_probs",
    "final_metrics",
    "MetricEvaluator",
    "MLPTrainer",
    "Trainer",
]
