from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np

from pamod.base import BaseEvaluator
from pamod.resampling import MetricEvaluator
from pamod.training import Trainer


class MetaTuner(ABC):
    """Abstract base class enforcing implementation of tuning strategies."""

    @abstractmethod
    def cv(self, *args, **kwargs):
        """Perform cross-validation based tuning."""
        pass

    @abstractmethod
    def holdout(self, *args, **kwargs):
        """Perform holdout based tuning."""
        pass


class BaseTuner(BaseEvaluator):
    """Base class for different hyperparameter tuning strategies."""

    def __init__(
        self, classification: str, criterion: str, tuning: str, hpo: str
    ) -> None:
        """Initializes the base tuner class with common parameters.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            criterion (str): The evaluation criterion (e.g., 'f1', 'brier_score').
            tuning (str): The type of tuning ('holdout' or 'cv').
            hpo (str): The hyperparameter optimization method.
        """
        super().__init__(classification, criterion, tuning, hpo)
        self.trainer = Trainer(
            self.classification, self.criterion, self.tuning, self.hpo
        )
        self.metric_evaluator = MetricEvaluator(self.classification, self.criterion)

    def _print_iteration_info(
        self,
        iteration: int,
        model,
        params_dict: Dict[str, Union[float, int]],
        score: float,
        threshold: Optional[float] = None,
    ) -> None:
        """Common method for printing iteration info during tuning.

        Args:
            iteration (int): The current iteration index.
            model: The machine learning model being evaluated.
            params_dict (Dict[str, Union[float, int]]): The suggested hyperparameters
                as a dictionary.
            score (float): The score achieved in the current iteration.
            threshold (Optional[float]): The threshold if applicable
                (for binary classification).
        """
        model_name = model.__class__.__name__
        params_str = ", ".join(
            [
                (
                    f"{key}={value:.4f}"
                    if isinstance(value, (int, float))
                    else f"{key}={value}"
                )
                for key, value in params_dict.items()
            ]
        )
        score_value = (
            f"{score:.4f}"
            if np.isscalar(score) and isinstance(score, (int, float))
            else None
        )

        if self.tuning == "holdout":
            print(
                f"{self.hpo} val_split iteration {iteration + 1} {model_name}: "
                f"'{params_str}', {self.criterion}={score_value}, "
                f"threshold={threshold}"
            )
        elif self.tuning == "cv":
            print(
                f"{self.hpo} CV iteration {iteration + 1} {model_name}: "
                f"'{params_str}', {self.criterion}={score_value}"
            )
