from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np

from pamod.base import BaseEvaluator
from pamod.training import MetricEvaluator, Trainer


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
        self,
        classification: str,
        criterion: str,
        tuning: str,
        hpo: str,
        n_configs: int = 10,
        n_jobs: Optional[int] = None,
        verbosity: bool = True,
        trainer: Optional[Trainer] = None,
        metric_evaluator: Optional[MetricEvaluator] = None,
        mlp_training: bool = True,
    ) -> None:
        """Initializes the base tuner class with common parameters.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            criterion (str): The evaluation criterion (e.g., 'f1', 'brier_score').
            tuning (str): The type of tuning ('holdout' or 'cv').
            hpo (str): The hyperparameter optimization method.
            n_configs (int): The number of configurations to evaluate during HPO.
            n_jobs (Optional[int]): The number of parallel jobs for model training.
            verbosity (bool): Whether to print detailed logs during optimization.
            trainer (Optional[Trainer]): Instance of Trainer class.
            metric_evaluator (Optional[MetricEvaluator]): Instance of MetricEvaluator.
            mlp_training (bool): Flag for MLP training with early stopping.
        """
        super().__init__(classification, criterion, tuning, hpo)
        self.n_configs = n_configs
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.verbosity = verbosity
        self.mlp_training = mlp_training

        self.metric_evaluator = (
            metric_evaluator
            if metric_evaluator
            else MetricEvaluator(self.classification, self.criterion)
        )
        self.trainer = (
            trainer
            if trainer
            else Trainer(
                self.classification,
                self.criterion,
                self.tuning,
                self.hpo,
                self.mlp_training,
                self.metric_evaluator,
            )
        )

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
                f"{self.hpo} holdout iteration {iteration + 1} {model_name}: "
                f"'{params_str}', {self.criterion}={score_value}, "
                f"threshold={threshold}"
            )
        elif self.tuning == "cv":
            print(
                f"{self.hpo} CV iteration {iteration + 1} {model_name}: "
                f"'{params_str}', {self.criterion}={score_value}"
            )
