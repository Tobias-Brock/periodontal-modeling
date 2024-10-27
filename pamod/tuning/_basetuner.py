from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..base import BaseEvaluator
from ..training import Trainer


class BaseTuner(BaseEvaluator, ABC):
    """Base class for different hyperparameter tuning strategies."""

    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: str,
        hpo: str,
        n_configs: int,
        n_jobs: Optional[int],
        verbosity: bool,
        trainer: Optional[Trainer],
        mlp_training: bool,
        threshold_tuning: bool,
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
            mlp_training (bool): Flag for MLP training with early stopping.
            threshold_tuning (bool): Perform threshold tuning for binary classification
                if the criterion is "f1".
        """
        super().__init__(
            classification=classification, criterion=criterion, tuning=tuning, hpo=hpo
        )
        self.n_configs = n_configs
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.verbosity = verbosity
        self.mlp_training = mlp_training
        self.threshold_tuning = threshold_tuning
        self.trainer = (
            trainer
            if trainer
            else Trainer(
                classification=self.classification,
                criterion=self.criterion,
                tuning=self.tuning,
                hpo=self.hpo,
                mlp_training=self.mlp_training,
                threshold_tuning=self.threshold_tuning,
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

    @abstractmethod
    def cv(
        self,
        learner: str,
        outer_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        racing_folds: Optional[int],
    ):
        """Perform cross-validation with optional tuning.

        Args:
            learner (str): The model to evaluate.
            outer_splits (List[Tuple[pd.DataFrame, pd.DataFrame]]): Train/validation
                splits.
            racing_folds (Optional[int]): Number of racing folds; if None regular
                cross-validation is performed.
        """

    @abstractmethod
    def holdout(
        self,
        learner: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """Perform random search on the holdout set for binary and multiclass .

        Args:
            learner (str): The machine learning model used for evaluation.
            X_train (pd.DataFrame): Training features for the holdout set.
            y_train (pd.Series): Training labels for the holdout set.
            X_val (pd.DataFrame): Validation features for the holdout set.
            y_val (pd.Series): Validation labels for the holdout set.
        """
