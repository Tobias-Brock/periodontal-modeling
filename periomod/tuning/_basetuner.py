from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..base import BaseValidator
from ..training import Trainer


class BaseTuner(BaseValidator, ABC):
    """Base class for implementing hyperparameter tuning strategies.

    This class provides a framework for various hyperparameter optimization
    (HPO) strategies, supporting cross-validation (CV) and holdout tuning
    with options for binary and multiclass classification. Subclasses are
    expected to implement specific tuning methods, including holdout and CV
    procedures, while inheriting shared parameters, evaluation, and iteration
    logging functions.

    Inherits:
        - `BaseValidator`: Validates instance-level variables.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        classification (str): The type of classification ('binary' or 'multiclass').
        criterion (str): The evaluation criterion (e.g., 'f1', 'brier_score').
        tuning (str): The tuning type ('holdout' or 'cv').
        hpo (str): The hyperparameter optimization method (e.g., 'random_search').
        n_configs (int): Number of configurations to evaluate during HPO.
        n_jobs (int): Number of parallel jobs for model training.
        verbose (bool): Enables detailed logs during tuning if True.
        trainer (Optional[Trainer]): Trainer instance for evaluation.
        mlp_training (bool): Enables MLP training with early stopping.
        threshold_tuning (bool): Performs threshold tuning for binary
            classification if criterion is 'f1'.

    Attributes:
        classification (str): Type of classification ('binary' or 'multiclass').
        criterion (str): The performance criterion for optimization
            (e.g., 'f1', 'brier_score').
        tuning (str): Indicates the tuning approach ('holdout' or 'cv').
        hpo (str): Hyperparameter optimization method (e.g., 'random_search').
        n_configs (int): Number of configurations for HPO.
        n_jobs (int): Number of parallel jobs for evaluation.
        verbose (bool): Enables logs during tuning if True.
        mlp_training (bool): Flag to enable MLP training with early stopping.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        trainer (Trainer): Trainer instance to handle model training and evaluation.

    Abstract Methods:
        - `cv`: Defines cross-validation strategy with or without tuning.
        - `holdout`: Implements holdout tuning on a validation set for selected
          hyperparameter configurations.
    """

    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: str,
        hpo: str,
        n_configs: int,
        n_jobs: int,
        verbose: bool,
        trainer: Optional[Trainer],
        mlp_training: bool,
        threshold_tuning: bool,
    ) -> None:
        """Initializes the base tuner class with common parameters."""
        super().__init__(
            classification=classification, criterion=criterion, tuning=tuning, hpo=hpo
        )
        self.n_configs = n_configs
        self.n_jobs = n_jobs
        self.verbose = verbose
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
