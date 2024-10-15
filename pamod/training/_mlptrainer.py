from typing import Optional

import numpy as np
import pandas as pd

from pamod.base import BaseEvaluator
from pamod.training import MetricEvaluator, get_probs


class MLPTrainer(BaseEvaluator):
    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: Optional[str],
        hpo: Optional[str],
        metric_evaluator: Optional[MetricEvaluator] = None,
    ) -> None:
        """Initializes the MLPTrainer with training parameters.

        Args:
            classification (str): The type of classification ('binary' or
                'multiclass').
            criterion (str): The performance criterion to optimize (e.g., 'f1',
                'brier_score').
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (str): The hyperparameter optimization method (default is 'HEBO').
            metric_evaluator (Optional[MetricEvaluator]): Instance of MetricEvaluator.
                If None, a default instance will be created.
        """
        super().__init__(classification, criterion, tuning, hpo)
        self.metric_evaluator = (
            metric_evaluator
            if metric_evaluator
            else MetricEvaluator(self.classification, self.criterion)
        )

    def train(
        self,
        mlp_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        final: bool = False,
    ):
        """General method for training MLPClassifier with early stopping.

        Applies evaluation for both binary and multiclass classification.

        Args:
            mlp_model (MLPClassifier): The MLPClassifier to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            final (bool): Flag for final model training.

        Returns:
            tuple: The best validation score, trained MLPClassifier, and the
                optimal threshold (None for multiclass).
        """
        best_val_score = (
            -float("inf") if self.criterion in ["f1", "macro_f1"] else float("inf")
        )
        best_threshold = None
        no_improvement_count = 0

        for _ in range(mlp_model.max_iter):
            mlp_model.partial_fit(X_train, y_train, classes=np.unique(y_train))

            probs = get_probs(mlp_model, self.classification, X_val)
            if self.classification == "binary":
                if final or (self.tuning == "cv" or self.hpo == "hebo"):
                    score = self.metric_evaluator.evaluate_metric(
                        mlp_model, y_val, probs
                    )
            else:
                score, best_threshold = self.metric_evaluator.evaluate(y_val, probs)

            if self._is_improvement(score, best_val_score):
                best_val_score = score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.n_iter_no_change:
                break  # Stop early

        return (
            best_val_score,
            mlp_model,
            best_threshold,
        )

    def _is_improvement(self, score, best_val_score):
        """Determines if there is an improvement in the validation score.

        Args:
            score (float): Current validation score.
            best_val_score (float): Best validation score so far.

        Returns:
            bool: Whether the current score is an improvement.
        """
        if self.criterion in ["f1", "macro_f1"]:
            return score > best_val_score + self.tol
        else:
            return score < best_val_score - self.tol
