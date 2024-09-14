import numpy as np
import pandas as pd

from pamod.base import BaseEvaluator
from pamod.resampling import MetricEvaluator


class MLPTrainer(BaseEvaluator):
    def __init__(self, classification: str, criterion: str) -> None:
        """
        Initializes the MLPTrainer with metric evaluator and training parameters.

        Args:
            metric_evaluator (MetricEvaluator): An instance of MetricEvaluator for evaluating metrics.
            tol (float): Tolerance for improvement. Stops training if improvement is less than tol.
            n_iter_no_change (int): Number of iterations with no improvement to wait before stopping.
        """
        super().__init__(classification, criterion)
        self.metric_evaluator = MetricEvaluator(self.classification, self.criterion)

    def train(
        self,
        mlp_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """
        Generalized method for training MLPClassifier with early stopping and evaluation for both binary and multiclass.

        Args:
            mlp_model (MLPClassifier): The MLPClassifier to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            binary (bool): Whether it's binary classification.

        Returns:
            tuple: The best validation score, trained MLPClassifier, and the optimal threshold (None for multiclass).
        """
        best_val_score = -float("inf") if self.criterion in ["f1", "macro_f1"] else float("inf")
        best_threshold = 0.5 if self.classification == "binary" else None
        no_improvement_count = 0

        for _ in range(mlp_model.max_iter):
            mlp_model.partial_fit(X_train, y_train, classes=np.unique(y_train))

            probs = self._get_probabilities(mlp_model, X_val)
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
            best_threshold if self.classification == "binary" else None,
        )

    def _get_probabilities(self, mlp_model, X_val):
        """
        Gets the predicted probabilities from the MLP model.

        Args:
            mlp_model (MLPClassifier): The trained MLPClassifier model.
            X_val (pd.DataFrame): Validation features.
            binary (bool): Whether it's binary classification.

        Returns:
            array-like: Predicted probabilities.
        """
        if self.classification == "binary":
            return mlp_model.predict_proba(X_val)[:, 1]
        else:
            return mlp_model.predict_proba(X_val)

    def _is_improvement(self, score, best_val_score):
        """
        Determines if there is an improvement in the validation score.

        Args:
            score (float): Current validation score.
            best_val_score (float): Best validation score so far.
            criterion (str): Criterion for optimization.

        Returns:
            bool: Whether the current score is an improvement.
        """
        if self.criterion in ["f1", "macro_f1"]:
            return score > best_val_score + self.tol
        else:
            return score < best_val_score - self.tol
