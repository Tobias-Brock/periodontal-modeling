from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from pamod.base import BaseEvaluator
from pamod.resampling import MetricEvaluator, get_probs
from pamod.training import MLPTrainer


class Trainer(BaseEvaluator):
    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: Optional[str],
        hpo: Optional[str],
    ) -> None:
        """Initializes the Trainer with classification type and criterion.

        Args:
            classification (str): The type of classification ('binary' or
                'multiclass').
            criterion (str): The performance criterion to optimize (e.g., 'f1',
                'brier_score').
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (Optional[str]): The hyperparameter optimization method. Can be None.
        """
        super().__init__(classification, criterion, tuning, hpo)
        self.metric_evaluator = MetricEvaluator(self.classification, self.criterion)

    def train(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[float, object, Union[float, None]]:
        """Trains either an MLP model with custom logic or a standard model.

        Args:
            model (sklearn estimator): The machine learning model to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.

        Returns:
            Tuple: The evaluation score, trained model, and the best threshold.
        """
        if isinstance(model, MLPClassifier):
            mlp_trainer = MLPTrainer(
                self.classification, self.criterion, self.tuning, self.hpo
            )
            score, model, best_threshold = mlp_trainer.train(
                model, X_train, y_train, X_val, y_val
            )
        else:
            # For non-MLP models, perform standard training and evaluation
            model.fit(X_train, y_train)
            probs = get_probs(model, self.classification, X_val)
            best_threshold = None

            if self.classification == "binary" and (
                self.tuning == "cv" or self.hpo == "HEBO"
            ):
                score = self.metric_evaluator.evaluate_metric(model, y_val, probs)
            else:
                score, best_threshold = self.metric_evaluator.evaluate(y_val, probs)

        return score, model, best_threshold

    def evaluate_cv(self, model, fold: Tuple) -> float:
        """Evaluates a model on a specific training-validation fold.

        Based on a chosen performance criterion.

        Args:
            model (sklearn estimator): The machine learning model used for
                evaluation.
            fold (tuple): A tuple containing two tuples:
                - The first tuple contains the training data (features and labels).
                - The second tuple contains the validation data (features and labels).
                Specifically, it is structured as ((X_train, y_train), (X_val, y_val)),
                where X_train and X_val are the feature matrices, and y_train and y_val
                are the target vectors.

        Returns:
            float: The calculated score of the model on the validation data according
                to the specified criterion. Higher scores indicate better performance
                for 'f1', while lower scores are better for 'brier_score'.

        Raises:
            ValueError: If an invalid evaluation criterion is specified.
        """
        (X_train, y_train), (X_val, y_val) = fold
        score, _, _ = self.train(model, X_train, y_train, X_val, y_val)

        return score

    def find_optimal_threshold(
        self, true_labels: np.ndarray, probs: np.ndarray
    ) -> Union[float, None]:
        """Find the optimal threshold based on the criterion.

        Converts probabilities into binary decisions.

        Args:
            true_labels (np.ndarray): The true labels for validation or test data.
            probs (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            float or None: The optimal threshold for 'f1', or None if the criterion is
                'brier_score'.
        """
        if self.criterion == "brier_score":
            return None  # Thresholding is not applicable for Brier score

        elif self.criterion == "f1":
            thresholds = np.linspace(0, 1, 101)
            scores = [
                f1_score(true_labels, probs >= th, pos_label=0) for th in thresholds
            ]
            best_threshold = thresholds[np.argmax(scores)]
            print(f"Best threshold: {best_threshold}, Best F1 score: {np.max(scores)}")
            return best_threshold
        raise ValueError(f"Invalid criterion: {self.criterion}")

    def optimize_threshold(
        self,
        model,
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]],
    ) -> Union[float, None]:
        """Optimize the decision threshold using cross-validation.

        Aggregates probability predictions across cross-validation folds.

        Args:
            model (sklearn estimator): The trained machine learning model.
            best_params (dict): The best hyperparameters obtained from optimization.
            outer_splits (List[Tuple]): List of ((X_train, y_train), (X_val, y_val)).

        Returns:
            float or None: The optimal threshold for 'f1', or None if the criterion is
                'brier_score'.
        """
        if outer_splits is None:
            return None
        all_true_labels = []
        all_probs = []

        for (X_train, y_train), (X_val, y_val) in outer_splits:
            _, best_model, _ = self.train(model, X_train, y_train, X_val, y_val)
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(X_val)[:, 1]
            else:
                raise AttributeError(
                    f"The model {type(best_model)} does not support predict_proba."
                )

            all_probs.extend(probs)
            all_true_labels.extend(y_val)

        all_true_labels = np.array(all_true_labels)
        all_probs = np.array(all_probs)

        return self.find_optimal_threshold(all_true_labels, all_probs)
