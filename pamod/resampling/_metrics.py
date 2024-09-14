from typing import Tuple, Union

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.preprocessing import label_binarize

from pamod.base import BaseEvaluator


def brier_loss_multi(y_val: np.ndarray, probs: np.ndarray) -> float:
    """
    Calculates multiclass Brier score.

    Args:
        y_val (np.ndarray): True labels for the validation data.
        probs (np.ndarray): Probability predictions for each class. For binary classification, this is the
                            probability for the positive class. For multiclass, it is a 2D array with probabilities.

    Returns:
        float: The calculated multiclass Brier score.
    """
    y_bin = label_binarize(y_val, classes=np.unique(y_val))
    g = y_bin.shape[1]  # number of classes
    return np.mean([brier_score_loss(y_bin[:, i], probs[:, i]) for i in range(g)]) * (g / 2)


class MetricEvaluator(BaseEvaluator):
    def __init__(self, classification: str, criterion: str) -> None:
        """
        Initializes the MetricEvaluator with a classification type.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            criterion (str): The performance criterion to evaluate - 'f1', 'brier_score' for binary or
                             'macro_f1', 'brier_score' for multiclass.
        """
        super().__init__(classification, criterion)

    def evaluate(self, y_val: np.ndarray, probs: np.ndarray) -> Union[Tuple[float, float], float]:
        """
        Evaluates the model performance based on the criterion for binary or multiclass classification.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class. For binary classification, this is the
                                probability for the positive class. For multiclass, it is a 2D array with probabilities.

        Returns:
            tuple or float: The calculated score and the optimal threshold (if applicable for binary classification).
                            For multiclass, only the score is returned.
        """
        if self.classification == "binary":
            return self._evaluate_binary(y_val, probs)
        else:
            return self._evaluate_multiclass(y_val, probs)

    def _evaluate_binary(
        self, y_val: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, Union[float, None]]:
        """
        Evaluates binary classification metrics based on probabilities.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for the positive class.
            criterion (str): The performance criterion to evaluate - 'f1' or 'brier_score'.

        Returns:
            tuple: The calculated score and the optimal threshold (if applicable).
        """
        if self.criterion == "f1":
            # Calculate F1 score for a range of thresholds to find the best one
            scores, thresholds = [], np.linspace(0, 1, 101)
            for threshold in thresholds:
                preds = (probs >= threshold).astype(int)
                scores.append(f1_score(y_val, preds, pos_label=0))
            best_idx = np.argmax(scores)
            return scores[best_idx], thresholds[best_idx]
        elif self.criterion == "brier_score":
            # Direct calculation of Brier score
            return brier_score_loss(y_val, probs), None

    def _evaluate_multiclass(self, y_val: np.ndarray, probs: np.ndarray) -> float:
        """
        Evaluates multiclass classification metrics based on probabilities.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class (2D array).

        Returns:
            float: The calculated score based on the specified criterion.
        """
        # Convert probabilities to predicted labels (choosing class with highest probability)
        preds = np.argmax(probs, axis=1)

        if self.criterion == "macro_f1":
            return f1_score(y_val, preds, average="macro"), None
        elif self.criterion == "brier_score":
            return brier_loss_multi(y_val, probs), None

    def evaluate_metric(self, model, y_val: np.ndarray, preds: np.ndarray) -> float:
        """
        Evaluates the performance of model predictions against cross-validation data based on a specified criterion.

        Args:
            model (sklearn estimator): The machine learning model used for evaluation.
            y_val (np.ndarray): True labels for the cross-validation data.
            probs (np.ndarray): Probability predictions from the model for the positive class or direct predictions
                                depending on the model's capabilities and the specified criterion.

        Returns:
            float: The calculated score based on the specified criterion.

        Raises:
            ValueError: If an invalid criterion is specified or if the model does not support probability estimates
                        required for the Brier score evaluation.
        """
        if self.criterion == "f1":
            preds = preds if not hasattr(model, "predict_proba") else (preds > 0.5).astype(int)
            return f1_score(y_val, preds, pos_label=0)
        elif self.criterion == "macro_f1":
            return f1_score(y_val, preds, average="macro")
        elif self.criterion == "brier_score":
            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "Model does not support probability estimates required for Brier score evaluation."
                )
            if self.classification == "binary":
                return brier_score_loss(y_val, preds)
            elif self.classification == "multiclass":
                return brier_loss_multi(y_val, preds)
