from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from pamod.base import BaseEvaluator


def get_probs(model, classification: str, X_val: pd.DataFrame) -> np.ndarray:
    """Gets the predicted probabilities from the model.

    Args:
        model: The trained model.
        classification (str): The type of classification.
        X_val (pd.DataFrame): Validation features.

    Returns:
        array-like: Predicted probabilities.
    """
    if classification == "binary":
        return model.predict_proba(X_val)[:, 1]
    else:
        return model.predict_proba(X_val)


def brier_loss_multi(y_val: np.ndarray, probs: np.ndarray) -> float:
    """Calculates the multiclass Brier score.

    Args:
        y_val (np.ndarray): True labels for the validation data.
        probs (np.ndarray): Probability predictions for each class.
            For binary classification, this is the probability for the positive class.
            For multiclass, it is a 2D array with probabilities.

    Returns:
        float: The calculated multiclass Brier score.
    """
    y_bin = label_binarize(y_val, classes=np.unique(y_val))
    g = y_bin.shape[1]  # number of classes
    return np.mean([brier_score_loss(y_bin[:, i], probs[:, i]) for i in range(g)]) * (
        g / 2
    )


def final_metrics(
    classification: str,
    y_test: np.ndarray,
    preds: np.ndarray,
    probs: Union[np.ndarray, None],
    threshold: float,
) -> Dict[str, Any]:
    """Calculate final metrics for binary or multiclass classification.

    Args:
        classification (str): The type of classification.
        y_test (np.ndarray): Ground truth (actual) labels.
        preds (np.ndarray): Predicted labels from the model.
        probs (Union[np.ndarray, None]): Predicted probabilities from model.
            Only used for binary classification and if available.
        threshold (float): Best threshold used for binary classification.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    if classification == "binary":
        f1: float = f1_score(y_test, preds, pos_label=0)
        precision: float = precision_score(y_test, preds, pos_label=0)
        recall: float = recall_score(y_test, preds, pos_label=0)
        accuracy: float = accuracy_score(y_test, preds)
        brier_score_value: Union[float, None] = (
            brier_score_loss(y_test, probs) if probs is not None else None
        )
        roc_auc_value: Union[float, None] = (
            roc_auc_score(y_test, probs) if probs is not None else None
        )
        conf_matrix: np.ndarray = confusion_matrix(y_test, preds)

        return {
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "Brier Score": brier_score_value,
            "ROC AUC Score": roc_auc_value,
            "Confusion Matrix": conf_matrix,
            "Best Threshold": threshold,
        }

    elif classification == "multiclass":
        brier_score: float = brier_loss_multi(y_test, probs)

        return {
            "Macro F1": f1_score(y_test, preds, average="macro"),
            "Accuracy": accuracy_score(y_test, preds),
            "Class F1 Scores": f1_score(y_test, preds, average=None),
            "Multiclass Brier Score": brier_score,
        }

    raise ValueError(f"Unsupported classification type: {classification}")


class MetricEvaluator(BaseEvaluator):
    def __init__(self, classification: str, criterion: str) -> None:
        """Initializes the MetricEvaluator with a classification type.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            criterion (str): The performance criterion to evaluate.

                Options are:
                - For binary: 'f1', 'brier_score'.
                - For multiclass: 'macro_f1', 'brier_score'.
        """
        super().__init__(classification, criterion)

    def evaluate(
        self, y_val: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Evaluates model performance based on the classification criterion.

        For binary or multiclass classification.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class.
                For binary classification, the probability for the positive class.
                For multiclass, a 2D array with probabilities.

        Returns:
            Union[Tuple[float, float], float]: The calculated score and the optimal
                threshold (if applicable for binary classification).
                For multiclass, only the score is returned.
        """
        if self.classification == "binary":
            return self._evaluate_binary(y_val, probs)
        else:
            return self._evaluate_multiclass(y_val, probs)

    def _evaluate_binary(
        self, y_val: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Evaluates binary classification metrics based on probabilities.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for the positive class.

        Returns:
            Tuple[float, Union[float, None]]: The calculated score and the optimal
                threshold (if applicable).
        """
        if self.criterion == "f1":
            scores, thresholds = [], np.linspace(0, 1, 101)
            for threshold in thresholds:
                preds = (probs >= threshold).astype(int)
                scores.append(f1_score(y_val, preds, pos_label=0))
            best_idx = np.argmax(scores)
            return scores[best_idx], thresholds[best_idx]
        else:
            return brier_score_loss(y_val, probs)

    def _evaluate_multiclass(
        self, y_val: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Evaluates multiclass classification metrics based on probabilities.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class (2D array).

        Returns:
            float: The calculated score.
        """
        preds = np.argmax(probs, axis=1)

        if self.criterion == "macro_f1":
            return f1_score(y_val, preds, average="macro"), None
        else:
            return brier_loss_multi(y_val, probs), None

    def evaluate_metric(self, model, y_val: np.ndarray, probs: np.ndarray) -> float:
        """Evaluates the model's performance against cross-validation data.

        Based on a specified criterion.

        Args:
            model (sklearn estimator): The machine learning model used for evaluation.
            y_val (np.ndarray): True labels for the cross-validation data.
            probs (np.ndarray): Model probabilities.

        Returns:
            float: The calculated score based on the specified criterion.

        Raises:
            ValueError: Jf the model does not support probability estimates required for
                Brier score evaluation.
        """
        if self.criterion == "f1":
            preds = (probs >= 0.5).astype(int)
            return f1_score(y_val, preds, pos_label=0)
        else:
            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "Model does not support probability estimates required for Brier "
                    "score evaluation."
                )
            return brier_score_loss(y_val, probs)
