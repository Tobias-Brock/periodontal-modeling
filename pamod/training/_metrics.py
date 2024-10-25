from typing import Any, Dict, Union

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


def get_probs(model, classification: str, X: pd.DataFrame) -> np.ndarray:
    """Gets the predicted probabilities from the model.

    Args:
        model: The trained model.
        classification (str): The type of classification.
        X (pd.DataFrame): Predict features.

    Returns:
        array-like: Predicted probabilities.
    """
    if classification == "binary":
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict_proba(X)


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
    g = y_bin.shape[1]
    return np.mean([brier_score_loss(y_bin[:, i], probs[:, i]) for i in range(g)]) * (
        g / 2
    )


def final_metrics(
    classification: str,
    y_test: np.ndarray,
    preds: np.ndarray,
    probs: Union[np.ndarray, None],
    threshold: Union[float, None] = None,
) -> Dict[str, Any]:
    """Calculate final metrics for binary or multiclass classification.

    Args:
        classification (str): The type of classification.
        y_test (np.ndarray): Ground truth (actual) labels.
        preds (np.ndarray): Predicted labels from the model.
        probs (Union[np.ndarray, None]): Predicted probabilities from model.
            Only used for binary classification and if available.
        threshold (Union[float, None]): Best threshold used for binary classification.
            Defaults to None.

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
