"""Tests for metrics."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from periomod.training._metrics import brier_loss_multi, final_metrics, get_probs


def create_sample_data(n_samples=100, n_features=5, n_classes=2, random_state=42):
    """Creates a sample dataset for testing.

    Args:
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of feature columns.
        n_classes (int): Number of output classes.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - `X` (np.ndarray): Feature matrix of shape (n_samples, n_features).
            - `y` (np.ndarray): Target array of shape (n_samples,).
    """
    if n_classes > 2:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            n_classes=n_classes,
            n_clusters_per_class=1,  # Ensures that each class forms a distinct cluster
            random_state=random_state,
        )
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=3,
            n_classes=n_classes,
            weights=[0.7, 0.3],
            random_state=random_state,
        )
    return X, y


def test_get_probs_binary():
    """Test get_probs function for binary classification."""
    X, y = create_sample_data(n_classes=2)
    model = LogisticRegression()
    model.fit(X, y)
    probs = get_probs(model, classification="binary", X=X)
    assert probs.shape[0] == X.shape[0]
    assert probs.ndim == 1


def test_get_probs_multiclass():
    """Test get_probs function for multiclass classification."""
    X, y = create_sample_data(n_classes=3)
    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)
    probs = get_probs(model, classification="multiclass", X=X)
    assert probs.shape == (X.shape[0], 3)  # 3 classes
    assert probs.ndim == 2


def test_brier_loss_multi():
    """Test brier_loss_multi function."""
    X, y = create_sample_data(n_classes=3)
    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)
    probs = model.predict_proba(X)
    brier_score = brier_loss_multi(y, probs)
    assert isinstance(brier_score, float)
    assert brier_score >= 0


def test_final_metrics_binary():
    """Test final_metrics function for binary classification."""
    X, y = create_sample_data(n_classes=2)
    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    metrics = final_metrics(
        classification="binary", y=y, preds=preds, probs=probs, threshold=0.5
    )
    expected_keys = [
        "F1 Score",
        "Precision",
        "Recall",
        "Accuracy",
        "Brier Score",
        "ROC AUC Score",
        "Confusion Matrix",
        "Best Threshold",
    ]
    assert all(key in metrics for key in expected_keys)
    assert isinstance(metrics["Confusion Matrix"], np.ndarray)


def test_final_metrics_multiclass():
    """Test final_metrics function for multiclass classification."""
    X, y = create_sample_data(n_classes=3)
    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    print(probs)
    metrics = final_metrics(classification="multiclass", y=y, preds=preds, probs=probs)
    expected_keys = [
        "Macro F1",
        "Accuracy",
        "Class F1 Scores",
        "Multiclass Brier Score",
    ]
    assert all(key in metrics for key in expected_keys)
    assert isinstance(metrics["Class F1 Scores"], np.ndarray)
    assert len(metrics["Class F1 Scores"]) == 3
