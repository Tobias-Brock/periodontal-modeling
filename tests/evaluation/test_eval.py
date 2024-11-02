"""Tests for evaluation module."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from periomod.evaluation._eval import ModelEvaluator


def create_sample_data(n_samples=100, n_features=5, n_classes=2, random_state=42):
    """Creates a sample dataset for testing."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_classes=n_classes,
        random_state=random_state,
        weights=[0.7, 0.3] if n_classes == 2 else None,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y)
    return X, y


def test_evaluate_feature_importance():
    """Test evaluate_feature_importance method."""
    X, y = create_sample_data()
    model = RandomForestClassifier()
    model.fit(X, y)
    evaluator = ModelEvaluator(
        X=X, y=y, model=model, encoding="one_hot", aggregate=True
    )
    evaluator.evaluate_feature_importance(fi_types=["shap"])
    plt.close("all")
    evaluator.evaluate_feature_importance(fi_types=["permutation"])
    plt.close("all")
    evaluator.evaluate_feature_importance(fi_types=["standard"])
    plt.close("all")
    with pytest.raises(ValueError, match="Invalid fi_type: invalid_type"):
        evaluator.evaluate_feature_importance(fi_types=["invalid_type"])


def test_analyze_brier_within_clusters():
    """Test analyze_brier_within_clusters method."""
    X, y = create_sample_data()
    model = LogisticRegression()
    model.fit(X, y)
    evaluator = ModelEvaluator(
        X=X, y=y, model=model, encoding="one_hot", aggregate=True
    )
    _, _, X_clustered = evaluator.analyze_brier_within_clusters(n_clusters=3)
    assert isinstance(X_clustered, pd.DataFrame)
    assert "Cluster" in X_clustered.columns
    assert "Brier_Score" in X_clustered.columns
    plt.close("all")


def test_brier_score_groups():
    """Test brier_score_groups method."""
    X, y = create_sample_data()
    model = LogisticRegression()
    model.fit(X, y)
    evaluator = ModelEvaluator(X=X, y=y, model=model)
    evaluator.brier_score_groups(group_by="y")
    plt.close("all")


def test_plot_confusion_matrix():
    """Test plot_confusion_matrix method."""
    X, y = create_sample_data()
    model = RandomForestClassifier()
    model.fit(X, y)
    evaluator = ModelEvaluator(X=X, y=y, model=model)
    evaluator.plot_confusion_matrix()
    plt.close("all")
    evaluator.plot_confusion_matrix(normalize="columns")
    plt.close("all")
    with pytest.raises(
        ValueError, match="Invalid value for 'normalize'. Use 'rows' or 'columns'."
    ):
        evaluator.plot_confusion_matrix(normalize="invalid_option")


def test_evaluate_feature_importance_with_multiple_models():
    """Test evaluate_feature_importance with multiple models."""
    X, y = create_sample_data()
    model1 = RandomForestClassifier()
    model1.fit(X, y)
    model2 = LogisticRegression()
    model2.fit(X, y)
    models = [model1, model2]

    evaluator = ModelEvaluator(X=X, y=y, models=models)
    evaluator.evaluate_feature_importance(fi_types=["standard"])
    plt.close("all")


def test_model_evaluator_initialization():
    """Test initialization of ModelEvaluator."""
    X, y = create_sample_data()
    model = RandomForestClassifier()
    model.fit(X, y)
    evaluator = ModelEvaluator(
        X=X, y=y, model=model, encoding="one_hot", aggregate=False
    )
    assert evaluator.X.equals(X)
    assert evaluator.y.equals(y)
    assert evaluator.model == model
    assert evaluator.models == []


def test_aggregate_one_hot_importances():
    """Test the _aggregate_one_hot_importances static method."""
    fi_df = pd.DataFrame(
        {
            "Feature": [
                "feature1_0",
                "feature1_1",
                "feature_2",
                "feature3_0",
                "feature3_1",
            ],
            "Importance": [0.1, 0.2, 0.3, 0.15, 0.25],
        }
    )
    aggregated_df = ModelEvaluator._aggregate_one_hot_importances(fi_df)
    expected_features = ["feature1", "feature", "feature3"]
    print("Aggregated DF:\n", aggregated_df)
    assert all(
        feature in aggregated_df["Feature"].values for feature in expected_features
    )
    assert len(aggregated_df) == 3


def test_aggregate_shap_one_hot():
    """Test the _aggregate_shap_one_hot static method."""
    shap_values = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    feature_names = [
        "feature1_0",
        "feature1_1",
        "feature_2",
        "feature3_0",
        "feature3_1",
    ]
    aggregated_shap_values, updated_feature_names = (
        ModelEvaluator._aggregate_shap_one_hot(
            shap_values=shap_values, feature_names=feature_names
        )
    )
    expected_features = ["feature1", "feature", "feature3"]
    print("Updated Feature Names:", updated_feature_names)
    assert all(feature in updated_feature_names for feature in expected_features)
    assert aggregated_shap_values.shape[1] == len(expected_features)
