"""Tests for BaseResampler."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from periomod.resampling._resampler import Resampler


def create_sample_data(
    n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=42
):
    """Creates a sample dataset for testing.

    Args:
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of feature columns.
        n_informative (int): Number of informative features.
        n_classes (int): Number of output classes.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A synthetic dataset with `n_features` feature columns, a target
        column `y`, and a `group` column for group identifiers.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state,
        weights=[0.7, 0.3],
    )
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    data["y"] = y
    data["group"] = [i // 10 for i in range(n_samples)]  # Create group identifiers
    return data


def test_resampler_initialization():
    """Test initialization of Resampler."""
    resampler = Resampler(classification="binary", encoding="one_hot")
    assert resampler.classification == "binary"
    assert resampler.encoding == "one_hot"


def test_split_train_test_df():
    """Test splitting data into train and test DataFrames."""
    data = create_sample_data()
    resampler = Resampler(classification="binary", encoding="one_hot")
    resampler.y = "y"
    resampler.group_col = "group"
    train_df, test_df = resampler.split_train_test_df(data, seed=42, test_size=0.2)
    train_groups = set(train_df["group"])
    test_groups = set(test_df["group"])
    assert train_groups.isdisjoint(test_groups)
    assert len(train_df) > len(test_df)


def test_split_x_y():
    """Test splitting DataFrames into features and labels with sampling."""
    data = create_sample_data()
    resampler = Resampler(classification="binary", encoding="one_hot")
    resampler.y = "y"
    resampler.group_col = "group"
    resampler.all_cat_vars = []

    train_df, test_df = resampler.split_train_test_df(data, seed=42, test_size=0.2)
    X_train, y_train, X_test, y_test = resampler.split_x_y(
        train_df, test_df, sampling="upsampling", factor=1.5
    )
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_cv_folds():
    """Test cross-validation folds creation."""
    data = create_sample_data()
    resampler = Resampler(classification="binary", encoding="one_hot")
    resampler.y = "y"
    resampler.group_col = "group"
    resampler.n_folds = 5
    resampler.all_cat_vars = []

    outer_splits, cv_folds_indices = resampler.cv_folds(
        data, sampling="upsampling", factor=1.5, seed=42, n_folds=5
    )

    assert len(outer_splits) == 5
    assert len(cv_folds_indices) == 5

    for (X_train, _), (X_val, _) in outer_splits:
        train_groups = set(X_train["group"])
        val_groups = set(X_val["group"])
        assert train_groups.isdisjoint(val_groups)


def test_invalid_sampling_strategy():
    """Test that an invalid sampling strategy raises a ValueError."""
    data = create_sample_data()
    resampler = Resampler(classification="binary", encoding="one_hot")
    resampler.y = "y"
    resampler.group_col = "group"

    with pytest.raises(ValueError, match="Invalid sampling strategy: invalid_option."):
        resampler.apply_sampling(
            X=data.drop("y", axis=1),
            y=data["y"],
            sampling="invalid_option",
            sampling_factor=1.0,
        )


def test_apply_target_encoding():
    """Test target encoding application."""
    data = create_sample_data()
    data["cat_feature"] = ["A", "B"] * (len(data) // 2)
    resampler = Resampler(classification="binary", encoding="target")
    resampler.y = "y"
    resampler.group_col = "group"
    resampler.all_cat_vars = ["cat_feature"]

    train_df, test_df = resampler.split_train_test_df(data, seed=42, test_size=0.2)
    X_train, _, _, _ = resampler.split_x_y(train_df, test_df)

    assert any(col.startswith("cat_feature") for col in X_train.columns)


def test_validate_dataframe():
    """Test dataframe validation."""
    data = create_sample_data()
    resampler = Resampler(classification="binary", encoding="one_hot")

    data = data.drop(columns=["y"])

    with pytest.raises(
        ValueError, match="The following required columns are missing: y."
    ):
        resampler.validate_dataframe(df=data, required_columns=["y", "group"])


def test_validate_n_folds():
    """Test n_folds validation."""
    with pytest.raises(ValueError, match="'n_folds' must be a positive integer."):
        Resampler.validate_n_folds(n_folds=0)


def test_validate_sampling_strategy():
    """Test sampling strategy validation."""
    with pytest.raises(
        ValueError, match="Invalid sampling strategy: invalid_strategy."
    ):
        Resampler.validate_sampling_strategy(sampling="invalid_strategy")
