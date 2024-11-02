"""Tests for dataloader."""

from unittest.mock import patch

import pandas as pd
import pytest

from periomod.base import BaseConfig
from periomod.data._dataloader import ProcessedDataLoader


def test_processed_data_loader_initialization():
    """Test initialization of ProcessedDataLoader."""
    loader = ProcessedDataLoader(
        task="pocketclosure", encoding="one_hot", encode=True, scale=True
    )
    assert loader.task == "pocketclosure"
    assert loader.encoding == "one_hot"
    assert loader.encode is True
    assert loader.scale is True


def test_encode_categorical_columns_one_hot():
    """Test one-hot encoding of categorical columns."""
    with patch.object(BaseConfig, "__init__", lambda x: None):
        loader = ProcessedDataLoader(
            task="pocketclosure", encoding="one_hot", encode=True, scale=False
        )
        loader.all_cat_vars = ["smokingtype", "tooth", "side"]

    df = pd.DataFrame(
        {
            "smokingtype": [1, 0],
            "age": [25, 30],
            "tooth": [11, 12],
            "side": [1, 2],
        }
    )

    df_encoded = loader.encode_categorical_columns(df)
    assert "smokingtype" not in df_encoded.columns
    assert "tooth" not in df_encoded.columns
    assert "side" not in df_encoded.columns

    expected_columns = [
        "age",
        "smokingtype_0",
        "smokingtype_1",
        "tooth_11",
        "tooth_12",
        "side_1",
        "side_2",
    ]
    assert all(col in df_encoded.columns for col in expected_columns)


def test_encode_categorical_columns_target():
    """Test target encoding of categorical columns."""
    with patch.object(BaseConfig, "__init__", lambda x: None):
        loader = ProcessedDataLoader(
            task="pocketclosure", encoding="target", encode=True, scale=False
        )
        loader.all_cat_vars = ["smokingtype", "tooth", "side"]

    df = pd.DataFrame(
        {
            "smokingtype": [1, 0],
            "age": [25, 30],
            "tooth": [11, 12],
            "side": [1, 2],
        }
    )

    df_encoded = loader.encode_categorical_columns(df)
    assert "smokingtype" in df_encoded.columns
    assert "tooth" not in df_encoded.columns
    assert "side" not in df_encoded.columns
    assert "toothside" in df_encoded.columns


def test_encode_categorical_columns_invalid_encoding():
    """Test that invalid encoding raises ValueError."""
    loader = ProcessedDataLoader(
        task="pocketclosure", encoding="invalid_encoding", encode=True, scale=False
    )

    df = pd.DataFrame(
        {
            "gender": [0, 1],
            "age": [25, 30],
        }
    )

    with pytest.raises(
        ValueError, match="Invalid encoding 'invalid_encoding' specified."
    ):
        loader.encode_categorical_columns(df)


def test_scale_numeric_columns():
    """Test scaling of numeric columns."""
    with patch.object(BaseConfig, "__init__", lambda x: None):
        loader = ProcessedDataLoader(
            task="pocketclosure", encoding=None, encode=False, scale=True
        )
        loader.scale_vars = ["age"]

    df = pd.DataFrame(
        {
            "age": [25, 30],
            "gender": [0, 1],
        }
    )

    df_scaled = loader.scale_numeric_columns(df)
    assert df_scaled["age"].mean() == pytest.approx(0.0, abs=1e-6)
    assert df_scaled["age"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_transform_data():
    """Test the complete data transformation process."""
    with patch.object(BaseConfig, "__init__", lambda x: None):
        loader = ProcessedDataLoader(
            task="pocketclosure", encoding="one_hot", encode=True, scale=True
        )
        loader.all_cat_vars = ["tooth", "side"]
        loader.scale_vars = ["age"]
        loader.task_cols = ["pocketclosure", "improvement"]
        loader.no_train_cols = []

    df = pd.DataFrame(
        {
            "age": [25, 30],
            "gender": [0, 1],
            "pocketclosure": [1, 0],
            "improvement": [0, 1],
            "tooth": [11, 12],
            "side": [1, 2],
        }
    )

    df_transformed = loader.transform_data(df)
    assert "tooth_11" in df_transformed.columns
    assert "tooth_12" in df_transformed.columns
    assert "side_1" in df_transformed.columns
    assert "side_2" in df_transformed.columns
    assert "tooth" not in df_transformed.columns
    assert "side" not in df_transformed.columns
    assert df_transformed["age"].mean() == pytest.approx(0.0, abs=1e-6)
    assert df_transformed["age"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)
    assert "y" in df_transformed.columns
    assert "pocketclosure" not in df_transformed.columns
    assert "improvement" not in df_transformed.columns


def test_transform_data_invalid_task():
    """Test that an invalid task raises ValueError."""
    with patch.object(BaseConfig, "__init__", lambda x: None):
        loader = ProcessedDataLoader(
            task="invalid_task", encoding=None, encode=False, scale=False
        )
        loader.task_cols = ["pocketclosure", "improvement"]
        loader.no_train_cols = []

    df = pd.DataFrame(
        {
            "age": [25, 30],
            "gender": [0, 1],
            "pocketclosure": [1, 0],
            "improvement": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="Task 'invalid_task' not supported."):
        loader.transform_data(df)
