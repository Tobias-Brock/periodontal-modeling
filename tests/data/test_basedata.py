"""Tests for base data methods."""

import os
from pathlib import Path
from typing import Union
from unittest.mock import patch
import warnings

import pandas as pd
import pytest

from periomod.base import BaseConfig
from periomod.data._basedata import (
    BaseDataLoader,
    BaseLoader,
    BaseProcessor,
)


def test_base_loader_abstract_methods():
    """Test that BaseLoader cannot be instantiated due to abstract methods."""
    with pytest.raises(TypeError):
        BaseLoader()


def test_base_processor_abstract_methods():
    """Test that BaseProcessor cannot be instantiated due to abstract methods."""
    with pytest.raises(TypeError):
        BaseProcessor()


def test_base_data_loader_abstract_methods():
    """Test that BaseDataLoader cannot be instantiated due to abstract methods."""
    with pytest.raises(TypeError):
        BaseDataLoader(task="task_col", encoding=None, encode=False, scale=False)


def test_base_loader_save_data_empty_df():
    """Test that save_data raises ValueError when provided with an empty DataFrame."""

    class ConcreteLoader(BaseLoader):
        def load_data(self, path: Union[str, Path]):
            pass

        def save_data(self, df: pd.DataFrame, path: Union[str, Path]):
            super().save_data(df, path)

    loader = ConcreteLoader()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Data must be processed before saving."):
        loader.save_data(empty_df, path=Path("/tmp/test.csv"))


def test_base_loader_save_data():
    """Test the save_data method saves a DataFrame to the specified path."""

    class ConcreteLoader(BaseLoader):
        def load_data(self, path: Union[str, Path]):
            pass

        def save_data(self, df: pd.DataFrame, path: Union[str, Path]):
            super().save_data(df, path)

    loader = ConcreteLoader()
    data = pd.DataFrame({"A": [1, 2, 3]})
    path = Path("/tmp/test_save_data/test.csv")

    if path.exists():
        os.remove(path)

    loader.save_data(data, path)
    assert path.exists()
    os.remove(path)


def test_base_processor_load_data_missing_columns():
    """Test that load_data warns when required columns are missing."""

    class ConcreteProcessor(BaseProcessor):
        def impute_missing_values(self, df: pd.DataFrame):
            pass

        def create_tooth_features(self, df: pd.DataFrame):
            pass

        def create_outcome_variables(self, df: pd.DataFrame):
            pass

        def process_data(self, df: pd.DataFrame):
            pass

    with (
        patch("pandas.read_excel") as mock_read_excel,
        patch("pathlib.Path.exists") as mock_exists,
    ):
        mock_df = pd.DataFrame({
            "Age": [25, 30],
            "Gender": [0, 1],
        })
        mock_read_excel.return_value = mock_df
        mock_exists.return_value = True

        processor = ConcreteProcessor(behavior=True)

        with warnings.catch_warnings(record=True) as w:
            data = processor.load_data(path=Path("/tmp/test.xlsx"))
            assert len(w) == 2
            assert issubclass(w[-1].category, UserWarning)
            assert "Warning: Missing cols" in str(w[-1].message)

        assert data.shape[1] <= len(processor.required_columns)


def test_base_data_loader_check_encoded_columns():
    """Test that _check_encoded_columns works as expected."""

    class ConcreteDataLoader(BaseDataLoader):
        def encode_categorical_columns(self, df: pd.DataFrame, fit_encoder: bool):
            pass

        def scale_numeric_columns(self, df: pd.DataFrame):
            pass

        def transform_data(self, data: pd.DataFrame, fit_encoder: bool):
            pass

    with patch.object(BaseConfig, "__init__", lambda x: None):
        loader = ConcreteDataLoader(
            task="task_col", encoding="one_hot", encode=True, scale=False
        )
        loader.all_cat_vars = ["gender", "smoking_status"]

    data = pd.DataFrame({
        "gender_0": [1, 0],
        "gender_1": [0, 1],
        "smoking_status_0": [0, 1],
        "smoking_status_1": [1, 0],
        "age": [25, 30],
    })

    loader._check_encoded_columns(data)
    df_incomplete = pd.DataFrame({
        "gender": [0, 1],
        "smoking_status_0": [0, 1],
        "smoking_status_1": [1, 0],
        "age": [25, 30],
    })

    with pytest.raises(
        ValueError, match="Column 'gender' was not correctly one-hot encoded."
    ):
        loader._check_encoded_columns(df_incomplete)


def test_base_data_loader_check_scaled_columns():
    """Test that _check_scaled_columns works as expected."""

    class ConcreteDataLoader(BaseDataLoader):
        def encode_categorical_columns(self, df: pd.DataFrame, fit_encoder: bool):
            pass

        def scale_numeric_columns(self, df: pd.DataFrame):
            pass

        def transform_data(self, df: pd.DataFrame, fit_encoder: bool):
            pass

    with patch.object(BaseConfig, "__init__", lambda x: None):
        loader = ConcreteDataLoader(
            task="task_col", encoding=None, encode=False, scale=True
        )
        loader.scale_vars = ["age", "bmi"]

    data = pd.DataFrame({
        "age": [0.0, 1.0],
        "bmi": [-1.0, 0.5],
    })
    loader._check_scaled_columns(data)

    df_improper = pd.DataFrame({
        "age": [-10.0, 100.0],
        "bmi": [-1.0, 0.5],
    })

    with pytest.raises(ValueError, match="Column age is not correctly scaled."):
        loader._check_scaled_columns(df_improper)


def test_base_data_loader_save_and_load_data():
    """Test that save_data and load_data methods work as expected."""

    class ConcreteDataLoader(BaseDataLoader):
        def encode_categorical_columns(self, df: pd.DataFrame, fit_encoder: bool):
            pass

        def scale_numeric_columns(self, df: pd.DataFrame):
            pass

        def transform_data(self, df: pd.DataFrame, fit_encoder: bool):
            pass

    loader = ConcreteDataLoader(
        task="task_col", encoding=None, encode=False, scale=False
    )

    data = pd.DataFrame({
        "age": [25, 30],
        "gender": [0, 1],
    })

    path = Path("/tmp/test_data_loader/test_training_data.csv")

    if path.exists():
        os.remove(path)  # Use os.remove for files

    loader.save_data(data, path)
    loaded_df = loader.load_data(path)
    pd.testing.assert_frame_equal(data, loaded_df)
    os.remove(path)  # Clean up the created file
