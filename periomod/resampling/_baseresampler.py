from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.preprocessing._target_encoder import TargetEncoder

from ..base import BaseConfig


class BaseResampler(BaseConfig, ABC):
    """Abstract base class for implementing various resampling strategies.

    This class provides a foundational framework for resampling data and validating
    input parameters in classification tasks. It includes methods for applying
    upsampling, downsampling, and SMOTE, as well as handling target encoding,
    data validation, and configuring cross-validation folds.

    Inherits:
        - `BaseConfig`: Provides configuration settings for data processing.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        classification (str): Specifies the classification type ('binary' or
            'multiclass').
        encoding (str): Specifies the encoding type ('one_hot' or 'target').

    Attributes:
        classification (str): The type of classification task
            ('binary' or 'multiclass').
        encoding (str): Encoding method for categorical features
            ('one_hot' or 'target').
        all_cat_vars (list): List of all categorical variables in the dataset that
            can be used in target encoding.

    Methods:
        apply_sampling: Applies resampling techniques like SMOTE, upsampling,
            or downsampling to balance the dataset.
        apply_target_encoding: Encodes categorical features based on the
            target variable for improved model performance.
        validate_dataframe: Ensures the input DataFrame contains required
            columns and correct data types.
        validate_n_folds: Verifies that the cross-validation fold count is a
            positive integer.
        validate_sampling_strategy: Checks if the specified sampling strategy
            is valid.

    Abstract Methods:
        - `split_train_test_df`: Splits the dataset into training and testing sets
          based on group-based identifiers.
        - `split_x_y`: Divides the train and test DataFrames into feature and
          target sets, with optional resampling.
        - `cv_folds`: Performs cross-validation with group-based constraints and
          optional resampling for each fold.
    """

    def __init__(self, classification: str, encoding: str) -> None:
        """Base class to provide validation and error handling for other classes."""
        super().__init__()
        self.classification = classification
        self.encoding = encoding

    def apply_sampling(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling: str,
        sampling_factor: Optional[float] = None,
        random_state: Optional[int] = 0,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Applies resampling strategies to the dataset.

        Methods such as SMOTE, upsampling, or downsampling are applied.

        Args:
            X (pd.DataFrame): The feature set of the dataset.
            y (pd.Series): The target variable containing class labels.
            sampling (str): The type of sampling to apply. Options are 'smote',
                'upsampling', 'downsampling', or None.
            sampling_factor (Optional[float]): The factor by which to upsample or
                downsample.
            random_state (Optional[int]): Random state for sampling. Defaults to 0.

        Returns:
            Tuple: Resampled feature set (X_resampled) and target labels (y_resampled).

        Raises:
            ValueError: If an invalid sampling or classification method is specified.
        """
        self.validate_sampling_strategy(sampling=sampling)
        if sampling == "smote":
            if self.classification == "multiclass":
                smote_strategy = {
                    1: int(sum(y == 1) * sampling_factor),
                    2: int(sum(y == 2) * sampling_factor),
                }
            elif self.classification == "binary":
                smote_strategy = {1: int(sum(y == 1) * sampling_factor)}
            smote_sampler = SMOTE(
                sampling_strategy=smote_strategy,
                random_state=random_state,
            )
            return smote_sampler.fit_resample(X=X, y=y)

        elif sampling == "upsampling":
            if self.classification == "multiclass":
                up_strategy = {
                    1: int(sum(y == 1) * sampling_factor),
                    2: int(sum(y == 2) * sampling_factor),
                }
            elif self.classification == "binary":
                up_strategy = {0: int(sum(y == 0) * sampling_factor)}
            up_sampler = RandomOverSampler(
                sampling_strategy=up_strategy, random_state=random_state
            )
            return up_sampler.fit_resample(X=X, y=y)

        elif sampling == "downsampling":
            if self.classification in ["binary", "multiclass"]:
                down_strategy = {1: int(sum(y == 1) // sampling_factor)}
            down_sampler = RandomUnderSampler(
                sampling_strategy=down_strategy, random_state=random_state
            )
            return down_sampler.fit_resample(X=X, y=y)

        else:
            return X, y

    def apply_target_encoding(
        self,
        X: pd.DataFrame,
        X_val: pd.DataFrame,
        y: pd.Series,
        jackknife: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies target encoding to categorical variables.

        Args:
            X (pd.DataFrame): Training dataset.
            X_val (pd.DataFrame): Validation dataset.
            y (pd.Series): The target variable.
            jackknife (bool, optional): If True, do not transform X_val.
                Defaults to False.

        Returns:
            Tuple: X and X_val dataset with target encoded features.
        """
        cat_vars = [col for col in self.all_cat_vars if col in X.columns]

        if cat_vars:
            encoder = TargetEncoder(
                target_type=self.classification, random_state=self.target_state
            )
            X_encoded = encoder.fit_transform(X[cat_vars], y)

            if not jackknife and X_val is not None:
                X_val_encoded = encoder.transform(X_val[cat_vars])
            else:
                X_val_encoded = None

            if self.classification == "multiclass":
                n_classes = len(set(y))
                encoded_cols = [
                    f"{col}_class_{i}" for col in cat_vars for i in range(n_classes)
                ]
            else:
                encoded_cols = cat_vars

            X_encoded = pd.DataFrame(X_encoded, columns=encoded_cols, index=X.index)

            if X_val_encoded is not None:
                X_val_encoded = pd.DataFrame(
                    X_val_encoded, columns=encoded_cols, index=X_val.index
                )

            X.drop(columns=cat_vars, inplace=True)
            if X_val is not None:
                X_val.drop(columns=cat_vars, inplace=True)

            X = pd.concat([X, X_encoded], axis=1)
            if X_val_encoded is not None:
                X_val = pd.concat([X_val, X_val_encoded], axis=1)

        return X, X_val

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list) -> None:
        """Validate input is a pandas DataFrame and contains required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_columns (list): A list of column names that are required in
                the DataFrame.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            ValueError: If required columns are missing from the DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected input to be a pandas DataFrame, but got {type(df)}."
            )

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following required columns are missing: "
                f"{', '.join(missing_columns)}."
            )

    @staticmethod
    def validate_n_folds(n_folds: Optional[int]) -> None:
        """Validates the number of folds used in cross-validation.

        Args:
            n_folds (Optional[int]): The number of folds for cross-validation.

        Raises:
            ValueError: If the number of folds is not a positive integer.
        """
        if not (isinstance(n_folds, int) and n_folds > 0):
            raise ValueError("'n_folds' must be a positive integer.")

    @staticmethod
    def validate_sampling_strategy(sampling: str) -> None:
        """Validates the sampling strategy.

        Args:
            sampling (str): The sampling strategy to validate.

        Raises:
            ValueError: If the sampling strategy is invalid.
        """
        valid_strategies = ["smote", "upsampling", "downsampling", None]
        if sampling not in valid_strategies:
            raise ValueError(
                f"Invalid sampling strategy: {sampling}. Valid options are "
                f"{valid_strategies}."
            )

    @abstractmethod
    def split_train_test_df(
        self,
        df: pd.DataFrame,
        seed: int,
        test_size: float,
    ):
        """Splits the dataset into train_df and test_df based on group identifiers.

        Args:
            df (pd.DataFrame): Input DataFrame.
            seed (int): Random seed for splitting.
            test_size (float): Size of grouped train test split.
        """

    @abstractmethod
    def split_x_y(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sampling: Union[str, None],
        factor: Union[float, None],
    ):
        """Splits the train and test DataFrames into feature and label sets.

        Splits into (X_train, y_train, X_test, y_test).

        Args:
            train_df (pd.DataFrame): The training DataFrame.
            test_df (pd.DataFrame): The testing DataFrame.
            sampling (str, optional): Resampling method to apply (e.g.,
                'upsampling', 'downsampling', 'smote').
            factor (float, optional): Factor for sampling.
        """

    @abstractmethod
    def cv_folds(
        self,
        df: pd.DataFrame,
        seed: int,
        n_folds: int,
        sampling: Union[str, None],
        factor: Union[float, None],
    ):
        """Performs cross-validation with group constraints.

        Applies optional resampling strategies.

        Args:
            df (pd.DataFrame): Input DataFrame.
            seed (Optional[int], optional): Random seed for reproducibility.
            n_folds (Optional[int], optional): Number of folds for cross-validation.
            sampling (str, optional): Sampling method to apply (e.g.,
                'upsampling', 'downsampling', 'smote').
            factor (float, optional): Factor for resampling, applied to upsample,
                downsample, or SMOTE.
        """
