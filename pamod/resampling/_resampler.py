from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ._baseresampler import BaseResampler


class Resampler(BaseResampler):
    """Resampler class for handling data resampling and train-test splitting.

    This class extends `BaseResampler` to provide additional functionality
    for resampling datasets using various strategies (e.g., SMOTE, upsampling,
    downsampling) and for handling train-test splitting and cross-validation
    with group constraints.

    Inherits:
        - BaseResampler: Base class for resampling and validation methods.

    Args:
        classification (str): Specifies the type of classification ('binary'
            or 'multiclass').
        encoding (str): Specifies the encoding type ('one_hot' or 'target').

    Attributes:
        classification (str): Type of classification task ('binary' or 'multiclass').
        encoding (str): Encoding strategy for categorical features
            ('one_hot' or 'target').
        random_state_sampling (int): Seed used for reproducibility in sampling.
        all_cat_vars (list): List of categorical variables in the dataset, used in
            target encoding when applicable.

    Methods:
        split_train_test_df: Splits the dataset into train and test sets based
            on group constraints, ensuring reproducibility.
        split_x_y: Separates features and target labels in both train and test sets,
            applying optional sampling and encoding.
        cv_folds: Performs group-based cross-validation, applying resampling
            strategies to balance training data where specified.

    Inherited Methods:
        - `apply_sampling`: Applies specified sampling strategy to balance
          the dataset, supporting SMOTE, upsampling, and downsampling.
        - `apply_target_encoding`: Applies target encoding to categorical
          variables in the dataset.
        - `validate_dataframe`: Validates that input data meets requirements,
          such as having specified columns.
        - `validate_n_folds`: Ensures the number of cross-validation folds
          is a positive integer.
        - `validate_sampling_strategy`: Verifies the sampling strategy is
          one of the allowed options.

    Example:
        ```
        resampler = Resampler(classification="binary", encoding="one_hot")
        train_df, test_df = resampler.split_train_test_df(df, seed=42, test_size=0.3)
        X_train, y_train, X_test, y_test = resampler.split_x_y(
            train_df, test_df, sampling="upsampling", factor=1.5
        )
        outer_splits, cv_folds_indices = resampler.cv_folds(
            df, sampling="smote", factor=2.0, seed=42, n_folds=5
        )
        ```
    """

    def __init__(self, classification: str, encoding: str) -> None:
        """Initializes the Resampler class.

        Loads Hydra config values and sets parameters for random states, test sizes,
        and other relevant settings, including classification type.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            encoding (str): Tyoe if encoding ('one_hot' or 'target').
        """
        super().__init__(classification=classification, encoding=encoding)

    def split_train_test_df(
        self,
        df: pd.DataFrame,
        seed: Optional[int] = None,
        test_size: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into train_df and test_df based on group identifiers.

        Args:
            df (pd.DataFrame): Input DataFrame.
            seed (Optional[int], optional): Random seed for splitting. Defaults to None.
            test_size (Optional[float]): Size of grouped train test split.

        Returns:
            tuple: Tuple containing the training and test DataFrames
                (train_df, test_df).

        Raises:
            ValueError: If required columns are missing from the input DataFrame.
            TypeError: If the input DataFrame is not a pandas DataFrame.
        """
        self.validate_dataframe(df=df, required_columns=[self.y, self.group_col])

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size if test_size is not None else self.test_set_size,
            random_state=seed if seed is not None else self.random_state_split,
        )
        train_idx, test_idx = next(gss.split(df, groups=df[self.group_col]))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_patient_ids = set(train_df[self.group_col])
        test_patient_ids = set(test_df[self.group_col])
        if not train_patient_ids.isdisjoint(test_patient_ids):
            raise ValueError(
                "Overlapping group values between the train and test sets."
            )

        return train_df, test_df

    def split_x_y(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sampling: Union[str, None] = None,
        factor: Union[float, None] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Splits the train and test DataFrames into feature and label sets.

        Splits into (X_train, y_train, X_test, y_test).

        Args:
            train_df (pd.DataFrame): The training DataFrame.
            test_df (pd.DataFrame): The testing DataFrame.
            sampling (str, optional): Resampling method to apply (e.g.,
                'upsampling', 'downsampling', 'smote'), defaults to None.
            factor (float, optional): Factor for sampling, defaults to None.

        Returns:
            tuple: Tuple containing feature and label sets
                (X_train, y_train, X_test, y_test).

        Raises:
            ValueError: If required columns are missing or sampling method is invalid.
        """
        X_train = train_df.drop([self.y], axis=1)
        y_train = train_df[self.y]
        X_test = test_df.drop([self.y], axis=1)
        y_test = test_df[self.y]

        if self.encoding == "target":
            X_train, X_test = self.apply_target_encoding(
                X=X_train, X_val=X_test, y=y_train
            )

        if sampling is not None:
            X_train, y_train = self.apply_sampling(
                X=X_train, y=y_train, sampling=sampling, sampling_factor=factor
            )

        return (
            X_train.drop([self.group_col], axis=1),
            y_train,
            X_test.drop([self.group_col], axis=1),
            y_test,
        )

    def cv_folds(
        self,
        df: pd.DataFrame,
        sampling: Union[str, None] = None,
        factor: Union[float, None] = None,
        seed: Optional[int] = None,
        n_folds: Optional[int] = None,
    ) -> Tuple[list, list]:
        """Performs cross-validation with group constraints.

        Applies optional resampling strategies.

        Args:
            df (pd.DataFrame): Input DataFrame.
            sampling (str, optional): Sampling method to apply (e.g.,
                'upsampling', 'downsampling', 'smote').
            factor (float, optional): Factor for resampling, applied to upsample,
                downsample, or SMOTE.
            seed (Optional[int], optional): Random seed for reproducibility. Defaults
                to None.
            n_folds (Optional[int], optional): Number of folds for cross-validation.
                Defaults to None, in which case the class's `n_folds` will be used.


        Returns:
            tuple: A tuple containing outer splits and cross-validation fold indices.

        Raises:
            ValueError: If required columns are missing or folds are inconsistent.
            TypeError: If the input DataFrame is not a pandas DataFrame.
        """
        np.random.default_rng(seed=seed if seed is not None else self.random_state_cv)

        self.validate_dataframe(df=df, required_columns=[self.y, self.group_col])
        self.validate_n_folds(n_folds=self.n_folds)
        train_df, _ = self.split_train_test_df(df=df)
        gkf = GroupKFold(n_splits=n_folds if n_folds is not None else self.n_folds)

        cv_folds_indices = []
        outer_splits = []
        original_validation_data = []

        for train_idx, test_idx in gkf.split(train_df, groups=train_df[self.group_col]):
            X_train_fold = train_df.iloc[train_idx].drop([self.y], axis=1)
            y_train_fold = train_df.iloc[train_idx][self.y]
            X_test_fold = train_df.iloc[test_idx].drop([self.y], axis=1)
            y_test_fold = train_df.iloc[test_idx][self.y]

            original_validation_data.append(
                train_df.iloc[test_idx].drop([self.y], axis=1).reset_index(drop=True)
            )

            if sampling is not None:
                X_train_fold, y_train_fold = self.apply_sampling(
                    X=X_train_fold,
                    y=y_train_fold,
                    sampling=sampling,
                    sampling_factor=factor,
                )

            cv_folds_indices.append((train_idx, test_idx))
            outer_splits.append(
                ((X_train_fold, y_train_fold), (X_test_fold, y_test_fold))
            )

        for original_test_data, (_, (X_test_fold, _)) in zip(
            original_validation_data, outer_splits, strict=False
        ):
            if not original_test_data.equals(X_test_fold.reset_index(drop=True)):
                raise ValueError(
                    "Validation folds' data not consistent after applying sampling "
                    "strategies."
                )
        if self.encoding == "target":
            outer_splits_t = []

            for (X_t, y_t), (X_val, y_val) in outer_splits:
                X_t, X_val = self.apply_target_encoding(X=X_t, X_val=X_val, y=y_t)
                if sampling == "smote":
                    X_t, y_t = self.apply_sampling(
                        X=X_t, y=y_t, sampling=sampling, sampling_factor=factor
                    )

                outer_splits_t.append(((X_t, y_t), (X_val, y_val)))
            outer_splits = outer_splits_t

        return outer_splits, cv_folds_indices
