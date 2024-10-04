from typing import Optional, Tuple, Union

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing._target_encoder import TargetEncoder

from pamod.base import BaseData, BaseValidator


class Resampler(BaseValidator, BaseData):
    def __init__(self, classification: str, encoding: str) -> None:
        """Initializes the Resampler class.

        Loads Hydra config values and sets parameters for random states, test sizes,
        and other relevant settings, including classification type.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            encoding (str): Tyoe if encoding ('one_hot' or 'target').
        """
        BaseValidator.__init__(self, classification)
        BaseData.__init__(self)
        self.encoding = encoding

    def apply_sampling(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling: str,
        sampling_factor: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Applies resampling strategies to the dataset.

        Methods such as SMOTE, upsampling, or downsampling are applied.

        Args:
            X (pd.DataFrame): The feature set of the dataset.
            y (pd.Series): The target variable containing class labels.
            sampling (str): The type of sampling to apply. Options are 'smote',
                'upsampling', 'downsampling', or None.
            sampling_factor (float): The factor by which to upsample or downsample.

        Returns:
            tuple: Resampled feature set (X_resampled) and target labels (y_resampled).

        Raises:
            ValueError: If an invalid sampling or classification method is specified.
        """
        self.validate_sampling_strategy(sampling)
        if sampling == "smote":
            if self.classification == "multiclass":
                smote_strategy = {
                    1: sum(y == 1) * sampling_factor,
                    2: sum(y == 2) * sampling_factor,
                }
            elif self.classification == "binary":
                smote_strategy = {1: sum(y == 1) * sampling_factor}
            smote_sampler = SMOTE(
                sampling_strategy=smote_strategy,
                random_state=self.random_state_sampling,
            )
            return smote_sampler.fit_resample(X, y)

        elif sampling == "upsampling":
            if self.classification == "multiclass":
                up_strategy = {
                    1: sum(y == 1) * sampling_factor,
                    2: sum(y == 2) * sampling_factor,
                }
            elif self.classification == "binary":
                up_strategy = {0: sum(y == 0) * sampling_factor}
            up_sampler = RandomOverSampler(
                sampling_strategy=up_strategy, random_state=self.random_state_sampling
            )
            return up_sampler.fit_resample(X, y)

        elif sampling == "downsampling":
            if self.classification in ["binary", "multiclass"]:
                down_strategy = {1: sum(y == 1) // sampling_factor}
            down_sampler = RandomUnderSampler(
                sampling_strategy=down_strategy, random_state=self.random_state_sampling
            )
            return down_sampler.fit_resample(X, y)

        else:
            return X, y

    def apply_target_encoding(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """Applies target encoding to categorical variables.

        Args:
            X_train (pd.DataFrame): Training dataset.
            X_val (pd.DataFrame): Validation dataset.
            y_train (pd.Series): The target variable.

        Returns:
            pd.DataFrame: Dataset with target encoded features.
        """
        cat_vars = [col for col in self.all_cat_vars if col in X_train.columns]

        if cat_vars:
            encoder = TargetEncoder(target_type=self.classification)
            X_train_encoded = encoder.fit_transform(X_train[cat_vars], y_train)
            X_val_encoded = encoder.transform(X_val[cat_vars])

            if self.classification == "multiclass":
                n_classes = len(set(y_train))
                encoded_cols = [
                    f"{col}_class_{i}" for col in cat_vars for i in range(n_classes)
                ]
            else:
                encoded_cols = cat_vars

            X_train_encoded = pd.DataFrame(
                X_train_encoded, columns=encoded_cols, index=X_train.index
            )
            X_val_encoded = pd.DataFrame(
                X_val_encoded, columns=encoded_cols, index=X_val.index
            )

            X_train.drop(columns=cat_vars, inplace=True)
            X_val.drop(columns=cat_vars, inplace=True)
            X_train = pd.concat([X_train, X_train_encoded], axis=1)
            X_val = pd.concat([X_val, X_val_encoded], axis=1)

        return X_train, X_val

    def split_train_test_df(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into train_df and test_df based on group identifiers.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: Tuple containing the training and test DataFrames
                (train_df, test_df).

        Raises:
            ValueError: If required columns are missing from the input DataFrame.
            TypeError: If the input DataFrame is not a pandas DataFrame.
        """
        self.validate_dataframe(df, ["y", self.group_col])

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_set_size,
            random_state=self.random_state_split,
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
        X_train = train_df.drop(["y"], axis=1)
        y_train = train_df["y"]
        X_test = test_df.drop(["y"], axis=1)
        y_test = test_df["y"]

        if self.encoding == "target":
            X_train, X_test = self.apply_target_encoding(X_train, X_test, y_train)

        if sampling is not None:
            X_train, y_train = self.apply_sampling(X_train, y_train, sampling, factor)

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
    ) -> Tuple[list, list]:
        """Performs cross-validation with group constraints.

        Applies optional resampling strategies.

        Args:
            df (pd.DataFrame): Input DataFrame.
            sampling (str, optional): Sampling method to apply (e.g.,
                'upsampling', 'downsampling', 'smote').
            factor (float, optional): Factor for resampling, applied to upsample,
                downsample, or SMOTE.

        Returns:
            tuple: A tuple containing outer splits and cross-validation fold indices.

        Raises:
            ValueError: If required columns are missing or folds are inconsistent.
            TypeError: If the input DataFrame is not a pandas DataFrame.
        """
        np.random.default_rng(self.random_state_cv)

        self.validate_dataframe(df, ["y", self.group_col])
        self.validate_n_folds(self.n_folds)
        train_df, _ = self.split_train_test_df(df)
        gkf = GroupKFold(n_splits=self.n_folds)

        cv_folds_indices = []
        outer_splits = []
        original_validation_data = []

        for train_idx, test_idx in gkf.split(train_df, groups=train_df[self.group_col]):
            X_train_fold = train_df.iloc[train_idx].drop(["y"], axis=1)
            y_train_fold = train_df.iloc[train_idx]["y"]
            X_test_fold = train_df.iloc[test_idx].drop(["y"], axis=1)
            y_test_fold = train_df.iloc[test_idx]["y"]

            original_validation_data.append(
                train_df.iloc[test_idx].drop(["y"], axis=1).reset_index(drop=True)
            )

            if sampling is not None:
                X_train_fold, y_train_fold = self.apply_sampling(
                    X_train_fold, y_train_fold, sampling, factor
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
                    "Validation folds' data are not consistent after applying sampling "
                    "strategies."
                )
        if self.encoding == "target":
            outer_splits_t = []

            for (X_t, y_t), (X_val, y_val) in outer_splits:
                X_t, y_t, X_val = self.apply_target_encoding(X_t, y_t, X_val)
                if sampling == "smote":
                    X_t, y_t = self.apply_sampling(X_t, y_t, sampling, factor)

                outer_splits_t.append(((X_t, y_t), (X_val, y_val)))
            outer_splits = outer_splits_t

        return outer_splits, cv_folds_indices
