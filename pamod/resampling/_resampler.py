import numpy as np
import pandas as pd
import hydra
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


class Resampler:
    def __init__(self):
        """
        Initializes the ResamplingEngine class by loading Hydra config values and
        setting parameters for random states, test sizes, and other relevant settings.
        """
        with hydra.initialize(config_path="../../config", version_base="1.2"):
            cfg = hydra.compose(config_name="config")

        self.random_state_sampling = cfg.resample.random_state_sampling
        self.random_state_split = cfg.resample.random_state_split
        self.random_state_cv = cfg.resample.random_state_cv
        self.test_set_size = cfg.resample.test_set_size
        self.group_col = cfg.resample.group_col
        self.n_folds = cfg.resample.n_folds

    def apply_sampling(self, X, y, upsample, downsample, smote, classification):
        """
        Applies resampling strategies to the dataset, such as SMOTE, upsampling, or downsampling.

        Args:
            X (pd.DataFrame): The feature set of the dataset.
            y (pd.Series): The target variable containing class labels.
            upsample (int or None): Upsampling factor for minority classes. Ignored if 'smote' is specified.
            downsample (int or None): Downsampling factor for the majority class.
            smote (int or None): SMOTE factor for minority classes. If True, upsample and downsample are ignored.
            classification (str): The type of classification, either 'binary' or 'multiclass'.

        Returns:
            tuple: Resampled feature set (X_resampled) and target labels (y_resampled).

        Raises:
            ValueError: If conflicting resampling strategies are specified.
        """
        if smote and (upsample or downsample):
            raise ValueError(
                "Conflicting resampling strategies specified. Choose either SMOTE, upsampling, or downsampling."
            )

        if smote:
            if classification == "multiclass":
                smote_strategy = {1: sum(y == 1) * smote, 2: sum(y == 2) * smote}
            elif classification == "binary":
                smote_strategy = {1: sum(y == 1) * smote}
            else:
                raise ValueError("Invalid classification method specified.")
            smote_sampler = SMOTE(sampling_strategy=smote_strategy, random_state=self.random_state_sampling)
            X_resampled, y_resampled = smote_sampler.fit_resample(X, y)
        elif upsample:
            if classification == "multiclass":
                up_strategy = {1: sum(y == 1) * upsample, 2: sum(y == 2) * upsample}
            elif classification == "binary":
                up_strategy = {1: sum(y == 1) * upsample}
            else:
                raise ValueError("Invalid classification method specified.")
            up_sampler = RandomOverSampler(sampling_strategy=up_strategy, random_state=self.random_state_sampling)
            X_resampled, y_resampled = up_sampler.fit_resample(X, y)
        elif downsample:
            if classification in ["binary", "multiclass"]:
                down_strategy = {0: sum(y == 0) // downsample}
            else:
                raise ValueError("Invalid classification method specified.")
            down_sampler = RandomUnderSampler(sampling_strategy=down_strategy, random_state=self.random_state_sampling)
            X_resampled, y_resampled = down_sampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        return X_resampled, y_resampled

    def split_train_test(self, df, classification, upsample, downsample, smote):
        """
        Splits the dataset into training and testing sets based on group identifiers, with resampling applied.

        Args:
            df (pd.DataFrame): Input DataFrame.
            classification (str): Type of classification ('binary' or 'multiclass').
            upsample (int or None): Upsampling factor for minority classes.
            downsample (int or None): Downsampling factor for the majority class.
            smote (int or None): SMOTE factor for minority classes.

        Returns:
            tuple: Tuple containing training and test sets: (X_train, y_train, X_test, y_test, train_df, test_df).

        Raises:
            ValueError: If required columns are missing from the input DataFrame.
            TypeError: If the input DataFrame is not a pandas DataFrame.
        """
        required_columns = ["y", self.group_col]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_set_size, random_state=self.random_state_split)
        train_idx, test_idx = next(gss.split(df, groups=df[self.group_col]))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        X_train = train_df.drop(["y"], axis=1)
        y_train = train_df["y"]
        X_test = test_df.drop(["y"], axis=1)
        y_test = test_df["y"]

        train_patient_ids = set(train_df[self.group_col])
        test_patient_ids = set(test_df[self.group_col])
        if not train_patient_ids.isdisjoint(test_patient_ids):
            raise ValueError("Overlapping group values between the train and test sets.")

        X_train, y_train = self.apply_sampling(X_train, y_train, upsample, downsample, smote, classification)

        return (
            X_train.drop([self.group_col], axis=1),
            y_train,
            X_test.drop([self.group_col], axis=1),
            y_test,
            train_df,
            test_df,
        )

    def cv_folds(self, df, classification, upsample, downsample, smote):
        """
        Performs cross-validation with group constraints, applying optional resampling strategies.

        Args:
            df (pd.DataFrame): Input DataFrame.
            classification (str): Type of classification ('binary' or 'multiclass').
            upsample (int or None): Upsampling factor for minority classes.
            downsample (int or None): Downsampling factor for the majority class.
            smote (int or None): SMOTE factor for minority classes.

        Returns:
            tuple: A tuple containing outer splits and cross-validation fold indices.

        Raises:
            ValueError: If required columns are missing or validation folds are inconsistent.
            TypeError: If the input DataFrame is not a pandas DataFrame.
        """
        np.random.seed(self.random_state_cv)

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if "y" not in df.columns:
            raise ValueError("Column 'y' not found in DataFrame.")
        if self.group_col not in df.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in DataFrame.")
        if not isinstance(self.n_folds, int) or self.n_folds <= 0:
            raise ValueError("'n_folds' must be a positive integer.")

        _, _, _, _, train_df, _ = self.split_train_test(df, classification, upsample=None, downsample=None, smote=None)
        gkf = GroupKFold(n_splits=self.n_folds)

        cv_folds_indices = []
        outer_splits = []
        original_validation_data = []

        for train_idx, test_idx in gkf.split(train_df, groups=train_df[self.group_col]):
            X_train_fold = train_df.iloc[train_idx].drop(["y"], axis=1)
            y_train_fold = train_df.iloc[train_idx]["y"]
            X_test_fold = train_df.iloc[test_idx].drop(["y"], axis=1)
            y_test_fold = train_df.iloc[test_idx]["y"]

            original_validation_data.append(train_df.iloc[test_idx].drop(["y"], axis=1).reset_index(drop=True))

            if any([upsample, downsample, smote]):
                X_train_fold, y_train_fold = self.apply_sampling(
                    X_train_fold, y_train_fold, upsample, downsample, smote, classification
                )

            cv_folds_indices.append((train_idx, test_idx))
            outer_splits.append(((X_train_fold, y_train_fold), (X_test_fold, y_test_fold)))

        for original_test_data, (_, (X_test_fold, _)) in zip(original_validation_data, outer_splits):
            if not original_test_data.equals(X_test_fold.reset_index(drop=True)):
                raise ValueError("Validation folds' data are not consistent after applying sampling strategies.")

        return outer_splits, cv_folds_indices
