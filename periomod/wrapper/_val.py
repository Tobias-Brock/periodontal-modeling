from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..data import ProcessedDataLoader
from ..resampling import Resampler
from ..training import final_metrics, get_probs
from ..wrapper import ModelExtractor


class Validator(ModelExtractor):
    """Validator class for evaluating trained models on a separate validation dataset.

    This class loads a validation dataset, applies necessary transformations,
    handles encoding, and evaluates a pre-trained model.

    Inherits:
        - `ModelExtractor`: Base class for extracting trained models and evaluation.

    Args:
        learners_dict (Dict): Dictionary containing trained models.
        criterion (str): Performance criterion for evaluation
            (e.g., "f1", "brier_score").
        aggregate (bool): Whether to aggregate results across multiple models.
        path_train (Path): Path to the training dataset used for encoding reference.
        path_val (Path): Path to the validation dataset.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.
        random_state (Optional[int], optional): Random seed for reproducibility.
            Defaults to 0.
        test_size (float, optional): Proportion of the dataset to use as a test set when
            performing target encoding. Defaults to 0.2.

    Attributes:
        dataloader (ProcessedDataLoader): Data loader for preprocessing input data.
        resampler (Resampler): Resampler for handling encoding and dataset splitting.
        path_train (Path): Path to the training dataset.
        path_val (Path): Path to the validation dataset.
        test_size (float): Proportion of training data used for validation when
            encoding.
        data (pd.DataFrame): Raw validation dataset.
        data_processed (pd.DataFrame): Processed validation dataset.
        X (pd.DataFrame): Features from the validation dataset.
        y (pd.Series): Target labels from the validation dataset.

    Methods:
        _prepare_validation_data: Loads, processes, and encodes validation data to match
            training features.
        perform_validation: Runs model evaluation on the validation dataset, returning
            performance metrics.

    Inherited Methods:
        - `load_learners`: Loads trained models from a specified directory.
        - `apply_target_encoding`: Applies target encoding to categorical variables.
        - `apply_sampling`: Applies specified sampling strategy to balance the dataset.
        - `validate_dataframe`: Validates that input data meets requirements.
        - `get_probs`: Computes model prediction probabilities.
        - `final_metrics`: Computes final evaluation metrics based on task.

    Example:
        ```
        from periomod.wrapper import Validator

        validator = Validator(
            learners_dict=learners,
            criterion="f1",
            aggregate=True,
            path_train=Path("../data/processed/processed_data.csv"),
            path_val=Path("../data/processed/processed_data_val.csv"),
            random_state=42,
            test_size=0.2
        )

        results = validator.perform_validation(verbose=True)
        ```
    """

    def __init__(
        self,
        learners_dict: Dict,
        criterion: str,
        aggregate: bool,
        path_train: Path,
        path_val: Path,
        verbose: bool = False,
        random_state: int = 0,
        test_size: float = 0.2,
    ):
        """Initializes the Validator class.

        Args:
            learners_dict (Dict): Dictionary containing trained models.
            criterion (str): Performance criterion for evaluation
                (e.g., "f1", "brier_score").
            aggregate (bool): Whether to aggregate results across multiple models.
            path_train (Path): Path to the training dataset used for encoding reference.
            path_val (Path): Path to the validation dataset.
            verbose (bool, optional): Whether to print detailed logs. Defaults to False.
            random_state (Optional[int], optional): Random seed for reproducibility.
                Defaults to 0.
            test_size (float, optional): Proportion of the dataset to use as a test set
                when performing target encoding. Defaults to 0.2.
        """
        super().__init__(
            learners_dict=learners_dict,
            criterion=criterion,
            aggregate=aggregate,
            verbose=verbose,
            random_state=random_state,
        )
        self.dataloader = ProcessedDataLoader(task=self.task, encoding=self.encoding)
        self.resampler = Resampler(
            classification=self.classification, encoding=self.encoding
        )
        self.path_train = path_train
        self.path_val = path_val
        self.test_size = test_size
        self.data, self.data_processed, self.X, self.y = self._prepare_validation_data()

    def _prepare_validation_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Loads and prepares validation data for model evaluation.

        This function loads the dataset, applies necessary transformations,
        and encodes categorical variables if required.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
                - data (pd.DataFrame): Raw validation dataset.
                - data_processed (pd.DataFrame): Preprocessed dataset.
                - X (pd.DataFrame): Feature matrix.
                - y (pd.Series): Target labels.

        Raises:
            ValueError: If model type is not supported for extraction.
        """
        data_val = self.dataloader.load_data(path=self.path_val)
        data_processed = self.dataloader.transform_data(data=data_val)

        X_val = data_processed.drop(columns=[self.y])
        y = data_processed[self.y]

        if self.encoding == "target":
            data_train = self.dataloader.load_data(path=self.path_train)
            data_train = self.dataloader.transform_data(data=data_train)
            train_df, test_df = self.resampler.split_train_test_df(
                df=data_train, seed=self.random_state, test_size=self.test_size
            )

            X_train, y_train, _, _ = self.resampler.split_x_y(
                train_df=train_df, test_df=test_df
            )

            X_train, X_val = self.resampler.apply_target_encoding(
                X=X_train, X_val=X_val, y=y_train
            )

        if self.encoding == "one_hot":
            if hasattr(self.model, "get_booster"):
                expected_columns = set(self.model.get_booster().feature_names)
            elif hasattr(self.model, "feature_names_in_"):
                expected_columns = set(self.model.feature_names_in_)
            else:
                raise ValueError("Model type not supported for feature extraction")
            current_columns = set(X_val.columns)
            missing_columns = expected_columns - current_columns
            for col in missing_columns:
                X_val[col] = 0

            X_val = X_val[list(self.model.feature_names_in_)]

        if self.group_col in X_val:
            X_val = X_val.drop(columns=[self.group_col])

        return (
            data_val,
            data_processed,
            X_val,
            y,
        )

    def perform_validation(self, verbose: bool = False) -> pd.DataFrame:
        """Runs model evaluation on the validation dataset.

        This function computes predictions and evaluates the model's performance
        using predefined metrics such as F1-score or other classification metrics.

        Args:
            verbose (bool, optional): Whether to print detailed evaluation metrics.
                Defaults to False.

        Returns:
            pd.DataFrane: A dataframe containing computed evaluation metrics.
        """
        best_threshold = getattr(self.model, "best_threshold", None)

        final_probs = get_probs(
            model=self.model, classification=self.classification, X=self.X
        )

        if (
            self.criterion == "f1"
            and final_probs is not None
            and np.any(final_probs)
            and best_threshold is not None
        ):
            final_predictions = (final_probs >= best_threshold).astype(int)
        else:
            final_predictions = self.model.predict(self.X)

        metrics = final_metrics(
            classification=self.classification,
            y=self.y,
            preds=final_predictions,
            probs=final_probs,
            threshold=best_threshold,
        )
        unpacked_metrics = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
        }
        results = {
            "Learner": self.learner,
            "Tuning": "final",
            "Criterion": self.criterion,
            **unpacked_metrics,
        }

        df_results = pd.DataFrame([results])
        if verbose:
            pd.set_option("display.max_columns", None, "display.width", 1000)
            print("\nFinal Model Metrics Summary:\n", df_results)
        return df_results
