from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import warnings

import pandas as pd
from sklearn import clone
from sklearn.exceptions import ConvergenceWarning

from ..base import BaseConfig
from ..data import ProcessedDataLoader, StaticProcessEngine
from ..resampling import Resampler
from ..training import get_probs


class BaseModelInference(BaseConfig, ABC):
    """Abstract base class for performing model inference and jackknife resampling.

    This class defines methods for generating predictions, preparing data for
    inference, and implementing jackknife resampling with confidence intervals.
    It is designed to handle binary and multiclass classification tasks and
    allows encoding configurations for model compatibility.

    Inherits:
        - `BaseConfig`: Provides configuration settings for data processing.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        classification (str): The type of classification task, either 'binary'
            or 'multiclass', used to configure the inference process.
        model: A trained model instance that implements a `predict_proba` method
            for generating class probabilities.
        verbose (bool): If True, enables detailed logging of inference steps.

    Attributes:
        classification (str): Stores the classification type ('binary' or 'multiclass')
            for model compatibility.
        model: The trained model used to make predictions during inference.
        verbose (bool): Indicates if verbose logging is enabled during inference.

    Methods:
        predict: Run predictions on a batch of input data, returning
            predicted classes and probabilities.
        create_predict_data: Prepare and encode data for inference based on raw data
            and patient data, supporting one-hot or target encoding formats.
        prepare_inference: Prepares data for inference, performing any
            necessary preprocessing and scaling.
        patient_inference: Runs predictions on specific patient data,
            returning results with predicted classes and probabilities.
        process_patient: Processes a patient’s data for jackknife resampling,
            retraining the model while excluding the patient from training.

    Abstract Methods:
        - `jackknife_resampling`: Performs jackknife resampling by retraining
          the model on various patient subsets.
        - `jackknife_confidence_intervals`: Computes confidence intervals
          based on jackknife resampling results.
        - `plot_jackknife_intervals`: Visualizes jackknife confidence intervals
          for predictions.
        - `jackknife_inference`: Executes full jackknife inference, including
          interval computation and optional plotting.
    """

    def __init__(self, classification: str, model: Any, verbose: bool):
        """Initialize the ModelInference class with a trained model."""
        super().__init__()
        self.classification = classification
        self.model = model
        self.verbose = verbose

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a batch of input data.

        Args:
            input_data (pd.DataFrame): DataFrame containing feature values.

        Returns:
            probs_df: DataFrame with predictions and probabilities for each class.
        """
        probs = self.model.predict_proba(input_data)

        if self.classification == "binary":
            if (
                hasattr(self.model, "best_threshold")
                and self.model.best_threshold is not None
            ):
                preds = (probs[:, 1] >= self.model.best_threshold).astype(int)
        preds = self.model.predict(input_data)
        classes = [str(cls) for cls in self.model.classes_]
        probs_df = pd.DataFrame(probs, columns=classes, index=input_data.index)
        probs_df["prediction"] = preds
        return probs_df

    def create_predict_data(
        self,
        raw_data: pd.DataFrame,
        patient_data: pd.DataFrame,
        encoding: str,
    ) -> pd.DataFrame:
        """Creates prediction data for model inference.

        Args:
            raw_data (pd.DataFrame): The raw, preprocessed data.
            patient_data (pd.DataFrame): Original patient data before preprocessing.
            encoding (str): Type of encoding used ('one_hot' or 'target').

        Returns:
            predict_data: A DataFrame containing the prepared data for model prediction.
        """
        base_data = raw_data.copy()

        if encoding == "one_hot":
            drop_columns = self.cat_vars + self.infect_vars
            base_data = base_data.drop(columns=drop_columns, errors="ignore")
            encoded_data = pd.DataFrame(index=base_data.index)

            for tooth_num in range(11, 49):
                if tooth_num % 10 == 0 or tooth_num % 10 == 9:
                    continue
                encoded_data[f"tooth_{tooth_num}"] = 0

            for feature, max_val in self.cat_map.items():
                for i in range(0, max_val + 1):
                    encoded_data[f"{feature}_{i}"] = 0

            for idx, row in patient_data.iterrows():
                encoded_data.at[idx, f"tooth_{row['tooth']}"] = 1
                for feature in self.cat_map:
                    encoded_data.at[idx, f"{feature}_{row[feature]}"] = 1

                complete_data = pd.concat(
                    [
                        base_data.reset_index(drop=True),
                        encoded_data.reset_index(drop=True),
                    ],
                    axis=1,
                )

            complete_data = complete_data.loc[:, ~complete_data.columns.duplicated()]
            duplicates = complete_data.columns[
                complete_data.columns.duplicated()
            ].unique()
            if len(duplicates) > 0:
                print("Duplicate columns found:", duplicates)

        elif encoding == "target":
            complete_data = base_data.copy()
            for column in self.target_cols:
                if column in patient_data.columns:
                    complete_data[column] = patient_data[column].values
        else:
            raise ValueError(f"Unsupported encoding type: {encoding}")

        if hasattr(self.model, "get_booster"):
            model_features = self.model.get_booster().feature_names
        elif hasattr(self.model, "feature_names_in_"):
            model_features = self.model.feature_names_in_
        else:
            raise ValueError("Model type not supported for feature extraction")

        for feature in model_features:
            if feature not in complete_data.columns:
                complete_data[feature] = 0

        predict_data = complete_data[model_features]

        return predict_data

    def prepare_inference(
        self,
        task: str,
        patient_data: pd.DataFrame,
        encoding: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepares the data for inference.

        Args:
            task (str): The task name for which the model was trained.
            patient_data (pd.DataFrame): The patient's data as a DataFrame.
            encoding (str): Encoding type ("one_hot" or "target").
            X_train (pd.DataFrame): Training features for target encoding.
            y_train (pd.Series): Training target for target encoding.

        Returns:
            Tuple: Transformed patient data for prediction and patient data.
        """
        if patient_data.empty:
            raise ValueError(
                "Patient data empty. Please submit data before running inference."
            )
        if self.verbose:
            print("Patient Data Received for Inference:\n", patient_data)

        engine = StaticProcessEngine()
        dataloader = ProcessedDataLoader(task, encoding)
        patient_data[self.group_col] = "inference_patient"
        raw_data = engine.create_tooth_features(
            df=patient_data, neighbors=True, patient_id=False
        )

        if encoding == "target":
            raw_data = dataloader.encode_categorical_columns(df=raw_data)
            resampler = Resampler(self.classification, encoding)
            _, raw_data = resampler.apply_target_encoding(
                X=X_train, X_val=raw_data, y=y_train
            )

            for key in raw_data.columns:
                if key not in self.cat_vars and key in patient_data.columns:
                    raw_data[key] = patient_data[key].values
        else:
            raw_data = self.create_predict_data(
                raw_data=raw_data, patient_data=patient_data, encoding=encoding
            )

        predict_data = self.create_predict_data(
            raw_data=raw_data, patient_data=patient_data, encoding=encoding
        )
        predict_data = dataloader.scale_numeric_columns(df=predict_data)

        return predict_data, patient_data

    def patient_inference(
        self, predict_data: pd.DataFrame, patient_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run inference on the patient's data.

        Args:
            predict_data (pd.DataFrame): Transformed patient data for prediction.
            patient_data (pd.DataFrame): The patient's data as a DataFrame.

        Returns:
            Tuple:
                - predict_data: Transformed patient data for prediction.
                - output_data: DataFrame with columns "tooth", "side",
                transformed "prediction", and "probability".
                - results: Original results from the model inference.
        """
        results = self.predict(predict_data)
        output_data = patient_data[["tooth", "side"]].copy()
        output_data["prediction"] = results["prediction"]
        output_data["probability"] = results.drop(columns=["prediction"]).max(axis=1)
        return predict_data, output_data, results

    def process_patient(
        self,
        patient_id: int,
        train_df: pd.DataFrame,
        patient_data: pd.DataFrame,
        encoding: str,
        model_params: dict,
        resampler: Resampler,
    ) -> pd.DataFrame:
        """Processes a single patient's data for jackknife resampling.

        Args:
            patient_id (int): ID of the patient to exclude from training.
            train_df (pd.DataFrame): Full training dataset.
            patient_data (pd.DataFrame): The data for the patient(s) to predict on.
            encoding (str): Encoding type used ('one_hot' or 'target').
            model_params (dict): Parameters for the model initialization.
            resampler (Resampler): Instance of the Resampler class for encoding.

        Returns:
            predictions_df: DataFrame containing patient predictions and probabilities.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

        train_data = train_df[train_df[self.group_col] != patient_id]
        X_train = train_data.drop(columns=[self.y])
        y_train = train_data[self.y]

        if encoding == "target":
            X_train = X_train.drop(columns=[self.group_col], errors="ignore")
            X_train_enc, _ = resampler.apply_target_encoding(
                X=X_train, X_val=None, y=y_train, jackknife=True
            )
        else:
            X_train_enc = X_train.drop(columns=[self.group_col], errors="ignore")

        predictor = clone(self.model)
        predictor.set_params(**model_params)
        predictor.fit(X_train_enc, y_train)

        if self.classification == "binary" and hasattr(predictor, "best_threshold"):
            probs = get_probs(
                model=predictor, classification=self.classification, X=patient_data
            )
            if probs is not None:
                val_pred_classes = (probs >= predictor.best_threshold).astype(int)
            else:
                val_pred_classes = predictor.predict(patient_data)
        else:
            val_pred_classes = predictor.predict(patient_data)
            probs = predictor.predict_proba(patient_data)

        predictions_df = pd.DataFrame(
            probs,
            columns=[str(cls) for cls in predictor.classes_],
            index=patient_data.index,
        )
        return predictions_df.assign(
            prediction=val_pred_classes,
            iteration=patient_id,
            data_index=patient_data.index,
        )

    @abstractmethod
    def jackknife_resampling(
        self,
        train_df: pd.DataFrame,
        patient_data: pd.DataFrame,
        encoding: str,
        model_params: dict,
        sample_fraction: float,
        n_jobs: int,
    ):
        """Perform jackknife resampling with retraining for each patient.

        Args:
            train_df (pd.DataFrame): Full training dataset.
            patient_data (pd.DataFrame): The data for the patient(s) to predict on.
            encoding (str): Encoding type used ('one_hot' or 'target').
            model_params (dict): Parameters for the model initialization.
            sample_fraction (float, optional): Proportion of patient IDs to use for
                jackknife resampling.
            n_jobs (int, optional): Number of jobs to run in parallel.
        """

    @abstractmethod
    def jackknife_confidence_intervals(
        self, jackknife_results: pd.DataFrame, alpha: float
    ):
        """Compute confidence intervals from jackknife results.

        Args:
            jackknife_results (pd.DataFrame): DataFrame with jackknife predictions.
            alpha (float, optional): Significance level for confidence intervals.
        """

    @abstractmethod
    def plot_jackknife_intervals(
        self,
        ci_dict: Dict[int, Dict[str, Dict[str, float]]],
        data_indices: List[int],
        original_preds: pd.DataFrame,
    ):
        """Plot Jackknife confidence intervals.

        Args:
            ci_dict (Dict[int, Dict[str, Dict[str, float]]]): Confidence intervals for
                each data index and class.
            data_indices (List[int]): List of data indices to plot.
            original_preds (pd.DataFrame): DataFrame containing original predictions and
                probabilities for each data point.
        """

    @abstractmethod
    def jackknife_inference(
        self,
        model: Any,
        train_df: pd.DataFrame,
        patient_data: pd.DataFrame,
        encoding: str,
        inference_results: pd.DataFrame,
        alpha: float,
        sample_fraction: float,
        n_jobs: int,
        max_plots: int,
    ):
        """Run jackknife inference and generate confidence intervals and plots.

        Args:
            model (Any): Trained model instance.
            train_df (pd.DataFrame): Training DataFrame.
            patient_data (pd.DataFrame): Patient data to predict on.
            encoding (str): Encoding type.
            inference_results (pd.DataFrame): Original inference results.
            alpha (float, optional): Significance level for confidence intervals.
            sample_fraction (float, optional): Fraction of patient IDs for jackknife.
            n_jobs (int, optional): Number of parallel jobs.
            max_plots (int): Maximum number of plots for jackknife intervals.
        """
