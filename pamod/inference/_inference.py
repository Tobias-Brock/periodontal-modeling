from typing import Dict, List, Tuple
import warnings

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy import stats
from sklearn import clone
from sklearn.exceptions import ConvergenceWarning

from pamod.base import BaseHydra, create_predict_data
from pamod.data import ProcessedDataLoader, StaticProcessEngine
from pamod.resampling import Resampler
from pamod.training import get_probs


class ModelInference(BaseHydra):
    def __init__(self, classification: str, model, verbosity: bool = True):
        """Initialize the ModelInference class with a trained model.

        Args:
            classification (str): Classification type ('binary' or 'multiclass').
            model: Trained classification model with a `predict_proba` method.
            verbosity (bool): Activates verbosity if set to True.
        """
        super().__init__()
        self.classification = classification
        self.model = model
        self.verbosity = verbosity

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a batch of input data.

        Args:
            input_data (pd.DataFrame): DataFrame containing feature values.

        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities for each class.
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
            pd.DataFrame: Data prepared for model inference.
        """
        if patient_data.empty:
            raise ValueError(
                "Patient data empty. Please submit data before running inference."
            )
        if self.verbosity:
            print("Patient Data Received for Inference:")
            print(patient_data)
        engine = StaticProcessEngine(behavior=False)
        dataloader = ProcessedDataLoader(task, encoding)
        patient_data[self.group_col] = "inference_patient"
        raw_data = engine.create_tooth_features(
            patient_data, neighbors=True, patient_id=False
        )

        if encoding == "target":
            raw_data = dataloader.encode_categorical_columns(raw_data)
            resampler = Resampler(self.classification, encoding)
            _, raw_data = resampler.apply_target_encoding(X_train, raw_data, y_train)

            encoded_fields = [
                "restoration",
                "periofamilyhistory",
                "diabetes",
                "toothtype",
                "toothside",
                "furcationbaseline",
                "smokingtype",
                "stresslvl",
            ]

            for key in raw_data.columns:
                if key not in encoded_fields and key in patient_data.columns:
                    raw_data[key] = patient_data[key].values
        else:
            raw_data = create_predict_data(raw_data, patient_data, encoding, self.model)

        predict_data = create_predict_data(raw_data, patient_data, encoding, self.model)
        predict_data = dataloader.scale_numeric_columns(predict_data)

        return predict_data, patient_data

    def patient_inference(
        self, predict_data: pd.DataFrame, patient_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run inference on the patient's data.

        Args:
            predict_data (pd.DataFrame): Transformed patient data for prediction.
            patient_data (pd.DataFrame): The patient's data as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing tooth, side, prediction, and probability.
        """
        results = self.predict(predict_data)
        output_data = patient_data[["tooth", "side"]].copy()
        output_data["prediction"] = results["prediction"]
        output_data["probability"] = results.drop(columns=["prediction"]).max(axis=1)
        return predict_data, output_data, results

    def jackknife_resampling(
        self,
        train_df: pd.DataFrame,
        patient_data: pd.DataFrame,
        encoding: str,
        model_params: dict,
        sample_fraction: float = 1.0,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Perform jackknife resampling with retraining for each patient.

        Args:
            train_df (pd.DataFrame): Full training dataset.
            patient_data (pd.DataFrame): The data for the patient(s) to predict on.
            encoding (str): Encoding type used ('one_hot' or 'target').
            model_params (dict): Parameters for the model initialization.
            sample_fraction (float, optional): Proportion of patient IDs to use for
                jackknife resampling. Defaults to 1.0.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.

        Returns:
            pd.DataFrame: DataFrame containing predictions for each iteration.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

        resampler = Resampler(self.classification, encoding)
        patient_ids = train_df[self.group_col].unique()

        if sample_fraction < 1.0:
            num_patients = int(len(patient_ids) * sample_fraction)
            rng = default_rng()
            patient_ids = rng.choice(patient_ids, num_patients, replace=False)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_patient)(
                patient_id, train_df, patient_data, encoding, model_params, resampler
            )
            for patient_id in patient_ids
        )

        jackknife_results = pd.concat(results, ignore_index=True)
        return jackknife_results

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
            pd.DataFrame: DataFrame containing patient predictions and probabilities.
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
                X_train, None, y_train, jackknife=True
            )
        else:
            X_train_enc = X_train.drop(columns=[self.group_col], errors="ignore")

        predictor = clone(self.model)
        predictor.set_params(**model_params)
        predictor.fit(X_train_enc, y_train)

        if self.classification == "binary" and hasattr(predictor, "best_threshold"):
            probs = get_probs(predictor, self.classification, patient_data)
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
        predictions_df["prediction"] = val_pred_classes
        predictions_df["iteration"] = patient_id
        predictions_df["data_index"] = patient_data.index

        return predictions_df

    def jackknife_confidence_intervals(
        self, jackknife_results: pd.DataFrame, alpha: float = 0.05
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Compute confidence intervals from jackknife results.

        Args:
            jackknife_results (pd.DataFrame): DataFrame with jackknife predictions.
            alpha (float, optional): Significance level for confidence intervals.
                Defaults to 0.05.

        Returns:
            Dict[int, Dict[str, Dict[str, float]]]: Confidence intervals for each data
            index and class.
        """
        ci_dict: Dict[int, Dict[str, Dict[str, float]]] = {}
        probability_columns = [
            col
            for col in jackknife_results.columns
            if col not in ["prediction", "iteration", "data_index"]
        ]
        grouped = jackknife_results.groupby("data_index")

        for data_idx, group in grouped:
            class_probs = group[probability_columns]
            mean_probs = class_probs.mean()
            se_probs = class_probs.std(ddof=1) / np.sqrt(len(class_probs))
            z_score = stats.norm.ppf(1 - alpha / 2)
            ci_lower = mean_probs - z_score * se_probs
            ci_upper = mean_probs + z_score * se_probs

            ci_dict[data_idx] = {}
            for class_name in class_probs.columns:
                ci_dict[data_idx][class_name] = {
                    "mean": mean_probs[class_name],
                    "lower": ci_lower[class_name],
                    "upper": ci_upper[class_name],
                }
        return ci_dict

    def plot_jackknife_intervals(
        self,
        ci_dict: Dict[int, Dict[str, Dict[str, float]]],
        data_indices: List[int],
        original_preds: pd.DataFrame,
    ) -> plt.Figure:
        """Plot Jackknife confidence intervals.

        Args:
            ci_dict (Dict[int, Dict[str, Dict[str, float]]]): Confidence intervals for
                each data index and class.
            data_indices (List[int]): List of data indices to plot.
            original_preds (pd.DataFrame): DataFrame containing original predictions and
                probabilities for each data point.

        Returns:
            plt.Figure: Figure object containing the plots, with one subplot per class.
        """
        classes = list(next(iter(ci_dict.values())).keys())
        num_classes = len(classes)
        ncols = num_classes
        nrows = 1

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6), sharey=True
        )
        axes = np.atleast_1d(axes).flatten()
        predicted_classes = original_preds["prediction"]

        for idx, class_name in enumerate(classes):
            ax = axes[idx]
            means = []
            lowers = []
            uppers = []
            data_indices_plot = []

            for data_index in data_indices:
                if predicted_classes.loc[data_index] == int(class_name):
                    ci = ci_dict[data_index][class_name]
                    mean = ci["mean"]
                    lower = ci["lower"]
                    upper = ci["upper"]
                    means.append(mean)
                    lowers.append(lower)
                    uppers.append(upper)
                    data_indices_plot.append(data_index)

            if means:
                errors = [
                    np.array(means) - np.array(lowers),
                    np.array(uppers) - np.array(means),
                ]

                ax.errorbar(
                    means,
                    data_indices_plot,
                    xerr=errors,
                    fmt="o",
                    color="skyblue",
                    ecolor="black",
                    capsize=5,
                    label="Jackknife CI",
                )

                orig_probs = original_preds.loc[data_indices_plot, class_name]
                ax.scatter(
                    orig_probs,
                    data_indices_plot,
                    color="red",
                    marker="x",
                    s=100,
                    label="Original Prediction",
                )

            ax.set_xlabel("Predicted Probability")
            if idx == 0:
                ax.set_ylabel("Data Point Index")
            ax.set_title(f"Class {class_name}")

            x_min = min(lowers) if lowers else 0
            x_max = max(uppers) if uppers else 1
            x_range = x_max - x_min
            if x_range == 0:
                x_range = 0.1
            ax.set_xlim([x_min - 0.1 * x_range, x_max + 0.1 * x_range])

            ax.legend()

        plt.tight_layout()
        return fig

    def jackknife_inference(
        self,
        model,
        train_df: pd.DataFrame,
        patient_data: pd.DataFrame,
        encoding: str,
        inference_results: pd.DataFrame,
        sample_fraction: float = 1.0,
        n_jobs: int = -1,
        max_plots: int = 12,
    ) -> Tuple[pd.DataFrame, plt.Figure]:
        """Run jackknife inference and generate confidence intervals and plots.

        Args:
            model: Dictionary of trained models.
            train_df (pd.DataFrame): Training DataFrame.
            patient_data (pd.DataFrame): Patient data to predict on.
            encoding (str): Encoding type.
            inference_results (pd.DataFrame): Original inference results.
            sample_fraction (float, optional): Fraction of patient IDs for jackknife.
                Defaults to 1.0.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
            max_plots (int): Maximum number of plots for jackknife intervals.

        Returns:
            Tuple[pd.DataFrame, plt.Figure]: Jackknife results and the plot.
        """
        model_params = model.get_params()

        if self.classification == "multiclass":
            num_classes = len(np.unique(train_df[self.y]))
            model_params["num_class"] = num_classes

        jackknife_results = self.jackknife_resampling(
            train_df=train_df,
            patient_data=patient_data,
            encoding=encoding,
            model_params=model_params,
            sample_fraction=sample_fraction,
            n_jobs=n_jobs,
        )
        ci_dict = self.jackknife_confidence_intervals(jackknife_results)
        data_indices = patient_data.index[:max_plots]
        ci_plot = self.plot_jackknife_intervals(
            ci_dict, data_indices, original_preds=inference_results
        )

        return jackknife_results, ci_plot
