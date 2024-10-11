from typing import Dict, List, Optional

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy import stats
from sklearn import clone

from pamod.base import BaseHydra
from pamod.resampling import Resampler


class ModelInference(BaseHydra):
    def __init__(self, classification: str, model):
        """Initialize the ModelInference class with a trained model.

        Args:
            classification (str): Classification type ('binary' or 'multiclass').
            model: Trained classification model with a `predict_proba` method.
        """
        super().__init__()
        self.classification = classification
        self.model = model

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a batch of input data.

        Args:
            input_data (pd.DataFrame): DataFrame containing feature values.

        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities for each class.
        """
        preds = self.model.predict(input_data)
        probs = self.model.predict_proba(input_data)
        classes = [str(cls) for cls in self.model.classes_]
        probs_df = pd.DataFrame(probs, columns=classes, index=input_data.index)
        probs_df["prediction"] = preds
        return probs_df

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
        resampler = Resampler(self.classification, encoding)
        patient_ids = train_df[self.group_col].unique()

        if sample_fraction < 1.0:
            num_patients = int(len(patient_ids) * sample_fraction)
            rng = default_rng()
            patient_ids = rng.choice(patient_ids, num_patients, replace=False)

        def process_patient(patient_id):
            train_data = train_df[train_df[self.group_col] != patient_id]
            X_train = train_data.drop(columns=[self.target])
            y_train = train_data[self.target]

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

            val_predictions = predictor.predict_proba(patient_data)
            val_pred_classes = predictor.predict(patient_data)

            predictions_df = pd.DataFrame(
                val_predictions,
                columns=[str(cls) for cls in predictor.classes_],
                index=patient_data.index,
            )
            predictions_df["prediction"] = val_pred_classes
            predictions_df["iteration"] = patient_id
            predictions_df["data_index"] = patient_data.index  # Include data index
            return predictions_df

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_patient)(patient_id) for patient_id in patient_ids
        )

        jackknife_results = pd.concat(results, ignore_index=True)
        return jackknife_results

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
        original_preds: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> plt.Figure:
        """Plot Jackknife confidence intervals for multiple data points as subplots.

        Args:
            ci_dict (Dict[int, Dict[str, Dict[str, float]]]): Confidence intervals for
                each data index and class.
            data_indices (List[int]): List of data indices to plot.
            original_preds (Optional[Dict[int, Dict[str, float]]], optional):
                Original predictions per data index. Defaults to None.

        Returns:
            plt.Figure: Figure object containing the plots.
        """
        classes = list(next(iter(ci_dict.values())).keys())
        num_classes = len(classes)

        fig, axes = plt.subplots(
            nrows=1, ncols=num_classes, figsize=(6 * num_classes, 6), sharey=True
        )

        if num_classes == 1:
            axes = [axes]  # Ensure axes is iterable

        for idx, class_name in enumerate(classes):
            ax = axes[idx]
            means = []
            lowers = []
            uppers = []
            data_indices_plot = []

            for data_index in data_indices:
                ci = ci_dict[data_index][class_name]
                mean = ci["mean"]
                lower = ci["lower"]
                upper = ci["upper"]
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)
                data_indices_plot.append(str(data_index))

            errors = [
                np.array(means) - np.array(lowers),
                np.array(uppers) - np.array(means),
            ]

            ax.barh(
                data_indices_plot,
                means,
                xerr=errors,
                align="center",
                alpha=0.7,
                ecolor="black",
                capsize=5,
                color="skyblue",
                label="Jackknife CI",
            )

            if original_preds is not None:
                orig_probs = [
                    original_preds[data_index].get(class_name, 0)
                    for data_index in data_indices
                ]
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

            x_min = min(min(lowers), min(orig_probs) if original_preds else 0)
            x_max = max(max(uppers), max(orig_probs) if original_preds else 1)
            x_range = x_max - x_min
            if x_range == 0:
                x_range = 0.1  # Avoid zero range
            ax.set_xlim([x_min - 0.1 * x_range, x_max + 0.1 * x_range])

            ax.legend()

        plt.tight_layout()
        return fig
