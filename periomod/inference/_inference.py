from typing import Any, Dict, List, Tuple

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy import stats

from ..resampling import Resampler
from ._baseinference import BaseModelInference


class ModelInference(BaseModelInference):
    """Performs model inference and jackknife resampling on patient data.

    This class extends `BaseModelInference` with specific implementations for
    jackknife resampling, confidence interval computation, and visualization of
    prediction intervals for binary and multiclass classification models. It
    incorporates methods for generating predictions, preparing data for model
    inference, and applying jackknife inference, thus enabling robust model
    evaluation with confidence bounds.

    Inherits:
        - `BaseModelInference`: Base class that implements prediction and
            preprocessing methods.

    Args:
        classification (str): Specifies the classification type ('binary' or
            'multiclass').
        model: A trained model instance compatible with scikit-learn's
            `predict_proba` method.
        verbose (bool): Enables detailed logging if set to True.

    Methods:
        jackknife_resampling: Re-trains the model on subsets of data,
            excluding each patient iteratively to compute jackknife estimates.
        jackknife_confidence_intervals: Calculates confidence intervals based on
            jackknife results, returning bounds for each data index and class.
        plot_jackknife_intervals: Visualizes jackknife confidence intervals for specific
            data points. Displays the estimated intervals and original predictions.
        jackknife_inference: Runs the complete jackknife inference
            workflow, generating confidence intervals and an optional plot to
            illustrate interval bounds across specified data points.

    Inherited Methods:
        - `predict`: Runs predictions on a batch of input data, returning
          probabilities and predicted classes.
        - `create_predict_data`: Encodes and prepares raw patient data for
          model prediction.
        - `prepare_inference`: Prepares data for inference by processing and
          encoding patient data.
        - `patient_inference`: Generates prediction probabilities for
          a specified patient's data.
        - `process_patient`: Excludes data for each patient iteratively and
          retrains the model for jackknife resampling.

    Example:
        ```
        from periomod.base import Patient, patient_to_dataframe
        from periomod.inference import ModelInference

        model_inference = ModelInference(
            classification="binary", model=trained_model, verbose=True
        )

        # Define a patient instance
        patient = Patient()
        patient_df = patient_to_df(patient=patient)

        # Prepare data for inference
        prepared_data, patient_data = model_inference.prepare_inference(
            task="pocketclosure",
            patient_data=patient_df,
            encoding="one_hot",
            X_train=X_train,
            y_train=y_train,
        )

        # Run inference on patient data
        inference_results = model_inference.patient_inference(
            predict_data=prepared_data, patient_data=patient_data
        )

        # Perform jackknife inference with confidence interval plotting
        jackknife_results, ci_plot = model_inference.jackknife_inference(
            model=trained_model,
            train_df=train_df,
            patient_data=patient_df,
            encoding="target",
            inference_results=inference_results,
            alpha=0.05,
            sample_fraction=0.8,
            n_jobs=4,
        )
        ```
    """

    def __init__(self, classification: str, model: Any, verbose: bool = True):
        """Initialize the ModelInference class with a trained model."""
        super().__init__(classification=classification, model=model, verbose=verbose)

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
            DataFrame: DataFrame containing predictions for each iteration.
        """
        resampler = Resampler(classification=self.classification, encoding=encoding)
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

        return pd.concat(results, ignore_index=True)

    def jackknife_confidence_intervals(
        self, jackknife_results: pd.DataFrame, alpha: float = 0.05
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Compute confidence intervals from jackknife results.

        Args:
            jackknife_results (pd.DataFrame): DataFrame with jackknife predictions.
            alpha (float, optional): Significance level for confidence intervals.
                Defaults to 0.05.

        Returns:
            Dict: Confidence intervals for each data index and class.
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
            Figure: Figure object containing the plots, with one subplot per class.
        """
        classes = list(next(iter(ci_dict.values())).keys())
        num_classes = len(classes)
        ncols = num_classes
        nrows = 1

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6), sharey=True, dpi=300
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
        model: Any,
        train_df: pd.DataFrame,
        patient_data: pd.DataFrame,
        encoding: str,
        inference_results: pd.DataFrame,
        alpha: float = 0.05,
        sample_fraction: float = 1.0,
        n_jobs: int = -1,
        max_plots: int = 12,
    ) -> Tuple[pd.DataFrame, plt.Figure]:
        """Run jackknife inference and generate confidence intervals and plots.

        Args:
            model (Any): Trained model instance.
            train_df (pd.DataFrame): Training DataFrame.
            patient_data (pd.DataFrame): Patient data to predict on.
            encoding (str): Encoding type.
            inference_results (pd.DataFrame): Original inference results.
            alpha (float, optional): Significance level for confidence intervals.
                Defaults to 0.05.
            sample_fraction (float, optional): Fraction of patient IDs for jackknife.
                Defaults to 1.0.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
            max_plots (int): Maximum number of plots for jackknife intervals.

        Returns:
            Tuple: Jackknife results and the plot.
        """
        model_params = model.get_params()

        if self.classification == "multiclass":
            num_classes = len(np.unique(train_df[self.y]))
            if "num_class" in model.get_params().keys():
                model_params["num_class"] = num_classes

        jackknife_results = self.jackknife_resampling(
            train_df=train_df,
            patient_data=patient_data,
            encoding=encoding,
            model_params=model_params,
            sample_fraction=sample_fraction,
            n_jobs=n_jobs,
        )
        ci_dict = self.jackknife_confidence_intervals(
            jackknife_results=jackknife_results, alpha=alpha
        )
        data_indices = patient_data.index[:max_plots]
        ci_plot = self.plot_jackknife_intervals(
            ci_dict=ci_dict, data_indices=data_indices, original_preds=inference_results
        )

        return jackknife_results, ci_plot
