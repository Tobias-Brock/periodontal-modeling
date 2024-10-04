from typing import Tuple

import jackknife as jk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ModelInference:
    def __init__(self, model):
        """Initialize the ModelInference class with a trained model.

        Args:
            model: Trained classification model with a `predict_proba` method.
        """
        self.model = model

    def predict(self, input_data: dict) -> Tuple[str, float]:
        """Run prediction on a single input instance.

        Args:
            input_data (dict): Dictionary of feature values for the model.

        Returns:
            Tuple[str, float]: Prediction and probability result for single input.
        """
        probability = self.model.predict_proba(input_data)[0]  # Get probabilities
        prediction = self.model.predict(input_data)[0]  # Perform prediction
        predicted_probability = probability[np.argmax(probability)]

        return str(prediction), predicted_probability

    def jackknife_confidence_intervals(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> dict:
        """Perform Jackknife resampling to compute confidence intervals.

        Args:
            X (pd.DataFrame): Input data for inference.
            alpha (float): Significance level for confidence intervals.

        Returns:
            dict: Jackknife confidence intervals for each class.
        """
        predictions = self.model.predict_proba(X)

        theta_subsample, theta_fullsample = jk.jk_loop(
            lambda x: x.mean(axis=0), predictions
        )
        _, _, theta_jack, se_jack, _ = jk.jk_stats(theta_subsample, theta_fullsample)

        ci_lower = theta_jack - 1.96 * se_jack  # Assuming normal distribution
        ci_upper = theta_jack + 1.96 * se_jack

        ci_dict = {}
        for class_idx in range(predictions.shape[1]):
            ci_dict[f"Class {class_idx}"] = {
                "lower": ci_lower[:, class_idx],
                "upper": ci_upper[:, class_idx],
            }

        return ci_dict

    def plot_jackknife_intervals(self, ci_dict: dict) -> plt.Figure:
        """Plot Jackknife confidence intervals.

        Args:
            ci_dict (dict): Jackknife confidence intervals for each class.

        Returns:
            plt.Figure: Figure object containing the plot.
        """
        plt.figure(figsize=(10, 6))
        for class_name, ci in ci_dict.items():
            plt.fill_between(
                range(len(ci["lower"])),
                ci["lower"],
                ci["upper"],
                alpha=0.2,
                label=class_name,
            )
        plt.title("Jackknife Confidence Intervals for Predictions")
        plt.xlabel("Data Points")
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
