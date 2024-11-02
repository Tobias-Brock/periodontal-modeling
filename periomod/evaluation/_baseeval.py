from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type, Union

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from ..training import get_probs


def _is_one_hot_encoded(feature: str) -> bool:
    """Check if a feature is one-hot encoded.

    Args:
        feature (str): The name of the feature to check.

    Returns:
        bool: True if the feature is one-hot encoded.
    """
    parts = feature.rsplit("_", 1)
    return len(parts) > 1 and (parts[1].isdigit())


def _get_base_name(feature: str) -> str:
    """Extract the base name of a feature.

    Args:
        feature (str): The name of the feature to process.

    Returns:
        str: The base name of the feature.
    """
    if _is_one_hot_encoded(feature=feature):
        return feature.rsplit("_", 1)[0]
    return feature


class BaseModelEvaluator(ABC):
    """Abstract base class for evaluating machine learning model performance.

    This class provides methods for calculating model performance metrics,
    plotting confusion matrices, and evaluating feature importance, with options
    for handling one-hot encoded features and aggregating SHAP values.

    Inherits:
        - ABC: Specifies abstract methods for subclasses to implement.

    Args:
        X (pd.DataFrame): Test dataset features.
        y (pd.Series): Test dataset labels.
        model (Union[sklearn estimators, None]): A trained sklearn model instance.
        models (Union[List[sklearn estimators], None]): List of trained models.
        encoding (Optional[str]): Encoding type for plot titles (e.g., 'one_hot'
            or 'target').
        aggregate (bool): If True, aggregates importance values of one-hot encoded
            features.

    Attributes:
        X (pd.DataFrame): Stores the test dataset features for evaluation.
        y (pd.Series): Stores the test dataset labels for evaluation.
        model (Union[sklearn estimators, None]): The primary model for evaluation.
        models (List[sklearn estimators]): List of trained models for multi-model
            evaluation.
        encoding (Optional[str]): The encoding type used, impacting plot titles
            and feature grouping.
        aggregate (bool): Determines if importance values of one-hot encoded
            features are aggregated for interpretability.

    Methods:
        brier_score_groups: Calculates Brier score within specified groups.
        plot_confusion_matrix: Generates a styled confusion matrix heatmap
            for the test data and model predictions.
        evaluate_feature_importance: Abstract method for evaluating feature
            importance across models.
        analyze_brier_within_clusters: Abstract method for analyzing Brier
            score distribution within clusters.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Union[
            RandomForestClassifier,
            LogisticRegression,
            MLPClassifier,
            XGBClassifier,
            None,
        ],
        models: Union[
            List[
                Union[
                    RandomForestClassifier,
                    LogisticRegression,
                    MLPClassifier,
                    XGBClassifier,
                ]
            ],
            None,
        ],
        encoding: Optional[str],
        aggregate: bool,
    ) -> None:
        """Initialize the FeatureImportance class.

        Args:
            X (pd.DataFrame): Test dataset features.
            y (pd.Series): Test dataset labels.
            model ([sklearn estimators]): Trained sklearn models.
            models (List[sklearn estimators]): List of trained models.
            encoding (Optional[str]): Determines encoding for plot titles
                ('one_hot' or 'target').
            aggregate (bool): If True, aggregates importance values of one-hot features.
        """
        self.X = X
        self.y = y
        self.model = model
        self.models = models if models is not None else []
        self.encoding = encoding
        self.aggregate = aggregate

    def brier_score_groups(self, group_by: str = "y") -> None:
        """Calculates Brier score within groups.

        Args:
            group_by (str): Grouping variable. Defaults to "y".

        Returns:
            pd.DataFrame: DataFrame containing group variable, label, and Brier scores.
        """
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("The provided model cannot predict probabilities.")

        if self.model is None:
            return None

        probas = self.model.predict_proba(self.X)

        if len(probas[0]) == 1:
            brier_scores = [
                brier_score_loss(y_true=[true_label], y_proba=[pred_proba[0]])
                for true_label, pred_proba in zip(self.y, probas, strict=False)
            ]
        else:
            brier_scores = [
                brier_score_loss(
                    [1 if true_label == idx else 0 for idx in range(len(proba))], proba
                )
                for true_label, proba in zip(self.y, probas, strict=False)
            ]

            data = pd.DataFrame({group_by: self.y, "Brier_Score": brier_scores})
            data_grouped = data.groupby(group_by)

        summary = data_grouped["Brier_Score"].agg(["mean", "median"]).reset_index()
        print(f"Average and Median Brier Scores by {group_by}:\n{summary}")

        plt.figure(figsize=(10, 6), dpi=300)
        sns.boxplot(x=group_by, y="Brier_Score", data=data)
        plt.title("Distribution of Brier Scores", fontsize=18)
        plt.xlabel(f'{"y" if group_by == "y" else group_by}', fontsize=18)
        plt.ylabel("Brier Score", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_confusion_matrix(
        self,
        col: Optional[pd.Series] = None,
        y_label: str = "True",
        normalize: str = "rows",
    ):
        """Generates a styled confusion matrix for the given model and test data.

        Args:
            col (Optional[pd.Series]): Column for y label. Defaults to None.
            y_label (str): Description of y label. Defaults to "True".
            normalize (str, optional): Normalization method ('rows' or 'columns').
                Defaults to 'rows'.

        Returns:
        plt.Figure: Confusion matrix heatmap plot.
        """
        if not self.model:
            return "No model available."
        if (
            hasattr(self.model, "best_threshold")
            and self.model.best_threshold is not None
        ):
            final_probs = get_probs(model=self.model, classification="binary", X=self.X)
            if final_probs is not None:
                pred = (final_probs >= self.model.best_threshold).astype(int)
            else:
                pred = self.model.predict(self.X)
        else:
            pred = self.model.predict(self.X)

        if col is not None:
            cm = confusion_matrix(y_true=col, y_pred=pred)
        else:
            cm = confusion_matrix(y_true=self.y, y_pred=pred)

        if normalize == "rows":
            row_sums = cm.sum(axis=1)
            normalized_cm = (cm / row_sums[:, np.newaxis]) * 100
        elif normalize == "columns":
            col_sums = cm.sum(axis=0)
            normalized_cm = (cm / col_sums) * 100
        else:
            raise ValueError("Invalid value for 'normalize'. Use 'rows' or 'columns'.")

        custom_cmap = LinearSegmentedColormap.from_list(
            "teal_cmap", ["#FFFFFF", "#078294"]
        )

        plt.figure(figsize=(6, 4), dpi=300)
        sns.heatmap(
            normalized_cm,
            cmap=custom_cmap,
            fmt="g",
            linewidths=0.5,
            square=True,
            annot=False,
            cbar_kws={"label": "Percent"},
        )

        [
            plt.text(
                j + 0.5,
                i + 0.5,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if normalized_cm[i, j] > 50 else "black",
            )
            for i in range(len(cm))
            for j in range(len(cm))
        ]

        plt.title("Confusion Matrix", fontsize=12)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        ax = plt.gca()

        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        cbar = ax.collections[0].colorbar
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1)

        ax.add_patch(
            Rectangle(
                (0, 0), cm.shape[1], cm.shape[0], fill=False, edgecolor="black", lw=2
            )
        )

        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.show()

    @staticmethod
    def _aggregate_one_hot_importances(
        fi_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate importance scores of one-hot encoded variables.

        Args:
            fi_df (pd.DataFrame): DataFrame with features and their
                importance scores.

        Returns:
            pd.DataFrame: Updated DataFrame with aggregated importance scores.
        """
        base_names = fi_df["Feature"].apply(_get_base_name)
        aggregated_importances = (
            fi_df.groupby(base_names)["Importance"].sum().reset_index()
        )
        aggregated_importances.columns = ["Feature", "Importance"]
        original_features = fi_df["Feature"][
            ~fi_df["Feature"].apply(_is_one_hot_encoded)
        ].unique()

        aggregated_or_original = (
            pd.concat(
                [
                    aggregated_importances,
                    fi_df[fi_df["Feature"].isin(original_features)],
                ]
            )
            .drop_duplicates()
            .sort_values(by="Importance", ascending=False)
        )

        return aggregated_or_original.reset_index(drop=True)

    @staticmethod
    def _aggregate_shap_one_hot(
        shap_values: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Aggregate SHAP values of one-hot encoded variables.

        Args:
            shap_values (np.ndarray): SHAP values.
            feature_names (List[str]): List of features corresponding to SHAP values.

        Returns:
            Tuple[np.ndarray, List[str]]: Aggregated SHAP values and updated list of
            feature names.
        """
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=2)

        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        base_names = [_get_base_name(feature=feature) for feature in shap_df.columns]
        feature_mapping = dict(zip(shap_df.columns, base_names, strict=False))
        aggregated_shap_df = shap_df.groupby(feature_mapping, axis=1).sum()
        updated_feature_names = list(aggregated_shap_df.columns)
        aggregated_shap_values = aggregated_shap_df.values

        return aggregated_shap_values, updated_feature_names

    @staticmethod
    def _aggregate_one_hot_features_for_clustering(X: pd.DataFrame) -> pd.DataFrame:
        """Aggregate one-hot encoded features for clustering.

        Args:
            X (pd.DataFrame): Input DataFrame with one-hot encoded features.

        Returns:
            pd.DataFrame: DataFrame with aggregated one-hot encoded features.
        """
        X_copy = X.copy()
        one_hot_encoded_cols = [
            col for col in X_copy.columns if _is_one_hot_encoded(feature=col)
        ]
        base_names = {col: _get_base_name(feature=col) for col in one_hot_encoded_cols}
        aggregated_data = X_copy.groupby(base_names, axis=1).sum()
        non_one_hot_cols = [
            col for col in X_copy.columns if col not in one_hot_encoded_cols
        ]
        X_aggregated = pd.concat([X_copy[non_one_hot_cols], aggregated_data], axis=1)

        return X_aggregated

    @abstractmethod
    def evaluate_feature_importance(self, importance_types: List[str]):
        """Evaluate the feature importance for a list of trained models.

        Args:
            importance_types (List[str]): Methods of feature importance evaluation:
                'shap', 'permutation', 'standard'.
        """

    @abstractmethod
    def analyze_brier_within_clusters(
        self,
        clustering_algorithm: Type,
        n_clusters: int,
    ):
        """Analyze distribution of Brier scores within clusters formed by input data.

        Args:
            clustering_algorithm (Type): Clustering algorithm class from sklearn to use
                for clustering.
            n_clusters (int): Number of clusters to form.
        """