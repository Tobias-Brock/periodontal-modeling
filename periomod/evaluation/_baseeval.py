from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from ..base import BaseConfig
from ..training import brier_loss_multi, get_probs


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


def _label_mapping(task: Optional[str]) -> dict:
    """Returns a label mapping dictionary based on the provided task.

    Args:
        task (Optional[str]): Task name for which the label mapping is required.

    Returns:
        dict: Dictionary with label mappings, or None if no mapping is needed.
    """
    if task in ["pocketclosure", "pocketclosureinf"]:
        return {1: "Closed", 0: "Not closed"}
    elif task == "improvement":
        return {1: "Improved", 0: "Not improved"}
    elif task == "pdgrouprevaluation":
        return {0: "< 4 mm", 1: "4 or 5 mm", 2: "> 5 mm"}
    else:
        raise ValueError(f"Task: {task} invalid.")


task_map = {
    "pocketclosure": "Pocket closure",
    "pocketclosureinf": "Pocket closure PdBaseline > 3",
    "improvement": "Pocket improvement",
    "pdgrouprevaluation": "Pocket groups",
}


class EvaluatorMethods(BaseConfig):
    """Base class that contains methods for ModelEvalutor.

    Inherits:
        - `BaseConfig`: Provides base configuration settings.

    Args:
        X (pd.DataFrame): The test dataset features.
        y (pd.Series): The test dataset labels.
        model (sklearn.base.BaseEstimator): A trained sklearn model instance
            for single-model evaluation.
        encoding (Optional[str]): Encoding type for categorical features, e.g.,
            'one_hot' or 'target', used for labeling and grouping in plots.
        aggregate (bool): If True, aggregates the importance values of multi-category
            encoded features for interpretability.

    Attributes:
        X (pd.DataFrame): Holds the test dataset features for evaluation.
        y (pd.Series): Holds the test dataset labels for evaluation.
        model (Optional[sklearn.base.BaseEstimator]): The primary model instance used
            for evaluation, if single-model evaluation is performed.
        encoding (Optional[str]): Indicates the encoding type used, which impacts
            plot titles and feature grouping in evaluations.
        aggregate (bool): Indicates whether to aggregate importance values of
            multi-category encoded features, enhancing interpretability in feature
            importance plots.

    Methods:
        brier_scores: Calculates Brier score for each instance in the evaluator's
            dataset based on the model's predicted probabilities. Returns series of
            Brier scores indexed by instance.
        model_predictions: Generates model predictions for evaluator's feature
            set, applying threshold-based binarization if specified, and returns
            predictions as a series indexed by instance.
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
        ],
        encoding: Optional[str],
        aggregate: bool,
    ) -> None:
        """Initialize the FeatureImportance class."""
        super().__init__()
        self.X = X
        self.y = y
        self.model = model
        self.encoding = encoding
        self.aggregate = aggregate

    def brier_scores(self) -> pd.Series:
        """Calculates Brier scores for each instance in the evaluator's dataset.

        Returns:
            Series: Brier scores for each instance.
        """
        probas = self.model.predict_proba(self.X)

        if probas.shape[1] == 1:
            brier_scores = [
                brier_score_loss([true_label], [pred_proba[0]])
                for true_label, pred_proba in zip(self.y, probas, strict=False)
            ]
        else:
            brier_scores = [
                brier_score_loss(
                    [1 if true_label == idx else 0 for idx in range(len(proba))], proba
                )
                for true_label, proba in zip(self.y, probas, strict=False)
            ]

        return pd.Series(brier_scores, index=self.y.index)

    def model_predictions(self) -> pd.Series:
        """Generates model predictions for the evaluator's feature set.

        Returns:
            pred: Predicted labels as a series.
        """
        if (
            hasattr(self.model, "best_threshold")
            and self.model.best_threshold is not None
        ):
            final_probs = get_probs(model=self.model, classification="binary", X=self.X)
            if final_probs is not None:
                pred = pd.Series(
                    (final_probs >= self.model.best_threshold).astype(int),
                    index=self.X.index,
                )
            else:
                pred = pd.Series(self.model.predict(self.X), index=self.X.index)
        else:
            pred = pd.Series(self.model.predict(self.X), index=self.X.index)

        return pred

    def _feature_mapping(self, features: List[str]) -> List[str]:
        """Maps a list of feature names to their original labels.

        Args:
            features (List[str]): List of feature names to be mapped.

        Returns:
            List[str]: List of mapped feature names, with original labels applied
                where available.
        """
        return [self.feature_mapping.get(feature, feature) for feature in features]

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
        return aggregated_shap_df.values, list(aggregated_shap_df.columns)

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
        return pd.concat([X_copy[non_one_hot_cols], aggregated_data], axis=1)


class BaseModelEvaluator(EvaluatorMethods, ABC):
    """Abstract base class for evaluating machine learning model performance.

    This class provides methods for calculating model performance metrics,
    plotting confusion matrices, and evaluating feature importance, with options
    for handling one-hot encoded features and aggregating SHAP values.

    Inherits:
        - `EvaluatorMethods`: Provides prediction and encoding methods.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        X (pd.DataFrame): The test dataset features.
        y (pd.Series): The test dataset labels.
        model (sklearn.base.BaseEstimator): A trained sklearn model instance
            for single-model evaluation.
        encoding (Optional[str]): Encoding type for categorical features, e.g.,
            'one_hot' or 'target', used for labeling and grouping in plots.
        aggregate (bool): If True, aggregates the importance values of multi-category
            encoded features for interpretability.

    Attributes:
        X (pd.DataFrame): Holds the test dataset features for evaluation.
        y (pd.Series): Holds the test dataset labels for evaluation.
        model (Optional[sklearn.base.BaseEstimator]): The primary model instance used
            for evaluation, if single-model evaluation is performed.
        encoding (Optional[str]): Indicates the encoding type used, which impacts
            plot titles and feature grouping in evaluations.
        aggregate (bool): Indicates whether to aggregate importance values of
            multi-category encoded features, enhancing interpretability in feature
            importance plots.

    Methods:
        calibration_plot: Plots calibration plot for model probabilities.
        brier_score_groups: Calculates Brier score within specified groups.
        bss_comparison: Compares Brier Skill Score of model with baseline.
        plot_confusion_matrix: Generates a styled confusion matrix heatmap
            for the test data and model predictions.

    Inherited Methods:
        - `brier_scores`: Calculates Brier score for each instance in the evaluator's
            dataset based on the model's predicted probabilities. Returns series of
            Brier scores indexed by instance.
        - `model_predictions`: Generates model predictions for evaluator's feature
            set, applying threshold-based binarization if specified, and returns
            predictions as a series indexed by instance.

    Abstract Methods:
        - `evaluate_feature_importance`: Abstract method for evaluating feature
            importance across models.
        - `analyze_brier_within_clusters`: Abstract method for analyzing Brier
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
        ],
        encoding: Optional[str],
        aggregate: bool,
    ) -> None:
        """Initialize the FeatureImportance class."""
        super().__init__(X=X, y=y, model=model, encoding=encoding, aggregate=aggregate)

    def calibration_plot(
        self,
        n_bins: int = 10,
        tight_layout: bool = False,
        task: Optional[str] = None,
    ) -> None:
        """Generates calibration plots for the model's predicted probabilities.

        Creates a calibration plot for binary classification or one for each
        class in a multiclass setup.

        Args:
            n_bins (int): Number of bins for plotting.
            tight_layout (bool): If True, applies tight layout to the plot.
                Defaults to False.
            task (Optional[str]): Task name to apply label mapping for the plot.
                Defaults to None.

        Raises:
            ValueError: If the model does not support probability predictions.
        """
        classification = "binary" if self.y.nunique() == 2 else "multiclass"
        probas = get_probs(self.model, classification=classification, X=self.X)

        if task is not None:
            label_mapping = _label_mapping(task)
            plot_title = (
                f"Calibration Plot \n {task_map.get(task, 'Binary')}"
                if classification == "binary"
                else f"Calibration Plot \n{task_map.get(task, 'Multiclass Task')}"
            )
        else:
            plot_title = "Calibration Plot"
            label_name = None

        if classification == "multiclass":
            num_classes = probas.shape[1]
            plt.figure(figsize=(4, 4), dpi=300)
            for class_idx in range(num_classes):
                class_probas = probas[:, class_idx]
                binarized_y = (self.y == class_idx).astype(int)
                prob_true, prob_pred = calibration_curve(
                    binarized_y, class_probas, n_bins=n_bins, strategy="uniform"
                )
                if task is not None:
                    label_name = label_mapping.get(class_idx, f"Class {class_idx}")
                plt.plot(prob_pred, prob_true, marker="o", label=label_name)
            plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            plt.xlabel("Mean Predicted Probability", fontsize=12)
            plt.ylabel("Fraction of Positives", fontsize=12)
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.title(plot_title, fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.legend(frameon=False)
            if tight_layout:
                plt.tight_layout()
            plt.show()
        else:
            prob_true, prob_pred = calibration_curve(
                self.y, probas, n_bins=n_bins, strategy="uniform"
            )
            plt.figure(figsize=(4, 4), dpi=300)
            plt.plot(prob_pred, prob_true, marker="o", label="Model", color="#078294")
            plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            plt.xlabel("Mean Predicted Probability", fontsize=12)
            plt.ylabel("Fraction of Positives", fontsize=12)
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(plot_title, fontsize=12)
            plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.legend(frameon=False)

            if tight_layout:
                plt.tight_layout()
            plt.show()

    def brier_score_groups(
        self,
        group_by: str = "y",
        task: Optional[str] = None,
        tight_layout: bool = False,
    ) -> None:
        """Calculates and displays Brier score within groups.

        Args:
            group_by (str): Grouping variable for calculating Brier scores.
                Defaults to "y".
            tight_layout (bool): If True, applies tight layout to the plot.
                Defaults to False.
            task (Optional[str]): Task name to apply label mapping for the plot.
                Defaults to None.
        """
        data = pd.DataFrame({group_by: self.y, "Brier_Score": self.brier_scores()})
        if task is not None:
            data[group_by] = data[group_by].map(_label_mapping(task))
        data_grouped = data.groupby(group_by)
        summary = data_grouped["Brier_Score"].agg(["mean", "median"]).reset_index()
        print(f"Average and Median Brier Scores by {group_by}:\n{summary}")

        plt.figure(figsize=(4, 4), dpi=300)
        plt.figure(figsize=(4, 4), dpi=300)
        sns.violinplot(
            x=group_by,
            y="Brier_Score",
            data=data,
            linewidth=0.5,
            color="#078294",
            inner_kws={"box_width": 4, "whis_width": 0.5},
        )
        sns.despine(top=True, right=True)
        plt.title("Distribution of Brier Scores", fontsize=12)
        plt.xlabel(f'{"y" if group_by == "y" else group_by}', fontsize=12)
        plt.ylabel("Brier Score", fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if tight_layout:
            plt.tight_layout()
        plt.show()

    def bss_comparison(
        self,
        baseline_models: Dict[Tuple[str, str], Any],
        classification: str,
        tight_layout: bool = False,
        num_patients: Optional[int] = None,
    ) -> None:
        """Compares the Brier Skill Scores (BSS) of the model with baseline models.

        Args:
            baseline_models (Dict[Tuple[str, str], Any]): A dictionary containing
                the baseline models. Keys are tuples of model name and type, and
                values are the trained model objects.
            classification (str): Classification type ('binary' or 'multiclass').
            tight_layout (bool): If True, applies tight layout to the plot.
                Defaults to False.
            num_patients (Optional[int]): Number of unique patients in the test set.
                Defaults to None.

        Raises:
            ValueError: If the model or any baseline model cannot predict probabilities.
        """
        trained_probs = (
            get_probs(self.model, classification=classification, X=self.X)
            if hasattr(self.model, "predict_proba")
            else None
        )

        if classification == "binary":
            trained_brier = brier_score_loss(y_true=self.y, y_proba=trained_probs)
        elif classification == "multiclass":
            trained_brier = brier_loss_multi(y=self.y, probs=trained_probs)
        else:
            raise ValueError(f"Unsupported classification type: {classification}")

        bss_data = []
        brier_scores = [{"Model": "Tuned Model", "Brier Score": trained_brier}]

        for model_name, model in baseline_models.items():
            baseline_probs = (
                get_probs(model, classification=classification, X=self.X)
                if hasattr(model, "predict_proba")
                else None
            )
            if classification == "binary":
                baseline_brier = brier_score_loss(y_true=self.y, y_proba=baseline_probs)
            else:
                baseline_brier = brier_loss_multi(y=self.y, probs=baseline_probs)

            bss = 1 - (trained_brier / baseline_brier)
            bss_data.append({"Model": model_name[0], "Brier Skill Score": bss})
            brier_scores.append({"Model": model_name[0], "Brier Score": baseline_brier})

        bss_df, brier_df = pd.DataFrame(bss_data), pd.DataFrame(brier_scores)
        df1_melted = brier_df.melt(
            id_vars=["Model"], var_name="Score", value_name="Value"
        )
        df2_melted = bss_df.melt(
            id_vars=["Model"], var_name="Score", value_name="Value"
        )

        combined_df = pd.concat([df1_melted, df2_melted], ignore_index=True)

        fig, axes = plt.subplots(figsize=(8, 4), dpi=300)

        model_order = [
            "Tuned Model",
            "Dummy Classifier",
            "Logistic Regression",
            "Random Forest",
        ]

        g = sns.barplot(
            data=combined_df,
            x="Model",
            y="Value",
            hue="Score",
            order=model_order,
            linewidth=1,
            edgecolor="black",
        )

        plt.axvline(x=0.5, color="gray", linestyle="--", linewidth=1)
        plt.axhline(y=0, color="black", linestyle="-", linewidth=1)

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        for _, e in enumerate(g.patches):
            if e.get_height() > 0:
                g.annotate(
                    f"{e.get_height():.2f}",
                    (e.get_x() + e.get_width() / 2, e.get_height()),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
            if e.get_height() < 0:
                g.annotate(
                    f"{e.get_height():.2f}",
                    (e.get_x() + e.get_width() / 2, e.get_height()),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    xytext=(0, -10),
                    textcoords="offset points",
                )

        plt.legend(
            title="Metric",
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(1.05, 0.5),
            loc="upper left",
        )
        if num_patients is not None:
            plt.title(
                f"Baseline Comparison \n"
                f"Number of Patients {num_patients}; Number of Sites: {len(self.y)}"
            )
        else:
            plt.title(f"Baseline Comparison \n Number of Sites: {len(self.y)}")

        labels = [label.get_text() for label in g.get_xticklabels()]
        g.set_xticklabels([label.replace(" ", "\n") for label in labels])

        if tight_layout:
            plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(
        self,
        col: Optional[pd.Series] = None,
        y_label: str = "True",
        normalize: str = "rows",
        tight_layout: bool = False,
        task: Optional[str] = None,
    ) -> plt.Figure:
        """Generates a styled confusion matrix for the given model and test data.

        Args:
            col (Optional[pd.Series]): Column for y label. Defaults to None.
            y_label (str): Description of y label. Defaults to "True".
            normalize (str, optional): Normalization method ('rows' or 'columns').
                Defaults to 'rows'.
            tight_layout (bool): If True, applies tight layout to the plot.
                Defaults to False.
            task (Optional[str]): Task name to apply label mapping for the plot.
                Defaults to None.

        Returns:
            Figure: Confusion matrix heatmap plot.
        """
        y_true = pd.Series(col if col is not None else self.y)
        pred = self.model_predictions()

        if task is not None:
            label_mapping = _label_mapping(task)
            y_true = y_true.map(label_mapping)
            pred = pd.Series(pred).map(label_mapping).values
            labels = list(label_mapping.values())
        else:
            labels = None

        cm = confusion_matrix(y_true=y_true, y_pred=pred, labels=labels)

        if normalize == "rows":
            row_sums = cm.sum(axis=1, keepdims=True)
            normalized_cm = (cm / row_sums) * 100
        elif normalize == "columns":
            col_sums = cm.sum(axis=0, keepdims=True)
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
            xticklabels=labels if labels else range(cm.shape[1]),
            yticklabels=labels if labels else range(cm.shape[0]),
        )

        for i in range(len(cm)):
            for j in range(len(cm)):
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if normalized_cm[i, j] > 50 else "black",
                )

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
        if tight_layout:
            plt.tight_layout()
        plt.show()

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
