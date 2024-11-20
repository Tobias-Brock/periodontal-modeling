from typing import List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from ._baseeval import BaseModelEvaluator


class ModelEvaluator(BaseModelEvaluator):
    """Concrete implementation for evaluating machine learning model performance.

    This class extends `BaseModelEvaluator` to provide methods for calculating
    feature importance using SHAP, permutation importance, and standard model
    importance. It also supports clustering analyses of Brier scores.

    Inherits:
        - `BaseModelEvaluator`: Provides methods for model evaluation, calculating
          Brier scores, plotting confusion matrices, and aggregating feature
          importance for one-hot encoded features.

    Args:
        X (pd.DataFrame): The dataset features used for testing the model's
            performance.
        y (pd.Series): The true labels for the test dataset.
        model (sklearn.base.BaseEstimator): A single trained model instance
            (e.g., `RandomForestClassifier` or `LogisticRegression`) for evaluation.
        encoding (Optional[str]): Encoding type for categorical variables used in plot
            titles and feature grouping (e.g., 'one_hot' or 'target').
        aggregate (bool): If True, aggregates the importance values of multi-category
            encoded features for interpretability.

    Attributes:
        X (pd.DataFrame): Stores the test dataset features for model evaluation.
        y (pd.Series): Stores the test dataset labels for model evaluation.
        model (sklearn.base.BaseEstimator): Primary model used for evaluation.
        encoding (Optional[str]): Indicates the encoding type used, which impacts
            plot titles and feature grouping in evaluations.
        aggregate (bool): Indicates whether to aggregate importance values of
            multi-category encoded features, enhancing interpretability in feature
            importance plots.

    Methods:
        evaluate_feature_importance: Calculates feature importance scores using
            specified methods (`shap`, `permutation`, or `standard`).
        analyze_brier_within_clusters: Computes Brier scores within clusters formed by a
            specified clustering algorithm and provides visualizations.

    Inherited Methods:
        - `brier_scores`: Calculates Brier score for each instance in the evaluator's
            dataset based on the model's predicted probabilities. Returns series of
            Brier scores indexed by instance.
        - `calibration_plot`: Plots calibration plot for model probabilities.
        - `model_predictions`: Generates model predictions for evaluator's feature
            set, applying threshold-based binarization if specified, and returns
            predictions as a series indexed by instance.
        - `brier_score_groups`: Calculates Brier score within specified groups
        - `bss_comparison`: Compares Brier Skill Score of model with baseline.
          based on a grouping variable (e.g., target class).
        - `plot_confusion_matrix`: Generates a styled confusion matrix heatmap
          for model predictions, with optional normalization.

    Example:
        ```
        # Use X_test, y_test obtained from Resampler
        evaluator = ModelEvaluator(
            X=X_test, y=y_test, model=trained_rf_model, encoding="one_hot"
            )
        evaluator.evaluate_feature_importance(fi_types=["shap", "permutation"])
        brier_plot, heatmap_plot, clustered_data = (
            evaluator.analyze_brier_within_clusters()
        )
        ```
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
        encoding: Optional[str] = None,
        aggregate: bool = True,
    ) -> None:
        """Initialize the FeatureImportance class."""
        super().__init__(X=X, y=y, model=model, encoding=encoding, aggregate=aggregate)

    def evaluate_feature_importance(self, fi_types: List[str]) -> None:
        """Evaluate the feature importance for a list of trained models.

        Args:
            fi_types (List[str]): Methods of feature importance evaluation:
                'shap', 'permutation', 'standard'.

        Returns:
            Plot: Feature importance plot for the specified method.
        """
        feature_names = self.X.columns.tolist()
        importance_dict = {}

        for fi_type in fi_types:
            model_name = type(self.model).__name__

            if fi_type == "shap":
                if isinstance(self.model, MLPClassifier):
                    explainer = shap.Explainer(self.model.predict_proba, self.X)
                else:
                    explainer = shap.Explainer(self.model, self.X)

                if isinstance(self.model, (RandomForestClassifier, XGBClassifier)):
                    shap_values = explainer.shap_values(self.X, check_additivity=False)
                    if (
                        isinstance(self.model, RandomForestClassifier)
                        and len(shap_values.shape) == 3
                    ):
                        shap_values = np.abs(shap_values).mean(axis=-1)
                else:
                    shap_values = explainer.shap_values(self.X)

                if isinstance(shap_values, list):
                    shap_values_stacked = np.stack(shap_values, axis=-1)
                    shap_values = np.abs(shap_values_stacked).mean(axis=-1)
                else:
                    shap_values = np.abs(shap_values)

            elif fi_type == "permutation":
                result = permutation_importance(
                    estimator=self.model,
                    X=self.X,
                    y=self.y,
                    n_repeats=10,
                    random_state=0,
                )
                fi_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": result.importances_mean,
                    }
                )

            elif fi_type == "standard":
                if isinstance(self.model, (RandomForestClassifier, XGBClassifier)):
                    importances = self.model.feature_importances_
                elif isinstance(self.model, LogisticRegression):
                    importances = abs(self.model.coef_[0])
                else:
                    print(f"Standard FI not supported for model type {model_name}.")
                    continue
                fi_df = pd.DataFrame(
                    {"Feature": feature_names, "Importance": importances}
                )

            else:
                raise ValueError(f"Invalid fi_type: {fi_type}")

            if self.aggregate:
                if fi_type == "shap":
                    aggregated_shap_values, aggregated_feature_names = (
                        self._aggregate_shap_one_hot(
                            shap_values=shap_values, feature_names=feature_names
                        )
                    )
                    aggregated_feature_names = self._feature_mapping(
                        aggregated_feature_names
                    )

                    aggregated_shap_df = pd.DataFrame(
                        aggregated_shap_values, columns=aggregated_feature_names
                    )
                    importance_dict[f"{model_name}_{fi_type}"] = aggregated_shap_df

                    plt.figure(figsize=(4, 4), dpi=300)
                    shap.summary_plot(
                        aggregated_shap_values,
                        feature_names=aggregated_feature_names,
                        plot_type="bar",
                        show=False,
                    )

                    ax = plt.gca()
                    for bar in ax.patches:
                        bar.set_edgecolor("black")
                        bar.set_linewidth(1)

                    ax.spines["left"].set_visible(True)
                    ax.spines["left"].set_color("black")
                    ax.spines["bottom"].set_color("black")
                    ax.tick_params(axis="y", colors="black")

                    plt.title(f"{model_name}: SHAP Feature Importance")
                    plt.tight_layout()

                else:
                    fi_df_aggregated = self._aggregate_one_hot_importances(fi_df=fi_df)
                    fi_df_aggregated.sort_values(
                        by="Importance", ascending=False, inplace=True
                    )

                    fi_df_aggregated["Feature"] = self._feature_mapping(
                        fi_df_aggregated["Feature"]
                    )

                    importance_dict[f"{model_name}_{fi_type}"] = fi_df_aggregated

                    top10_fi_df_aggregated = fi_df_aggregated.head(10)
                    bottom10_fi_df_aggregated = fi_df_aggregated.tail(10)

                    placeholder = pd.DataFrame(
                        [["[...]", 0]], columns=["Feature", "Importance"]
                    )
                    selected_fi_df_aggregated = pd.concat(
                        [
                            top10_fi_df_aggregated,
                            placeholder,
                            bottom10_fi_df_aggregated,
                        ],
                        ignore_index=True,
                    )

            else:
                if fi_type == "shap":
                    feature_names = self._feature_mapping(feature_names)

                    plt.figure(figsize=(4, 4), dpi=300)
                    shap.summary_plot(
                        shap_values,
                        self.X,
                        plot_type="bar",
                        feature_names=feature_names,
                        show=False,
                    )
                    ax = plt.gca()
                    for bar in ax.patches:
                        bar.set_edgecolor("black")
                        bar.set_linewidth(1)

                    ax.spines["left"].set_visible(True)
                    ax.spines["left"].set_color("black")
                    ax.spines["bottom"].set_color("black")
                    ax.tick_params(axis="y", colors="black")
                    plt.title(f"{model_name}: SHAP Feature Importance")
                    plt.tight_layout()

                else:
                    fi_df.sort_values(by="Importance", ascending=False, inplace=True)
                    fi_df["Feature"] = self._feature_mapping(fi_df["Feature"])
                    importance_dict[model_name] = fi_df

            if fi_type != "shap":
                plt.figure(figsize=(8, 6), dpi=300)

                if self.aggregate:
                    plt.bar(
                        selected_fi_df_aggregated["Feature"],
                        selected_fi_df_aggregated["Importance"],
                        edgecolor="black",
                        linewidth=1,
                        color="#078294",
                    )
                else:
                    plt.bar(fi_df["Feature"], fi_df["Importance"])

                plt.title(f"{model_name}: {fi_type.title()} Feature Importance")
                plt.xticks(rotation=90, fontsize=12)
                plt.yticks(fontsize=12)
                plt.axhline(y=0, color="black", linewidth=1)
                ax = plt.gca()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                plt.ylabel("Importance", fontsize=12)
                plt.tight_layout()
                plt.show()

    def analyze_brier_within_clusters(
        self,
        clustering_algorithm: Type = AgglomerativeClustering,
        n_clusters: int = 3,
        tight_layout: bool = False,
    ) -> Union[None, Tuple[plt.Figure, plt.Figure, pd.DataFrame]]:
        """Analyze distribution of Brier scores within clusters formed by input data.

        Args:
            clustering_algorithm (Type): Clustering algorithm class from sklearn to use
                for clustering.
            n_clusters (int): Number of clusters to form.
            tight_layout (bool): If True, applies tight layout to the plots. Defaults
                to False.

        Returns:
            Union: Tuple containing the Brier score plot, heatmap plot, and clustered
                DataFrame with 'Cluster' and 'Brier_Score' columns.

        Raises:
            ValueError: If the provided model cannot predict probabilities.
        """
        probas = self.model.predict_proba(self.X)[:, 1]
        brier_scores = [
            brier_score_loss(y_true=[true], y_proba=[proba])
            for true, proba in zip(self.y, probas, strict=False)
        ]

        X_cluster_input = (
            self._aggregate_one_hot_features_for_clustering(X=self.X)
            if self.aggregate
            else self.X
        )

        clustering_algo = clustering_algorithm(n_clusters=n_clusters)
        cluster_labels = clustering_algo.fit_predict(X_cluster_input)
        X_clustered = X_cluster_input.assign(
            Cluster=cluster_labels, Brier_Score=brier_scores
        )
        mean_brier_scores = X_clustered.groupby("Cluster")["Brier_Score"].mean()
        cluster_counts = X_clustered["Cluster"].value_counts().sort_index()

        print(
            "\nMean Brier Score per cluster:\n",
            mean_brier_scores,
            "\n\nNumber of observations per cluster:\n",
            cluster_counts,
        )

        feature_averages = (
            X_clustered.drop(["Cluster", "Brier_Score"], axis=1)
            .groupby(X_clustered["Cluster"])
            .mean()
        )

        feature_averages.columns = self._feature_mapping(
            features=feature_averages.columns
        )

        plt.figure(figsize=(8, 4), dpi=300)
        plt.rcParams.update({"font.size": 12})
        sns.violinplot(
            x="Cluster",
            y="Brier_Score",
            data=X_clustered,
            linewidth=0.5,
            color="#078294",
            inner_kws={"box_width": 6, "whis_width": 0.5},
        )
        sns.pointplot(
            x="Cluster",
            y="Brier_Score",
            data=mean_brier_scores.reset_index(),
            color="darkred",
            markers="D",
            scale=0.75,
            ci=None,
        )
        sns.despine(top=True, right=True)
        plt.ylabel("Brier Score")
        plt.xlabel("Cluster", fontsize=12)
        plt.title("Brier Score Distribution in Clusters", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if tight_layout:
            plt.tight_layout()
        brier_plot = plt.gcf()

        plt.figure(figsize=(8, 4), dpi=300)
        annot_array = np.around(feature_averages.values, decimals=1)
        sns.heatmap(
            feature_averages,
            cmap="viridis",
            annot=annot_array,
            fmt=".1f",
            annot_kws={"size": 5, "rotation": 90},
        )
        if tight_layout:
            plt.tight_layout()

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        cbar = ax.collections[0].colorbar
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1)
        heatmap_plot = plt.gcf()

        return brier_plot, heatmap_plot, X_clustered
