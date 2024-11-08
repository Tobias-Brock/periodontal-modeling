from typing import List, Optional, Type, Union

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
        - BaseModelEvaluator: Provides methods for model evaluation, calculating
          Brier scores, plotting confusion matrices, and aggregating feature
          importance for one-hot encoded features.

    Args:
        X (pd.DataFrame): The dataset features used for testing the model's
            performance.
        y (pd.Series): The true labels for the test dataset.
        model (Optional[sklearn.base.BaseEstimator]): A single trained model instance
            (e.g., `RandomForestClassifier` or `LogisticRegression`) for evaluation.
        models (Optional[List[sklearn.base.BaseEstimator]]): A list of trained model
            instances for evaluation, enabling multi-model analysis.
        encoding (Optional[str]): Encoding type for categorical variables used in plot
            titles and feature grouping (e.g., 'one_hot' or 'target').
        aggregate (bool): If True, aggregates the importance values of multi-category
            encoded features for interpretability.

    Attributes:
        X (pd.DataFrame): Stores the test dataset features for model evaluation.
        y (pd.Series): Stores the test dataset labels for model evaluation.
        model (Optional[sklearn.base.BaseEstimator]): Primary model used for evaluation.
        models (List[sklearn.base.BaseEstimator]): Collection of models for multi-model
            evaluations.
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
        - `model_predictions`: Generates model predictions for evaluator's feature
            set, applying threshold-based binarization if specified, and returns
            predictions as a series indexed by instance.
        - `brier_score_groups`: Calculates Brier score within specified groups
          based on a grouping variable (e.g., target class).
        - `plot_confusion_matrix`: Generates a styled confusion matrix heatmap
          for model predictions, with optional normalization.

    Example:
        ```
        evaluator = ModelEvaluator(X, y, model=trained_rf_model, encoding="one_hot")
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
            None,
        ] = None,
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
        ] = None,
        encoding: Optional[str] = None,
        aggregate: bool = True,
    ) -> None:
        """Initialize the FeatureImportance class."""
        super().__init__(
            X=X, y=y, model=model, models=models, encoding=encoding, aggregate=aggregate
        )

    def evaluate_feature_importance(self, fi_types: List[str]) -> None:
        """Evaluate the feature importance for a list of trained models.

        Args:
            fi_types (List[str]): Methods of feature importance evaluation:
                'shap', 'permutation', 'standard'.

        Returns:
            A feature importance plot for the specified method.
        """
        if self.models and self.model is None:
            return None

        if not self.models and self.model:
            self.models = [self.model]

        feature_names = self.X.columns.tolist()
        importance_dict = {}

        for model in self.models:
            for fi_type in fi_types:
                model_name = type(model).__name__

                if fi_type == "shap":
                    if isinstance(model, MLPClassifier):
                        explainer = shap.Explainer(model.predict_proba, self.X)
                    else:
                        explainer = shap.Explainer(model, self.X)

                    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                        shap_values = explainer.shap_values(
                            self.X, check_additivity=False
                        )
                        if (
                            isinstance(model, RandomForestClassifier)
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
                        estimator=model,
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
                    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                        importances = model.feature_importances_
                    elif isinstance(model, LogisticRegression):
                        importances = abs(model.coef_[0])
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
                        aggregated_shap_df = pd.DataFrame(
                            aggregated_shap_values, columns=aggregated_feature_names
                        )
                        importance_dict[f"{model_name}_{fi_type}"] = aggregated_shap_df
                        plt.figure(figsize=(3, 2), dpi=200)
                        shap.summary_plot(
                            aggregated_shap_values,
                            feature_names=aggregated_feature_names,
                            plot_type="bar",
                            show=False,
                        )

                        plt.title(
                            f"{model_name} SHAP Feature Importance {self.encoding}"
                        )
                    else:
                        fi_df_aggregated = self._aggregate_one_hot_importances(
                            fi_df=fi_df
                        )
                        fi_df_aggregated.sort_values(
                            by="Importance", ascending=False, inplace=True
                        )
                        importance_dict[f"{model_name}_{fi_type}"] = fi_df_aggregated
                else:
                    if fi_type == "shap":
                        plt.figure(figsize=(3, 2), dpi=200)
                        shap.summary_plot(
                            shap_values,
                            self.X,
                            plot_type="bar",
                            feature_names=feature_names,
                            show=False,
                        )
                        plt.title(
                            f"{model_name} SHAP Feature Importance {self.encoding}"
                        )
                    else:
                        fi_df.sort_values(
                            by="Importance", ascending=False, inplace=True
                        )
                        importance_dict[model_name] = fi_df

                if fi_type != "shap":
                    plt.figure(figsize=(6, 4), dpi=250)
                    if self.aggregate:
                        plt.bar(
                            fi_df_aggregated["Feature"],
                            fi_df_aggregated["Importance"],
                        )
                    else:
                        plt.bar(
                            fi_df["Feature"],
                            fi_df["Importance"],
                        )

                    plt.title(f"{model_name} {fi_type.title()} FI {self.encoding}")
                    plt.xticks(rotation=45, fontsize=3)
                    plt.ylabel("Importance")
                    plt.tight_layout()
                    plt.show()

    def analyze_brier_within_clusters(
        self,
        clustering_algorithm: Type = AgglomerativeClustering,
        n_clusters: int = 3,
    ) -> pd.DataFrame:
        """Analyze distribution of Brier scores within clusters formed by input data.

        Args:
            clustering_algorithm (Type): Clustering algorithm class from sklearn to use
                for clustering.
            n_clusters (int): Number of clusters to form.

        Returns:
            pd.DataFrame: The input DataFrame X with columns for 'Cluster' labels
            and 'Brier_Score'.

        Raises:
            ValueError: If the provided model cannot predict probabilities.
        """
        if self.model is None:
            return None

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("The provided model cannot predict probabilities.")

        probas = self.model.predict_proba(self.X)[:, 1]
        brier_scores = [
            brier_score_loss(y_true=[true], y_proba=[proba])
            for true, proba in zip(self.y, probas, strict=False)
        ]

        if self.aggregate:
            X_cluster_input = self._aggregate_one_hot_features_for_clustering(X=self.X)
        else:
            X_cluster_input = self.X

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

        plt.figure(figsize=(8, 6), dpi=150)
        sns.boxplot(x="Cluster", y="Brier_Score", data=X_clustered)

        sns.pointplot(
            x="Cluster",
            y="Brier_Score",
            data=mean_brier_scores.reset_index(),
            color="darkred",
            markers="D",
            scale=0.75,
            ci=None,
        )
        brier_plot = plt.gcf()

        plt.figure(figsize=(12, 8), dpi=150)
        annot_array = np.around(feature_averages.values, decimals=1)
        sns.heatmap(
            feature_averages,
            cmap="viridis",
            annot=annot_array,
            fmt=".1f",
            annot_kws={"size": 8},
        )
        heatmap_plot = plt.gcf()

        return brier_plot, heatmap_plot, X_clustered
