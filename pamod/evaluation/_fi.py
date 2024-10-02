from typing import List, Tuple, Type, Union

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


def _is_one_hot_encoded(feature: str) -> bool:
    """Check if a feature is one-hot encoded.

    Args:
        feature (str): The name of the feature to check.

    Returns:
        bool: True if the feature is one-hot encoded.
    """
    parts = feature.rsplit("_", 1)
    return len(parts) > 1 and (parts[1].isdigit() or feature.startswith("Stresslvl"))


def _get_base_name(feature: str) -> str:
    """Extract the base name of a feature.

    Args:
        feature (str): The name of the feature to process.

    Returns:
        str: The base name of the feature.
    """
    if feature.startswith("Stresslvl"):
        return "Stresslvl"
    if _is_one_hot_encoded(feature):
        return feature.rsplit("_", 1)[0]
    return feature


class FeatureImportanceEngine:
    def __init__(
        self,
        models: List[
            Union[
                RandomForestClassifier, LogisticRegression, MLPClassifier, XGBClassifier
            ]
        ],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        encoding: str,
        aggregate: bool = True,
    ) -> None:
        """Initialize the FeatureImportance class.

        Args:
            models (List[sklearn estimators]): List of trained models.
            X_test (pd.DataFrame): Test dataset features.
            y_test (pd.Series): Test dataset labels.
            encoding (str): Determines encoding for plot titles ('one_hot' or 'target').
            aggregate (bool): If True, aggregates importance values of one-hot features.
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.encoding = encoding
        self.aggregate = aggregate

    def _aggregate_one_hot_importances(
        self, feature_importance_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate importance scores of one-hot encoded variables.

        Args:
            feature_importance_df (pd.DataFrame): DataFrame with features and their
                importance scores.

        Returns:
            pd.DataFrame: Updated DataFrame with aggregated importance scores.
        """
        base_names = feature_importance_df["Feature"].apply(_get_base_name)
        aggregated_importances = (
            feature_importance_df.groupby(base_names)["Importance"].sum().reset_index()
        )
        aggregated_importances.columns = ["Feature", "Importance"]
        original_features = feature_importance_df["Feature"][
            ~feature_importance_df["Feature"].apply(_is_one_hot_encoded)
        ].unique()

        aggregated_or_original = (
            pd.concat(
                [
                    aggregated_importances,
                    feature_importance_df[
                        feature_importance_df["Feature"].isin(original_features)
                    ],
                ]
            )
            .drop_duplicates()
            .sort_values(by="Importance", ascending=False)
        )

        return aggregated_or_original.reset_index(drop=True)

    def _aggregate_shap_one_hot(
        self, shap_values: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Aggregate SHAP values of one-hot encoded variables.

        Args:
            shap_values (np.ndarray): SHAP values with shape (n_samples, n_features).
            feature_names (List[str]): List of features corresponding to SHAP values.

        Returns:
            Tuple[np.ndarray, List[str]]: Aggregated SHAP values and updated list of
            feature names.
        """
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        base_names = [_get_base_name(feature) for feature in shap_df.columns]
        feature_mapping = dict(zip(shap_df.columns, base_names, strict=False))
        aggregated_shap_df = shap_df.groupby(feature_mapping, axis=1).sum()
        updated_feature_names = list(aggregated_shap_df.columns)
        aggregated_shap_values = aggregated_shap_df.values

        return aggregated_shap_values, updated_feature_names

    def evaluate_feature_importance(self, importance_types: List[str]) -> None:
        """Evaluate the feature importance for a list of trained models.

        Args:
            importance_types (List[str]): Methods of feature importance evaluation:
                'shap', 'permutation', 'standard'.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing DataFrames of features and
            their importance scores for each model.
        """
        feature_names = self.X_test.columns.tolist()
        importance_dict = {}

        for model in self.models:
            for importance_type in importance_types:
                model_name = type(model).__name__

                if importance_type == "shap":
                    if isinstance(model, MLPClassifier):
                        explainer = shap.Explainer(model.predict_proba, self.X_test)
                    else:
                        explainer = shap.Explainer(model, self.X_test)

                    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                        shap_values = explainer.shap_values(
                            self.X_test, check_additivity=False
                        )
                        if (
                            isinstance(model, RandomForestClassifier)
                            and len(shap_values.shape) == 3
                        ):
                            shap_values = np.abs(shap_values).mean(axis=-1)
                    else:
                        shap_values = explainer.shap_values(self.X_test)

                    if isinstance(shap_values, list):
                        shap_values_stacked = np.stack(shap_values, axis=-1)
                        shap_values = np.abs(shap_values_stacked).mean(axis=-1)
                    else:
                        shap_values = np.abs(shap_values)

                elif importance_type == "permutation":
                    result = permutation_importance(
                        model, self.X_test, self.y_test, n_repeats=10, random_state=0
                    )
                    feature_importance_df = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Importance": result.importances_mean,
                        }
                    )

                elif importance_type == "standard":
                    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                        importances = model.feature_importances_
                    elif isinstance(model, LogisticRegression):
                        importances = abs(model.coef_[0])
                    else:
                        print(f"Standard FI not supported for model type {model_name}.")
                        continue
                    feature_importance_df = pd.DataFrame(
                        {"Feature": feature_names, "Importance": importances}
                    )

                else:
                    raise ValueError(f"Invalid importance_type: {importance_type}")

                if self.aggregate:
                    if importance_type == "shap":
                        aggregated_shap_values, aggregated_feature_names = (
                            self._aggregate_shap_one_hot(shap_values, feature_names)
                        )
                        aggregated_shap_df = pd.DataFrame(
                            aggregated_shap_values, columns=aggregated_feature_names
                        )
                        importance_dict[f"{model_name}_{importance_type}"] = (
                            aggregated_shap_df
                        )
                        plt.figure(figsize=(3, 2), dpi=150)
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
                        feature_importance_df_aggregated = (
                            self._aggregate_one_hot_importances(feature_importance_df)
                        )
                        feature_importance_df_aggregated.sort_values(
                            by="Importance", ascending=False, inplace=True
                        )
                        importance_dict[f"{model_name}_{importance_type}"] = (
                            feature_importance_df_aggregated
                        )
                else:
                    if importance_type == "shap":
                        plt.figure(figsize=(3, 2), dpi=150)
                        shap.summary_plot(
                            shap_values,
                            self.X_test,
                            plot_type="bar",
                            feature_names=feature_names,
                            show=False,
                        )
                        plt.title(
                            f"{model_name} SHAP Feature Importance {self.encoding}"
                        )
                    else:
                        feature_importance_df.sort_values(
                            by="Importance", ascending=False, inplace=True
                        )
                        importance_dict[model_name] = feature_importance_df

                if importance_type != "shap":
                    plt.figure(figsize=(6, 4), dpi=200)
                    if self.aggregate:
                        plt.bar(
                            feature_importance_df_aggregated["Feature"],
                            feature_importance_df_aggregated["Importance"],
                        )
                    else:
                        plt.bar(
                            feature_importance_df["Feature"],
                            feature_importance_df["Importance"],
                        )

                    plt.title(
                        f"{model_name} {importance_type.title()} FI {self.encoding}"
                    )
                    plt.xticks(rotation=45, fontsize=3)
                    plt.ylabel("Importance")
                    plt.tight_layout()
                    plt.show()

    def analyze_brier_within_clusters(
        self,
        model: Union[
            RandomForestClassifier, LogisticRegression, MLPClassifier, XGBClassifier
        ],
        clustering_algorithm: Type = AgglomerativeClustering,
        n_clusters: int = 3,
    ) -> pd.DataFrame:
        """Analyze distribution of Brier scores within clusters formed by input data.

        Args:
            model (sklearn estimator): A trained model that supports `predict_proba`.
            X (pd.DataFrame): Input features dataframe without the target variable.
            y (pd.Series or np.ndarray): True labels corresponding to X.
            clustering_algorithm (Type): Clustering algorithm class from sklearn to use
                for clustering.
            n_clusters (int): Number of clusters to form.

        Returns:
            pd.DataFrame: The input DataFrame X with columns for 'Cluster' labels
            and 'Brier_Score'.

        Raises:
            ValueError: If the provided model cannot predict probabilities.
        """
        if not hasattr(model, "predict_proba"):
            raise ValueError("The provided model cannot predict probabilities.")

        probas = model.predict_proba(self.X_test)[:, 1]
        brier_scores = [
            brier_score_loss([true], [proba])
            for true, proba in zip(self.y_test, probas, strict=False)
        ]
        clustering_algo = clustering_algorithm(n_clusters=n_clusters)
        cluster_labels = clustering_algo.fit_predict(self.X_test)

        X_clustered = self.X_test.copy()
        X_clustered["Cluster"] = cluster_labels
        X_clustered["Brier_Score"] = brier_scores

        mean_brier_scores = X_clustered.groupby("Cluster")["Brier_Score"].mean()
        feature_averages = (
            X_clustered.drop(["Cluster", "Brier_Score"], axis=1)
            .groupby(X_clustered["Cluster"])
            .mean()
        )

        plt.figure(figsize=(8, 6), dpi=150)
        sns.boxplot(x="Cluster", y="Brier_Score", data=X_clustered)
        mean_brier_scores = (
            X_clustered.groupby("Cluster")["Brier_Score"].mean().reset_index()
        )
        sns.pointplot(
            x="Cluster",
            y="Brier_Score",
            data=mean_brier_scores,
            color="darkred",
            markers="D",
            scale=0.75,
            ci=None,
        )
        plt.show()

        plt.figure(figsize=(12, 8), dpi=150)
        annot_array = np.around(feature_averages.values, decimals=1)
        sns.heatmap(
            feature_averages,
            cmap="viridis",
            annot=annot_array,
            fmt=".1f",
            annot_kws={"size": 8},
        )
        plt.show()

        return X_clustered
