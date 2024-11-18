from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..base import BaseConfig
from ..data import ProcessedDataLoader
from ..resampling import Resampler
from ..training import final_metrics, get_probs

column_order_binary = [
    "Model",
    "F1 Score",
    "Precision",
    "Recall",
    "Accuracy",
    "Brier Score",
    "Brier Skill Score",
    "ROC AUC Score",
    "Confusion Matrix",
]

column_order_multiclass = [
    "Model",
    "Macro F1",
    "Accuracy",
    "Class F1 Scores",
    "Multiclass Brier Score",
    "Brier Skill Score",
]


def _brier_skill_score(row, baseline_brier, ref_brier, metric_column):
    """Calculates the Brier Skill Score (BSS) for a given row.

    Args:
        row (pd.Series): The row from the DataFrame.
        baseline_brier (float): Brier score of the baseline (Dummy Classifier).
        ref_brier (float): Brier score of the reference model (Logistic Regression).
        metric_column (str): Column name for the Brier Score.

    Returns:
        float or None: The computed BSS, or None for the Dummy Classifier.
    """
    if row["Model"] == "Dummy Classifier":
        return None
    if row["Model"] == "Logistic Regression":
        return 1 - (row[metric_column] / baseline_brier)
    return 1 - (row[metric_column] / ref_brier)


class Baseline(BaseConfig):
    """Evaluates baseline models on a given dataset.

    This class loads, preprocesses, and evaluates a set of baseline models on a
    specified dataset. The baseline models include a Random Forest, Logistic
    Regression, and a Dummy Classifier, which are trained and evaluated on
    split data, returning a summary of performance metrics for each model.

    Inherits:
        - `BaseConfig`: Provides configuration settings for data processing.

    Args:
        task (str): Task name used to determine the classification type.
        encoding (str): Encoding type for categorical columns.
        random_state (int, optional): Random seed for reproducibility. Defaults to 0.
        lr_solver (str, optional): Solver used by Logistic Regression. Defaults to
            'saga'.
        dummy_strategy (str, optional): Strategy for DummyClassifier, defaults to
            'prior'.
        models (List[Tuple[str, object]], optional): List of models to benchmark.
            If not provided, default models are initialized.
        n_jobs (int): Number of parallel jobs. Defaults to -1.
        path (Path): Path to the directory containing processed data files.
            Defaults to Path("data/processed/processed_data.csv").

    Attributes:
        classification (str): Specifies classification type ('binary' or
            'multiclass') based on the task.
        resampler (Resampler): Strategy for resampling data during training/testing
            split.
        dataloader (ProcessedDataLoader): Loader for processing and transforming the
            dataset.
        dummy_strategy (str): Strategy used by the DummyClassifier, default is 'prior'.
        lr_solver (str): Solver for Logistic Regression, default is 'saga'.
        random_state (int): Random seed for reproducibility, default is 0.
        models (List[Tuple[str, object]]): List of models to benchmark, each
            represented as a tuple containing the model's name and the initialized
            model object.
        path (Path): Path to the directory containing processed data files.

    Methods:
        train_baselines: Trains and returns baseline models with test data.
        baseline: Trains and evaluates each model in the models list, returning
            a DataFrame with evaluation metrics.

    Example:
        ```
        # Initialize baseline evaluation for pocket closure task
        baseline = Baseline(
            task="pocketclosure",
            encoding="one_hot",
            random_state=42,
            lr_solver="saga",
            dummy_strategy="most_frequent"
        )

        # Evaluate baseline models and display results
        results_df = baseline.baseline()
        print(results_df)
        ```
    """

    def __init__(
        self,
        task: str,
        encoding: str,
        random_state: int = 0,
        lr_solver: str = "saga",
        dummy_strategy: str = "prior",
        models: Union[List[Tuple[str, object]], None] = None,
        n_jobs: int = -1,
        path: Path = Path("data/processed/processed_data.csv"),
    ) -> None:
        """Initializes the Baseline class with default or user-specified models."""
        if task in ["pocketclosure", "pocketclosureinf", "improvement"]:
            self.classification = "binary"
        elif task == "pdgrouprevaluation":
            self.classification = "multiclass"
        else:
            raise ValueError(
                f"Unknown task: {task}. Unable to determine classification."
            )

        self.resampler = Resampler(
            classification=self.classification, encoding=encoding
        )
        self.dataloader = ProcessedDataLoader(task=task, encoding=encoding)
        self.dummy_strategy = dummy_strategy
        self.lr_solver = lr_solver
        self.random_state = random_state
        self.path = path
        self.default_models: Union[list[str], None]

        if models is None:
            self.models = [
                (
                    "Random Forest",
                    RandomForestClassifier(
                        n_jobs=n_jobs, random_state=self.random_state
                    ),
                ),
                (
                    "Logistic Regression",
                    LogisticRegression(
                        solver=self.lr_solver,
                        random_state=self.random_state,
                        n_jobs=n_jobs,
                    ),
                ),
                (
                    "Dummy Classifier",
                    DummyClassifier(strategy=self.dummy_strategy),
                ),
            ]
            self.default_models = [name for name, _ in self.models]
        else:
            self.models = models
            self.default_models = None

    @staticmethod
    def _bss_helper(
        results_df: pd.DataFrame, classification: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Calculates Brier Skill Score (BSS) and determines column order.

        Args:
            results_df (pd.DataFrame): DataFrame containing evaluation metrics.
            classification (str): Classification type ('binary' or 'multiclass').

        Returns:
            Tuple[pd.DataFrame, List[str]]: Updated DataFrame with BSS and column order.
        """
        if classification == "binary":
            metric_column = "Brier Score"
            column_order = column_order_binary
        elif classification == "multiclass":
            metric_column = "Multiclass Brier Score"
            column_order = column_order_multiclass
            if "Class F1 Scores" in results_df.columns:
                results_df["Class F1 Scores"] = results_df["Class F1 Scores"].apply(
                    lambda scores: [round(score, 4) for score in scores]
                )
        else:
            raise ValueError(f"Unsupported classification type: {classification}")

        if metric_column in results_df.columns:
            dummy_brier = results_df.loc[
                results_df["Model"] == "Dummy Classifier", metric_column
            ].iloc[0]
            logreg_brier = results_df.loc[
                results_df["Model"] == "Logistic Regression", metric_column
            ].iloc[0]
            results_df["Brier Skill Score"] = results_df.apply(
                lambda row: _brier_skill_score(
                    row, dummy_brier, logreg_brier, metric_column
                ),
                axis=1,
            ).round(4)

        return results_df, column_order

    def train_baselines(
        self,
    ) -> Tuple[Dict[Tuple[str, str], Any], pd.DataFrame, pd.Series]:
        """Trains each model in the models list and returns related data splits.

        Returns:
            Tuple:
                - Dictionary containing trained models.
                - Testing feature set (X_test).
                - Testing labels (y_test).
        """
        df = self.dataloader.load_data(path=self.path)
        df = self.dataloader.transform_data(df=df)
        train_df, test_df = self.resampler.split_train_test_df(
            df=df, seed=self.random_state
        )
        X_train, y_train, X_test, y_test = self.resampler.split_x_y(
            train_df=train_df, test_df=test_df
        )

        trained_models = {}
        for model_name, model in self.models:
            model.fit(X_train, y_train)
            trained_models[(model_name, "Baseline")] = model

        return trained_models, X_test, y_test

    def baseline(self) -> pd.DataFrame:
        """Trains and evaluates each model in the models list on the given dataset.

        This method loads and transforms the dataset, splits it into training and
        testing sets, and evaluates each model in the `self.models` list. Metrics
        such as predictions and probabilities are computed and displayed.

        Returns:
            DataFrame: A DataFrame containing the evaluation metrics for each
                baseline model, with model names as row indices.
        """
        trained_models, X_test, y_test = self.train_baselines()
        results = []

        for model_name, model in self.models:
            preds = trained_models[(model_name, "Baseline")].predict(X_test)
            probs = (
                get_probs(
                    model=trained_models[(model_name, "Baseline")],
                    classification=self.classification,
                    X=X_test,
                )
                if hasattr(model, "predict_proba")
                else None
            )
            metrics = final_metrics(
                classification=self.classification,
                y=y_test,
                preds=preds,
                probs=probs,
            )
            metrics["Model"] = model_name
            results.append(metrics)

        results_df = pd.DataFrame(results).drop(
            columns=["Best Threshold"], errors="ignore"
        )

        results_df, column_order = self._bss_helper(
            results_df, classification=self.classification
        )

        existing_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[
            existing_columns
            + [col for col in results_df.columns if col not in existing_columns]
        ].round(4)

        if self.default_models is not None:
            baseline_order = [
                "Dummy Classifier",
                "Logistic Regression",
                "Random Forest",
            ]
            results_df["Model"] = pd.Categorical(
                results_df["Model"], categories=baseline_order, ordered=True
            )
            results_df = results_df.sort_values("Model").reset_index(drop=True)

        else:
            results_df = results_df.reset_index(drop=True)
        pd.set_option("display.max_columns", None, "display.width", 1000)

        return results_df
