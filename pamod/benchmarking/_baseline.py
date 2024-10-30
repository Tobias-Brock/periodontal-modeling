from typing import List, Tuple, Union

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..base import BaseConfig
from ..data import ProcessedDataLoader
from ..resampling import Resampler
from ..training import final_metrics, get_probs


class Baseline(BaseConfig):
    """Evaluates baseline models on a given dataset.

    This class loads, preprocesses, and evaluates a set of baseline models on a
    specified dataset. The baseline models include a Random Forest, Logistic
    Regression, and a Dummy Classifier, which are trained and evaluated on
    split data, returning a summary of performance metrics for each model.

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

    Methods:
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
    ) -> None:
        """Initializes the Baseline class with default or user-specified models.

        Args:
            task (str): Task name that determines the classification type.
            encoding (str): Encoding type used in data processing.
            random_state (int, optional): Random state for reproducibility. Defaults
                to 0.
            lr_solver (str, optional): Solver for Logistic Regression. Defaults to
                'saga'.
            dummy_strategy (str, optional): Strategy for the DummyClassifier.
                Defaults to 'prior'.
            models (List[Tuple[str, object]], optional): List of models to use for
                benchmarking. If not provided, default models are initialized.
        """
        self.classification = "multiclass" if task == "pdgrouprevaluation" else "binary"
        self.resampler = Resampler(
            classification=self.classification, encoding=encoding
        )
        self.dataloader = ProcessedDataLoader(task=task, encoding=encoding)
        self.dummy_strategy = dummy_strategy
        self.lr_solver = lr_solver
        self.random_state = random_state

        if models is None:
            self.models = [
                (
                    "Random Forest",
                    RandomForestClassifier(n_jobs=-1, random_state=self.random_state),
                ),
                (
                    "Logistic Regression",
                    LogisticRegression(
                        solver=self.lr_solver, random_state=self.random_state
                    ),
                ),
                (
                    "Dummy Classifier",
                    DummyClassifier(strategy=self.dummy_strategy),
                ),
            ]
        else:
            self.models = models

    def baseline(self) -> pd.DataFrame:
        """Trains and evaluates each model in the models list on the given dataset.

        This method loads and transforms the dataset, splits it into training and
        testing sets, and evaluates each model in the `self.models` list. Metrics
        such as predictions and probabilities are computed and displayed.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics for each
            baseline model, with model names as row indices.
        """
        df = self.dataloader.load_data()
        df = self.dataloader.transform_data(df=df)
        train_df, test_df = self.resampler.split_train_test_df(df=df)
        X_train, y_train, X_test, y_test = self.resampler.split_x_y(
            train_df=train_df, test_df=test_df
        )

        results = []
        model_names = []
        trained_models = {}

        for model_name, model in self.models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = (
                get_probs(model=model, classification=self.classification, X=X_test)
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
            model_names.append(model_name)
            trained_models[(model_name, "Baseline")] = model

        results_df = pd.DataFrame(results)
        baseline_order = ["Dummy Classifier", "Logistic Regression", "Random Forest"]
        results_df["Model"] = pd.Categorical(
            results_df["Model"], categories=baseline_order, ordered=True
        )
        results_df = results_df.sort_values("Model").reset_index(drop=True)
        pd.set_option("display.max_columns", None, "display.width", 1000)

        return results_df
