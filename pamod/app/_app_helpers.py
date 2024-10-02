from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns

from pamod.benchmarking import InputProcessor, MultiBenchmarker
from pamod.data import StaticProcessEngine
from pamod.descriptives import DescriptivesPlotter
from pamod.evaluation import FeatureImportanceEngine
from pamod.data import ProcessedDataLoader
from pamod.resampling import Resampler

plotter = None


def load_and_initialize_plotter() -> str:
    """Loads the data and initializes the DescriptivesPlotter.

    Returns:
        str: A message indicating that the data has been loaded and that the
        user can proceed with creating plots.
    """
    global plotter
    engine = StaticProcessEngine(behavior=False, scale=False)
    df = engine.load_data()
    df = engine.process_data(df)
    plotter = DescriptivesPlotter(df)
    return "Data loaded successfully. You can now create plots."


def plot_histogram_2d(column_before: str, column_after: str):
    """Plots a 2D histogram.

    Args:
        column_before (str): Name of column representing values before therapy.
        column_after (str): Name of column representing values after therapy.

    Returns:
        plt.Figure: The 2D histogram plot if successful, or a message prompting
        the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.histogram_2d(column_before, column_after)
    return plt.gcf()


def plot_pocket_comparison(column1: str, column2: str):
    """Plots a pocket depth comparison before and after therapy.

    Args:
        column1 (str): Name of column representing pocket depth before therapy.
        column2 (str): Name of column representing pocket depth after therapy.

    Returns:
        plt.Figure: The pocket comparison plot if successful, or a message prompting
        the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.pocket_comparison(column1, column2)
    return plt.gcf()


def plot_pocket_group_comparison(column_before: str, column_after: str):
    """Plots a pocket group comparison before and after therapy.

    Args:
        column_before (str): Name of column representing pocket group before therapy.
        column_after (str): Name of column representing pocket group after therapy.

    Returns:
        plt.Figure: Pocket group comparison plot if successful, or message prompting
        the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.pocket_group_comparison(column_before, column_after)
    return plt.gcf()


def plot_matrix(vertical: str, horizontal: str):
    """Plots confusion matrix or heatmap based on given vertical and horizontal columns.

    Args:
        vertical (str): The name of column used for the vertical axis.
        horizontal (str): The name of column used for the horizontal axis.

    Returns:
        plt.Figure: The matrix plot if successful, or a message prompting
        the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.plt_matrix(vertical, horizontal)
    return plt.gcf()


def plot_outcome_descriptive(outcome: str, title: str):
    """Plots a descriptive analysis for a given outcome variable.

    Args:
        outcome (str): The name of the outcome column.
        title (str): The title of the plot.

    Returns:
        plt.Figure: The descriptive outcome plot if successful, or a message
        prompting the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.outcome_descriptive(outcome, title)
    return plt.gcf()


def run_benchmarks(
    tasks: list,
    learners: list,
    tuning_methods: list,
    hpo_methods: list,
    criteria: list,
    encoding: list,
    sampling: Optional[str],
    factor: Optional[float],
    n_configs: int,
    racing_folds: int,
    n_jobs: int,
) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure], Optional[plt.Figure]]:
    """Run benchmark evaluations for different configurations.

    Args:
        tasks (list): List of task names to benchmark (e.g., ['pocketclosure',
            'improve']).
        learners (list): List of learners to be evaluated (e.g., ['xgb',
            'logreg']).
        tuning_methods (list): List of tuning methods to apply ('holdout', 'cv').
        hpo_methods (list): List of hyperparameter optimization (HPO) methods
            ('hebo', 'rs').
        criteria (list): List of evaluation criteria ('f1', 'brier_score', etc.).
        encoding (list): List of encoding methods to apply to categorical data
            ('one_hot', 'target').
        sampling (Optional[str]): Sampling strategy to use, if any.
        factor (Optional[float]): Factor to control the resampling process, if
            applicable.
        n_configs (int): Number of configurations for hyperparameter tuning.
        racing_folds (int): Number of folds to use for racing during random
            search (RS).
        n_jobs (int): Number of parallel jobs to run during evaluation.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[plt.Figure], Optional[plt.Figure]]:
            - A DataFrame containing benchmark results.
            - A matplotlib figure showing performance metrics (F1 Score,
              Accuracy, etc.) for each learner.
            - A confusion matrix plot (if available).
    """
    tasks = InputProcessor.process_tasks(tasks)
    learners = InputProcessor.process_learners(learners)
    tuning_methods = InputProcessor.process_tuning(tuning_methods)
    hpo_methods = InputProcessor.process_hpo(hpo_methods)
    criteria = InputProcessor.process_criteria(criteria)
    encodings = InputProcessor.process_encoding(encoding)

    encodings = [e for e in encoding if e in ["One-hot", "Target"]]
    if not encodings:
        raise ValueError("No valid encodings provided.")
    encodings = InputProcessor.process_encoding(encodings)

    sampling = None if sampling == "None" else sampling
    factor = None if factor == "None" else float(factor) if factor else None

    benchmarker = MultiBenchmarker(
        tasks=tasks,
        learners=learners,
        tuning_methods=tuning_methods,
        hpo_methods=hpo_methods,
        criteria=criteria,
        encodings=encodings,
        sampling=sampling,
        factor=factor,
        n_configs=int(n_configs),
        racing_folds=int(racing_folds),
        n_jobs=int(n_jobs),
    )

    df_results, learners_dict = benchmarker.run_all_benchmarks()

    if df_results.empty:
        return "No results to display", None, None

    df_results = df_results.round(4)

    plt.figure(figsize=(6, 4), dpi=300)
    df_results.plot(
        x="Learner",
        y=["F1 Score", "Accuracy", "Brier Score"],
        kind="bar",
        ax=plt.gca(),
    )
    plt.title("Benchmark Metrics for Each Learner")
    plt.xlabel("Learner")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()

    metrics_plot = plt.gcf()

    return df_results, metrics_plot, learners_dict


def load_data(
    task: str, encoding: str
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare the dataset for training and testing.

    Args:
        task (str): Task name (e.g., 'Pocket closure').
        encoding (str): Type of encoding to use (e.g., 'one_hot', 'target').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, y_train, X_test, y_test.
    """
    classification = "multiclass" if task == "pdgrouprevaluation" else "binary"

    dataloader = ProcessedDataLoader(task, encoding)
    df = dataloader.load_data()
    df_transformed = dataloader.transform_data(df)

    resampler = Resampler(classification, encoding)

    train_df, test_df = resampler.split_train_test_df(df_transformed)
    X_train, y_train, X_test, y_test = resampler.split_x_y(train_df, test_df)

    return "Data loaded successfully", X_train, y_train, X_test, y_test


def plot_cm(models, X_test, y_test) -> plt.Figure:
    """Generates a confusion matrix plot for the given model and test data."""
    if not models:
        return "No models available."

    model = list(models.values())[0]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    return plt.gcf()


def plot_fi(
    models: dict, importance_types: List[str], X_test, y_test, encoding: str
) -> plt.Figure:
    """Generates a feature importance plot using FeatureImportanceEngine."""
    if not models:
        return "No models available."

    model = list(models.values())[0]

    fi_engine = FeatureImportanceEngine(
        [model], X_test, y_test, encoding, aggregate=True
    )
    fi_engine.evaluate_feature_importance(importance_types)
    return plt.gcf()
