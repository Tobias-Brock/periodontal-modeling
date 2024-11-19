import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..base import Patient, Side, Tooth, patient_to_df
from ..benchmarking import Baseline, Benchmarker
from ..data import ProcessedDataLoader, StaticProcessEngine
from ..descriptives import DescriptivesPlotter
from ..evaluation import ModelEvaluator
from ..inference import ModelInference
from ..resampling import Resampler
from ._inputprocessor import InputProcessor

plotter = None
data_df = None

all_teeth = [
    18,
    17,
    16,
    15,
    14,
    13,
    12,
    11,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    48,
    47,
    46,
    45,
    44,
    43,
    42,
    41,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
]


def _load_data_engine(
    path: str,
) -> Tuple[str, dict, dict, dict, dict, dict, dict, dict, dict, dict]:
    """Loads the data and returns updates for dropdowns.

    Returns:
        Tuple[str, dict, dict, dict, dict, dict, dict, dict, dict]: A status message and
        updates for each dropdown component.
    """
    global data_df
    engine = StaticProcessEngine(behavior=False)
    data_df = engine.load_data(path=Path(path))
    columns = data_df.columns.tolist()
    dropdown_update = gr.update(choices=columns)
    return (
        "Data loaded successfully. You can now display, process, or save the data.",
        dropdown_update,
        dropdown_update,
        dropdown_update,
        dropdown_update,
        dropdown_update,
        dropdown_update,
        dropdown_update,
        dropdown_update,
        dropdown_update,
    )


def _display_data() -> gr.DataFrame:
    """Displays a preview of the loaded data.

    Returns:
        gr.DataFrame: The first few rows of the loaded data if available, or a message
        indicating that no data has been loaded.
    """
    if data_df is None:
        return "No data loaded. Please load data first."
    return data_df.head(10)


def _process_data() -> Tuple[str, dict, dict, dict, dict, dict, dict, dict, dict, dict]:
    global plotter, data_df
    if data_df is None:
        return ("No data loaded. Please load data first.",) + (gr.update(),) * 9

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        engine = StaticProcessEngine(behavior=False)
        data_df = engine.process_data(df=data_df)
        plotter = DescriptivesPlotter(df=data_df)

    process_output = output.getvalue()
    process_output += "\nData processed successfully. Ready for plotting."
    output.close()
    columns = data_df.columns.tolist()
    default_column_before = "pdbaseline" if "pdbaseline" in columns else columns[0]
    default_column_after = "pdrevaluation" if "pdrevaluation" in columns else columns[0]
    default_column1 = "pdbaseline" if "pdbaseline" in columns else columns[0]
    default_column2 = "pdrevaluation" if "pdrevaluation" in columns else columns[0]
    default_group_column_before = (
        "pdgroupbase" if "pdgroupbase" in columns else columns[0]
    )
    default_group_column_after = (
        "pdgrouprevaluation" if "pdgrouprevaluation" in columns else columns[0]
    )
    default_vertical = (
        "pdgrouprevaluation" if "pdgrouprevaluation" in columns else columns[0]
    )
    default_horizontal = "pdgroupbase" if "pdgroupbase" in columns else columns[0]
    default_outcome = (
        "pdgrouprevaluation" if "pdgrouprevaluation" in columns else columns[0]
    )

    dropdown_update_column_before = gr.update(
        choices=columns, value=default_column_before
    )
    dropdown_update_column_after = gr.update(
        choices=columns, value=default_column_after
    )
    dropdown_update_column1 = gr.update(choices=columns, value=default_column1)
    dropdown_update_column2 = gr.update(choices=columns, value=default_column2)
    dropdown_update_group_column_before = gr.update(
        choices=columns, value=default_group_column_before
    )
    dropdown_update_group_column_after = gr.update(
        choices=columns, value=default_group_column_after
    )
    dropdown_update_vertical = gr.update(choices=columns, value=default_vertical)
    dropdown_update_horizontal = gr.update(choices=columns, value=default_horizontal)
    dropdown_update_outcome = gr.update(choices=columns, value=default_outcome)

    return (
        process_output,
        dropdown_update_column_before,
        dropdown_update_column_after,
        dropdown_update_column1,
        dropdown_update_column2,
        dropdown_update_group_column_before,
        dropdown_update_group_column_after,
        dropdown_update_vertical,
        dropdown_update_horizontal,
        dropdown_update_outcome,
    )


def _save_data(save_path: Union[str, Path]) -> str:
    """Saves the processed data to the specified file path.

    Args:
        save_path (Union[str, Path]): The full path where the processed data should
        be saved, including the file name.

    Returns:
        str: A message indicating that the data was successfully saved to the specified
        path. If no data is loaded, returns an error message.
    """
    global data_df
    if data_df is None:
        return "No data loaded. Please load data first."

    engine = StaticProcessEngine(behavior=False)
    if isinstance(save_path, str):
        save_path = Path(save_path)

    engine.save_data(data_df, path=save_path)
    return f"Data saved successfully to {save_path}"


def _plot_histogram_2d(
    col_before: str, col_after: str, x_label: str, y_label: str
) -> Union[plt.Figure, str]:
    """Plots a 2D histogram.

    Args:
        col_before (str): Name of the column representing values before therapy.
        col_after (str): Name of the column representing values after therapy.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.

    Returns:
        Union[plt.Figure, str]: The 2D histogram plot if successful, or a message
        prompting the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.histogram_2d(
        col_before=col_before, col_after=col_after, x_label=x_label, y_label=y_label
    )
    return plt.gcf()


def _plot_pocket_comparison(
    col1: str, col2: str, title_1: str, title_2: str
) -> Union[plt.Figure, str]:
    """Plots a pocket depth comparison before and after therapy.

    Args:
        col1 (str): Name of column representing pocket depth before therapy.
        col2 (str): Name of column representing pocket depth after therapy.
        title_1 (str): Title of first plot.
        title_2 (str): Title of second plot.

    Returns:
        Union[plt.Figure, str]: The pocket comparison plot if successful, or
        a message prompting the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.pocket_comparison(col1=col1, col2=col2, title_1=title_1, title_2=title_2)
    return plt.gcf()


def _plot_pocket_group_comparison(
    col_before: str, col_after: str, title_1: str, title_2: str
) -> Union[plt.Figure, str]:
    """Plots a pocket group comparison before and after therapy.

    Args:
        col_before (str): Name of column representing pocket group before therapy.
        col_after (str): Name of column representing pocket group after therapy.
        title_1 (str): Title of first plot.
        title_2 (str): Title of second plot.

    Returns:
        Union[plt.Figure, str]: Pocket group comparison plot if successful,
        or message prompting the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.pocket_group_comparison(
        col_before=col_before, col_after=col_after, title_1=title_1, title_2=title_2
    )
    return plt.gcf()


def _plot_matrix(
    vertical: str, horizontal: str, x_label: str, y_label: str
) -> Union[plt.Figure, str]:
    """Plots confusion matrix or heatmap based on given vertical and horizontal columns.

    Args:
        vertical (str): The name of the column used for the vertical axis.
        horizontal (str): The name of the column used for the horizontal axis.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.

    Returns:
        Union[plt.Figure, str]: The matrix plot if successful, or a message
        prompting the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.plt_matrix(
        vertical=vertical, horizontal=horizontal, x_label=x_label, y_label=y_label
    )
    return plt.gcf()


def _plot_outcome_descriptive(outcome: str, title: str) -> Union[plt.Figure, str]:
    """Plots a descriptive analysis for a given outcome variable.

    Args:
        outcome (str): The name of the outcome column.
        title (str): The title of the plot.

    Returns:
        Union[plt.Figure, str]: The descriptive outcome plot if successful, or
        a message prompting the user to load the data first.
    """
    global plotter
    if plotter is None:
        return "Please load the data first."
    plotter.outcome_descriptive(outcome=outcome, title=title)
    return plt.gcf()


def _baseline_wrapper(task: str, encoding: str, path: str) -> pd.DataFrame:
    """Wraps the initialization and execution of the Baseline class.

    Args:
        task (str): Task name to be processed by the input processor.
        encoding (str): Encoding type to be processed by the input processor.
        path (str): Directory path for data loading.

    Returns:
        pd.DataFrame: DataFrame containing baseline evaluation metrics.
    """
    task = InputProcessor.process_task(task=task)
    encoding = InputProcessor.process_encoding(encoding=encoding)
    return Baseline(task=task, encoding=encoding, path=Path(path)).baseline()


def _update_tuning_method_fields(tuning_method: str) -> Tuple[Any, Any, Any]:
    """Update fields based on the selected tuning method.

    Args:
        tuning_method (str): The selected tuning method ("Cross-Validation"
            or "Holdout").

    Returns:
        Tuple[Any, Any, Any]: Updates for cv_folds_input, cv_seed_input,
            val_size_input.
    """
    if tuning_method == "Cross-Validation":
        return (
            gr.update(interactive=True, value=10),
            gr.update(interactive=True, value=0),
            gr.update(interactive=False, value=None),
        )
    elif tuning_method == "Holdout":
        return (
            gr.update(interactive=False, value=None),
            gr.update(interactive=False, value=None),
            gr.update(interactive=True, value=0.2),
        )
    else:
        return (
            gr.update(interactive=False, value=None),
            gr.update(interactive=False, value=None),
            gr.update(interactive=False, value=None),
        )


def _update_hpo_method_fields(tuning_method: str, hpo_method: str) -> Any:
    """Update racing_folds_input based on tuning and HPO methods.

    Args:
        tuning_method (str): The selected tuning method.
        hpo_method (str): The selected hyperparameter optimization method.

    Returns:
        Any: Update for racing_folds_input.
    """
    if tuning_method == "Cross-Validation" and hpo_method == "Random Search":
        return gr.update(interactive=True, value=5)
    else:
        return gr.update(interactive=False, value=None)


def _update_learners_fields(learners: List[str]) -> Any:
    """Update mlp_flag_input based on selected learners.

    Args:
        learners (List[str]): List of selected learners.

    Returns:
        Any: Update for mlp_flag_input.
    """
    if "Multilayer Perceptron" in learners:
        return gr.update(interactive=True, value=True)
    else:
        return gr.update(interactive=False, value=None)


def _update_criteria_fields(criteria: str) -> Any:
    """Update threshold_tuning_input based on selected criteria.

    Args:
        criteria (str): The selected evaluation criteria.

    Returns:
        Any: Update for threshold_tuning_input.
    """
    if criteria == "F1 Score":
        return gr.update(interactive=True, value=True)
    else:
        return gr.update(interactive=False, value=None)


def _update_task_fields(task: str) -> Any:
    """Update criteria_input choices based on selected task.

    Args:
        task (str): The selected task.

    Returns:
        Any: Update for criteria_input.
    """
    if task == "Pocket groups":
        return gr.update(
            choices=["Macro F1 Score", "Brier Score"], value="Macro F1 Score"
        )
    else:
        return gr.update(choices=["F1 Score", "Brier Score"], value="F1 Score")


def _initialize_benchmark(
    tuning_method_value: str,
    hpo_method_value: str,
    learners_value: List[str],
    criteria_value: str,
    task_value: str,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Initialize component states based on their default values.

    Args:
        tuning_method_value (str): The selected tuning method.
        hpo_method_value (str): The selected hyperparameter optimization method.
        learners_value (List[str]): List of selected learners.
        criteria_value (str): The selected evaluation criteria.
        task_value (str): The selected task.

    Returns:
        Tuple[Any, Any, Any, Any, Any, Any, Any]: Updates for cv_folds_input,
            cv_seed_input, val_size_input, racing_folds_input, mlp_flag_input,
            threshold_tuning_input, criteria_input.
    """
    tuning_updates = _update_tuning_method_fields(tuning_method_value)
    cv_folds_update = tuning_updates[0]
    cv_seed_update = tuning_updates[1]
    val_size_update = tuning_updates[2]
    racing_folds_update = _update_hpo_method_fields(
        tuning_method_value, hpo_method_value
    )
    mlp_flag_update = _update_learners_fields(learners_value)
    threshold_tuning_update = _update_criteria_fields(criteria_value)
    criteria_update = _update_task_fields(task_value)

    return (
        cv_folds_update,
        cv_seed_update,
        val_size_update,
        racing_folds_update,
        mlp_flag_update,
        threshold_tuning_update,
        criteria_update,
    )


def _run_benchmarks(
    task: str,
    learners: List[str],
    tuning_method: str,
    hpo_method: str,
    criterion: str,
    encoding: str,
    sampling: Optional[List[Optional[str]]],
    factor: Optional[float],
    n_configs: int,
    cv_folds: Optional[int],
    racing_folds: Optional[int],
    test_seed: int,
    test_size: float,
    val_size: Optional[float],
    cv_seed: Optional[int],
    mlp_flag: Optional[bool],
    threshold_tuning: Optional[bool],
    n_jobs: int,
    path: str,
) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure], Optional[plt.Figure]]:
    """Run benchmark evaluations for different configurations.

    Args:
        task (str): Task name to benchmark (e.g., 'pocketclosure',
            or 'improve').
        learners (list): List of learners to be evaluated (e.g., ['xgb',
            'logreg']).
        tuning_method (str): Tuning method to apply ('holdout', 'cv').
        hpo_method (str): Hyperparameter optimization (HPO) method
            ('hebo', 'rs').
        criterion (str): Evaluation criteria ('f1', 'brier_score', etc.).
        encoding (str): Encoding methods to apply to categorical data
            ('one_hot', 'target').
        sampling (Optional[List[Optional[str]]]): Sampling strategy to use, if any.
        factor (Optional[float]): Factor to control the resampling process, if
            applicable.
        n_configs (int): Number of configurations for hyperparameter tuning.
        cv_folds (Optional[int]): Number of cross-validation folds.
        racing_folds (Optional[int]): Number of folds to use for racing during random
            search (RS).
        test_seed (int): Seed for random train-test split.
        test_size (float): Proportion of data used for testing.
        val_size (Optional[float]): Proportion of data for validation in holdout.
        cv_seed (int): Seed for cross-validation splits.
        mlp_flag (Optional[bool]): Enables MLP training with early stopping.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        n_jobs (int): Number of parallel jobs to run during evaluation.
        path (str): The file path where data is stored.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[plt.Figure], Optional[plt.Figure]]:
            - A DataFrame containing benchmark results.
            - A matplotlib figure showing performance metrics (F1 Score,
              Accuracy, etc.) for each learner.
            - A confusion matrix plot (if available).
    """
    task = InputProcessor.process_task(task=task)
    learners = InputProcessor.process_learners(learners=learners)
    tuning_methods = InputProcessor.process_tuning(tuning_method=tuning_method)
    hpo_methods = InputProcessor.process_hpo(hpo_method=hpo_method)
    criteria = InputProcessor.process_criteria(criterion=criterion)
    encodings = [InputProcessor.process_encoding(encoding=encoding)]
    sampling_benchmark: Optional[List[Union[str, None]]] = (
        [s if s != "None" else None for s in sampling] if sampling else None
    )
    factor = None if factor == "" or factor is None else float(factor)

    factor = float(factor) if factor is not None else None
    cv_folds = int(cv_folds) if cv_folds is not None else None
    racing_folds = int(racing_folds) if racing_folds is not None else None
    val_size = float(val_size) if val_size is not None else None
    cv_seed = int(cv_seed) if cv_seed is not None else None
    mlp_flag = bool(mlp_flag) if mlp_flag is not None else None
    threshold_tuning = bool(threshold_tuning) if threshold_tuning is not None else None

    benchmarker = Benchmarker(
        task=task,
        learners=learners,
        tuning_methods=tuning_methods,
        hpo_methods=hpo_methods,
        criteria=criteria,
        encodings=encodings,
        sampling=sampling_benchmark,
        factor=factor,
        n_configs=int(n_configs),
        cv_folds=cv_folds,
        racing_folds=racing_folds,
        test_size=test_size,
        test_seed=test_seed,
        val_size=val_size,
        cv_seed=cv_seed,
        mlp_flag=mlp_flag,
        threshold_tuning=threshold_tuning,
        n_jobs=int(n_jobs),
        path=Path(path),
    )

    df_results, learners_dict = benchmarker.run_benchmarks()

    if df_results.empty:
        return "No results to display", None, None

    df_results = df_results.round(4)
    classification = "multiclass" if "pdgrouprevaluation" in task else "binary"
    available_metrics = df_results.columns.tolist()

    if classification == "binary":
        metrics_to_plot = ["F1 Score", "Accuracy", "Brier Score"]
    else:
        metrics_to_plot = ["Macro F1", "Accuracy", "Multiclass Brier Score"]

    metrics_to_plot = [
        metric for metric in metrics_to_plot if metric in available_metrics
    ]

    if not metrics_to_plot:
        raise ValueError("No matching metrics found in results to plot.")

    df_long = df_results.melt(
        id_vars="Learner",
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(5, 4), dpi=300)
    ax = sns.barplot(
        data=df_long, x="Metric", y="Score", hue="Learner", edgecolor="black"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.title("Benchmark Metrics Comparison by Metric", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(
        title="Learner",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=10,
        title_fontsize=10,
        frameon=False,
    )
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(
            container, fmt="%.2f", label_type="edge", rotation=90, fontsize=9, padding=5
        )

    metrics_plot = plt.gcf()

    return df_results, metrics_plot, learners_dict


def _benchmarks_wrapper(*args: Any) -> Any:
    """Wrapper function to pass arguments to the run_benchmarks function.

    Args:
        *args (Any): Arguments passed into the function, corresponding to:
            - task (str): The task to be executed.
            - learners (List[str]): List of learners.
            - tuning_methods (List[str]): List of tuning methods.
            - hpo_methods (List[str]): List of hyperparameter optimization methods.
            - criteria (List[str]): List of evaluation criteria.
            - encodings (List[str]): List of encodings for categorical features.
            - sampling (Optional[List[Optional[str]]]): Sampling strategies.
            - factor (Optional[float]): Factor for the resampling process.
            - n_configs (int): Number of configurations for HPO.
            - racing_folds (int): Number of folds for racing during random search.
            - cv_folds (Optional[int]): Number of cross-validation folds.
            - test_seed (Optional[int]): Seed for random train-test split.
            - test_size (Optional[float]): Proportion of data used for testing.
            - val_size (Optional[float]): Proportion of data used for validation in
                holdout.
            - cv_seed (Optional[int]): Seed for cross-validation splits.
            - mlp_flag (Optional[bool]): Enables MLP training with early stopping.
            - threshold_tuning (bool): Enables threshold tuning for binary
                classification.
            - n_jobs (int): Number of parallel jobs to run during evaluation.
            - path (str): The file path where data is stored.

    Returns:
        Any: Result from the run_benchmarks function.
    """
    (
        task,
        learners,
        tuning_method,
        hpo_method,
        criterion,
        encoding,
        sampling,
        factor,
        n_configs,
        cv_folds,
        racing_folds,
        test_seed,
        test_size,
        val_size,
        cv_seed,
        mlp_flag,
        threshold_tuning,
        n_jobs,
        path,
    ) = args

    return _run_benchmarks(
        task=task,
        learners=learners,
        tuning_method=tuning_method,
        hpo_method=hpo_method,
        criterion=criterion,
        encoding=encoding,
        sampling=sampling,
        factor=factor,
        n_configs=n_configs,
        cv_folds=cv_folds,
        racing_folds=racing_folds,
        test_seed=test_seed,
        test_size=test_size,
        val_size=val_size,
        cv_seed=cv_seed,
        mlp_flag=mlp_flag,
        threshold_tuning=threshold_tuning,
        n_jobs=n_jobs,
        path=path,
    )


def _load_data(
    task: str, encoding: str, path: str
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare the dataset for training and testing.

    Args:
        task (str): Task name (e.g., 'Pocket closure').
        encoding (str): Type of encoding to use (e.g., 'one_hot', 'target').
        path (str): Input path for dataloading.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, y_train, X_test, y_test.
    """
    classification = "multiclass" if task == "pdgrouprevaluation" else "binary"
    dataloader = ProcessedDataLoader(task=task, encoding=encoding)
    df = dataloader.load_data(path=Path(path))
    df_transformed = dataloader.transform_data(df=df)
    resampler = Resampler(classification, encoding)

    train_df, test_df = resampler.split_train_test_df(df=df_transformed)
    X_train, y_train, X_test, y_test = resampler.split_x_y(
        train_df=train_df, test_df=test_df
    )

    return "Data loaded successfully", train_df, X_train, y_train, X_test, y_test


def _load_data_wrapper(
    task: str, encoding: str, path: str
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wraooer to load data.

    Args:
        task (str)): Task input from UI.
        encoding (str): Encoding type.
        path (str): Input path for dataloading.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, y_train, X_test, y_test.
    """
    return _load_data(
        task=InputProcessor.process_task(task=task),
        encoding=InputProcessor.process_encoding(encoding=encoding),
        path=path,
    )


def _update_model_dropdown(models: Dict[str, Any]) -> dict:
    """Updates the model dropdown options based on the provided models.

    Args:
        models (Dict[str, Any]): Dictionary containing model keys and models.

    Returns:
        dict: A dictionary with updated dropdown choices and selected value.
    """
    model_keys = list(models.keys()) if models else []
    return gr.update(choices=model_keys, value=model_keys[0] if model_keys else None)


def _plot_cm(
    models: dict, selected_model: str, X: pd.DataFrame, y: pd.Series, task: str
) -> plt.Figure:
    """Generates a confusion matrix plot for the given model and test data.

    Args:
        models (dict): Dictionary of trained models where the keys are model names.
        selected_model (str): Name of the selected model from the dropdown.
        X (pd.DataFrame): Test features.
        y (pd.Series): True labels for the test set.
        task (str): Task mapping for outcome labels.

    Returns:
        plt.Figure: Confusion matrix heatmap plot.
    """
    ModelEvaluator(model=models[selected_model], X=X, y=y).plot_confusion_matrix(
        task=InputProcessor.process_task(task=task)
    )
    return plt.gcf()


def _plot_calibration(
    models: dict, selected_model: str, X: pd.DataFrame, y: pd.Series, task: str
) -> plt.Figure:
    """Generates a calibration plot for the given model and test data.

    Args:
        models (dict): Dictionary of trained models where the keys are model names.
        selected_model (str): Name of the selected model from the dropdown.
        X (pd.DataFrame): Test features.
        y (pd.Series): True labels for the test set.
        task (str): Task mapping for outcome labels.

    Returns:
        plt.Figure: Calibration plot for model probabilities.
    """
    ModelEvaluator(model=models[selected_model], X=X, y=y).calibration_plot(
        task=InputProcessor.process_task(task=task), tight_layout=True
    )
    return plt.gcf()


def _plot_bss(
    models: dict,
    selected_model: str,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    encoding: str,
    path: str,
) -> plt.Figure:
    """Generates BSS plot for the given model and test data.

    Args:
        models (dict): Dictionary of trained models where the keys are model names.
        selected_model (str): Name of the selected model from the dropdown.
        X (pd.DataFrame): Test features.
        y (pd.Series): True labels for the test set.
        task (str): Task mapping for outcome labels.
        encoding (str): Type of encoding.
        path (str): Input path from app.

    Returns:
        plt.Figure: Calibration plot for model probabilities.
    """
    baseline = Baseline(
        task=InputProcessor.process_task(task=task),
        encoding=InputProcessor.process_encoding(encoding=encoding),
        path=Path(path),
    )
    baseline_models, _, _ = baseline.train_baselines()
    classification = "multiclass" if task == "Pocket groups" else "binary"
    ModelEvaluator(model=models[selected_model], X=X, y=y).bss_comparison(
        baseline_models=baseline_models,
        classification=classification,
        tight_layout=True,
    )
    return plt.gcf()


def _plot_fi(
    model: Any,
    fi_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    encoding: str,
    aggregate: bool,
) -> plt.Figure:
    """Generates a feature importance plot using FeatureImportanceEngine.

    Args:
        model (Any): Trained model for feature importance extraction.
        fi_type (str): List of feature importance types to plot.
        X (pd.DataFrame): Test features.
        y (pd.Series): True labels for the test set.
        encoding (str): The encoding method used during preprocessing.
        aggregate (bool): Aggregates multi-category features.

    Returns:
        plt.Figure: Feature importance plot.
    """
    if not model:
        return "No model available."

    ModelEvaluator(
        model=model,
        X=X,
        y=y,
        encoding=InputProcessor.process_encoding(encoding=encoding),
        aggregate=aggregate,
    ).evaluate_feature_importance(fi_types=[fi_type])
    return plt.gcf()


def _plot_cluster(
    model: tuple[Any, Any],
    X: pd.DataFrame,
    y: pd.Series,
    encoding: str,
    aggregate: bool,
    n_clusters: int,
) -> Union[None, Tuple[plt.Figure, plt.Figure, pd.DataFrame]]:
    """Performs clustering on Brier score and returns related plots.

    Args:
        model (Any): Trained model for cluster analysis.
        X (pd.DataFrame): Test features.
        y (pd.Series): True labels for the test set.
        encoding (str): The encoding method used during preprocessing.
        aggregate (bool): Aggregates multi-category features.
        n_clusters (int): Number of clusters for Brier score analysis.

    Returns:
        Union[None, Tuple[plt.Figure, plt.Figure, pd.DataFrame]]: A tuple containing the
        Brier score plot, heatmap plot and the clustered data. Returns None if model is
        None.
    """
    if not model:
        return "No model available."

    return ModelEvaluator(
        model=model,
        X=X,
        y=y,
        encoding=InputProcessor.process_encoding(encoding=encoding),
        aggregate=aggregate,
    ).analyze_brier_within_clusters(n_clusters=n_clusters, tight_layout=True)


def _brier_score_wrapper(
    models: dict, selected_model: str, X: pd.DataFrame, y: pd.Series, task: str
) -> plt.Figure:
    """Wrapper function to generate Brier score plots.

    Args:
        models (dict): Dictionary of trained models where the keys are model names.
        selected_model (str): Name of the selected model from the dropdown.
        X (pd.DataFrame): Test dataset containing input features.
        y (pd.Series): Test dataset containing true labels.
        task (str): Task mapping for outcome labels.

    Returns:
        plt.Figure: Matplotlib figure showing the Brier score plot.
    """
    ModelEvaluator(model=models[selected_model], X=X, y=y).brier_score_groups(
        task=InputProcessor.process_task(task=task), tight_layout=True
    )
    return plt.gcf()


def _plot_fi_wrapper(
    models: dict,
    selected_model: str,
    fi_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    encoding: str,
    aggregate: bool,
) -> Any:
    """Wrapper function to call plot_fi.

    Args:
        models (dict): Dictionary containing models.
        selected_model (str): The key to access the selected model in the dict.
        fi_type (str): List of importance types.
        X (pd.DataFrame): Test dataset containing input features.
        y (pd.Series): Test dataset containing true labels.
        encoding (str): The encoding method used.
        aggregate (bool): Aggregates multi-category features.

    Returns:
        Any: The result from the plot_fi function.
    """
    return _plot_fi(
        model=models[selected_model],
        fi_type=fi_type,
        X=X,
        y=y,
        encoding=encoding,
        aggregate=aggregate,
    )


def _plot_cluster_wrapper(
    models: dict,
    selected_model: str,
    X: pd.DataFrame,
    y: pd.Series,
    encoding: str,
    aggregate: bool,
    n_clusters: int,
) -> Union[None, Tuple[plt.Figure, plt.Figure, pd.DataFrame]]:
    """Wrapper function to call plot_cluster.

    Args:
        models (dict): Dictionary containing models.
        selected_model (str): The key to access the selected model in the dict.
        X (pd.DataFrame): Test dataset containing input features.
        y (pd.Series): Test dataset containing true labels.
        encoding (str): The encoding method used.
        aggregate (bool): Aggregates multi-category features.
        n_clusters (int): Number of clusters.

    Returns:
        Union[None, Tuple[plt.Figure, plt.Figure, pd.DataFrame]]: A tuple containing the
        Brier score plot, heatmap plot and the clustered data. Returns None if model is
        None.
    """
    return _plot_cluster(
        model=models[selected_model],
        X=X,
        y=y,
        encoding=encoding,
        aggregate=aggregate,
        n_clusters=n_clusters,
    )


def _handle_tooth_selection(
    selected_tooth: Union[str, int],
    tooth_states_value: Dict[str, Any],
    tooth_components: Dict[str, gr.components.Component],
    sides_components: Dict[int, Dict[str, gr.components.Component]],
) -> List[Any]:
    """Handle the selection of a tooth and update UI components accordingly.

    Args:
        selected_tooth: The selected tooth number as a string or integer.
        tooth_states_value: A dictionary storing the state of each tooth.
        tooth_components: A dictionary of tooth-level UI components.
        sides_components: A dictionary of side-level UI components for each side.

    Returns:
        A list of updates for the UI components to reflect the selected tooth's data.
    """
    selected_tooth = str(selected_tooth)
    tooth_updates = []
    tooth_data = tooth_states_value.get(selected_tooth, {})

    for key, _ in tooth_components.items():
        value = tooth_data.get(key, None)
        tooth_updates.append(gr.update(value=value))

    side_updates = []
    for side_num in range(1, 7):
        side_data = tooth_data.get("sides", {}).get(str(side_num), {})
        side_components = sides_components[side_num]
        for key, _ in side_components.items():
            value = side_data.get(key, None)
            side_updates.append(gr.update(value=value))

    return tooth_updates + side_updates


def _update_tooth_state(
    tooth_states_value: Dict[str, Any],
    selected_tooth: Union[str, int],
    input_value: Any,
    input_name: str,
) -> Dict[str, Any]:
    """Update the state of a tooth when a tooth-level input changes.

    Args:
        tooth_states_value: A dictionary storing the state of each tooth.
        selected_tooth: The selected tooth number as a string or integer.
        input_value: The new value of the input that changed.
        input_name: The name of the input field.

    Returns:
        The updated tooth_states_value dictionary.
    """
    selected_tooth = str(selected_tooth)
    if selected_tooth not in tooth_states_value:
        tooth_states_value[selected_tooth] = {}
    tooth_states_value[selected_tooth][input_name] = input_value
    return tooth_states_value


def _update_side_state(
    tooth_states_value: Dict[str, Any],
    selected_tooth: Union[str, int],
    input_value: Any,
    side_num: int,
    input_name: str,
) -> Dict[str, Any]:
    """Update the state of a side when a side-level input changes.

    Args:
        tooth_states_value: A dictionary storing the state of each tooth.
        selected_tooth: The selected tooth number as a string or integer.
        input_value: The new value of the input that changed.
        side_num: The side number (1-6) where the input changed.
        input_name: The name of the input field.

    Returns:
        The updated tooth_states_value dictionary.
    """
    selected_tooth = str(selected_tooth)
    side_num_str = str(side_num)
    if selected_tooth not in tooth_states_value:
        tooth_states_value[selected_tooth] = {}
    if "sides" not in tooth_states_value[selected_tooth]:
        tooth_states_value[selected_tooth]["sides"] = {}
    if side_num_str not in tooth_states_value[selected_tooth]["sides"]:
        tooth_states_value[selected_tooth]["sides"][side_num_str] = {}
    tooth_states_value[selected_tooth]["sides"][side_num_str][input_name] = input_value
    return tooth_states_value


def _collect_data(
    age: Union[int, float],
    gender: str,
    bmi: float,
    perio_history: str,
    diabetes: str,
    smokingtype: str,
    cigarettenumber: int,
    antibiotics: str,
    stresslvl: str,
    tooth_states_value: Dict[str, Any],
) -> Tuple[str, pd.DataFrame]:
    """Collect data from the inputs and construct a Patient object and DataFrame.

    Args:
        age (Union[int, float]): The age of the patient.
        gender (str): The gender of the patient.
        bmi (float): The body mass index of the patient.
        perio_history (str): The periodontal family history.
        diabetes (str): The diabetes status.
        smokingtype (str): The smoking type.
        cigarettenumber (int): The number of cigarettes.
        antibiotics (str): The antibiotic treatment status.
        stresslvl (str): The stress level.
        tooth_states_value (Dict[str, Any]): Dictionary storing the state of each tooth.

    Returns:
        A tuple containing a success message and the patient data as a DataFrame.
    """
    patient = Patient(
        age=int(age),
        gender=int(InputProcessor.process_gender(gender=gender)),
        bodymassindex=float(bmi),
        periofamilyhistory=int(
            InputProcessor.process_periohistory(periohistory=perio_history)
        ),
        diabetes=int(InputProcessor.process_diabetes(diabetes=diabetes)),
        smokingtype=int(InputProcessor.process_smokingtype(smokingtype=smokingtype)),
        cigarettenumber=int(cigarettenumber),
        antibiotictreatment=int(
            InputProcessor.process_antibotics(antibiotics=antibiotics)
        ),
        stresslvl=int(InputProcessor.process_stresslvl(stresslvl=stresslvl)),
        teeth=[],
    )
    for tooth_str, tooth_data in tooth_states_value.items():
        tooth_number = int(tooth_str)
        if tooth_number in [11, 12, 21, 22, 31, 32, 41, 42, 13, 23, 33, 43]:
            toothtype = 0
            rootnumber = 0
        elif tooth_number in [14, 15, 24, 25, 34, 35, 44, 45]:
            toothtype = 1
            rootnumber = 1
        else:
            toothtype = 2
            rootnumber = 1

        tooth_has_data = any(
            tooth_data.get(key) is not None
            for key in ["mobility", "restoration", "percussion", "sensitivity"]
        )

        sides = []
        for side_num_str, side_data in tooth_data.get("sides", {}).items():
            side_has_data = any(
                side_data.get(key) not in (None, "", "None")
                for key in [
                    "furcationbaseline",
                    "pdbaseline",
                    "recbaseline",
                    "plaque",
                    "bop",
                ]
            )
            if side_has_data:
                side_obj = Side(
                    furcationbaseline=int(
                        InputProcessor.process_furcation(
                            side_data.get("furcationbaseline")
                        )
                    ),
                    side=int(side_num_str),
                    pdbaseline=int(side_data.get("pdbaseline")),
                    recbaseline=int(side_data.get("recbaseline")),
                    plaque=int(InputProcessor.process_plaque(side_data.get("plaque"))),
                    bop=int(InputProcessor.process_bop(side_data.get("bop"))),
                )
                sides.append(side_obj)

        if tooth_has_data or sides:
            tooth_obj = Tooth(
                tooth=tooth_number,
                toothtype=toothtype,
                rootnumber=rootnumber,
                mobility=(
                    int(InputProcessor.process_mobility(tooth_data.get("mobility")))
                    if tooth_data.get("mobility") is not None
                    else None
                ),
                restoration=(
                    int(
                        InputProcessor.process_restoration(
                            tooth_data.get("restoration")
                        )
                    )
                    if tooth_data.get("restoration") is not None
                    else None
                ),
                percussion=(
                    int(InputProcessor.process_percussion(tooth_data.get("percussion")))
                    if tooth_data.get("percussion") is not None
                    else None
                ),
                sensitivity=(
                    int(
                        InputProcessor.process_sensitivity(
                            tooth_data.get("sensitivity")
                        )
                    )
                    if tooth_data.get("sensitivity") is not None
                    else None
                ),
                sides=sides,
            )
            patient.teeth.append(tooth_obj)

    patient_df = patient_to_df(patient=patient)
    print("Collected Patient Data:\n", patient_df)
    return "Patient data collected successfully!", patient_df


def _app_inference(
    task: str,
    models: dict,
    selected_model: str,
    patient_data: pd.DataFrame,
    encoding: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run inference on the patient's data.

    Args:
        task (str): The task name for which the model was trained.
        models (dict): A dictionary containing the trained models.
        selected_model (str): The name of the selected model for inference.
        patient_data (pd.DataFrame): The patient's data as a DataFrame.
        encoding (str): Encoding type ("one_hot" or "target").
        X_train (pd.DataFrame): Training features for target encoding.
        y_train (pd.Series): Training target for target encoding.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - predict_data (pd.DataFrame): The transformed patient data for prediction.
            - output_data (pd.DataFrame): DataFrame with columns "tooth", "side",
              transformed "prediction", and "probability".
            - results (pd.DataFrame): Original results from the model inference.
    """
    model = models[selected_model]
    task_processed = InputProcessor.process_task(task=task)
    classification = (
        "multiclass" if task_processed == "pdgrouprevaluation" else "binary"
    )
    inference_engine = ModelInference(classification, model)
    predict_data, patient_data = inference_engine.prepare_inference(
        task=task,
        patient_data=patient_data,
        encoding=InputProcessor.process_encoding(encoding=encoding),
        X_train=X_train,
        y_train=y_train,
    )

    prediction_data, prediction_output, results = inference_engine.patient_inference(
        predict_data=predict_data, patient_data=patient_data
    )
    prediction_output = InputProcessor.transform_predictions(
        task_processed, prediction_output
    )
    return prediction_data, prediction_output, results


def _run_jackknife_inference(
    task: str,
    models: dict,
    selected_model: str,
    train_df: pd.DataFrame,
    patient_data: pd.DataFrame,
    encoding: str,
    inference_results: pd.DataFrame,
    alpha: float,
    sample_fraction: float,
    n_jobs: int,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """Run jackknife inference and generate confidence intervals and plots.

    Args:
        task (str): Task name.
        models (dict): Dictionary of trained models.
        selected_model (str): Name of the selected model.
        train_df (pd.DataFrame): Training DataFrame.
        patient_data (pd.DataFrame): Patient data to predict on.
        encoding (str): Encoding type.
        inference_results (pd.DataFrame): Original inference results.
        alpha (float, optional): Significance level for confidence intervals.
        sample_fraction (float, optional): Fraction of patient IDs to use in jackknife.
            Defaults to 1.0.
        n_jobs (int): Number of parallel jobs. Defaults to -1.

    Returns:
        Tuple[pd.DataFrame, plt.Figure]: Jackknife results and the plot.
    """
    classification = "multiclass" if task == "Pocket groups" else "binary"
    model = models[selected_model]

    inference_engine = ModelInference(classification=classification, model=model)
    _, ci_plot = inference_engine.jackknife_inference(
        model=model,
        train_df=train_df,
        patient_data=patient_data,
        encoding=InputProcessor.process_encoding(encoding=encoding),
        inference_results=inference_results,
        alpha=alpha,
        sample_fraction=sample_fraction,
        n_jobs=n_jobs,
    )

    return ci_plot
