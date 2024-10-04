from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from pamod.base import InferenceInput, create_predict_data
from pamod.benchmarking import Benchmarker, InputProcessor
from pamod.data import ProcessedDataLoader, StaticProcessEngine
from pamod.descriptives import DescriptivesPlotter
from pamod.evaluation import FeatureImportanceEngine, brier_score_groups
from pamod.inference import ModelInference
from pamod.resampling import Resampler

plotter = None


def load_and_initialize_plotter(path: str) -> str:
    """Loads the data and initializes the DescriptivesPlotter.

    Args:
        path (str): The full path to the data file.

    Returns:
        str: A message indicating that the data has been loaded and that the
        user can proceed with creating plots.
    """
    global plotter
    data_path = Path(path)
    engine = StaticProcessEngine(behavior=False, scale=False)
    df = engine.load_data(path=data_path.parent, name=data_path.name)
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
    path: str,
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
        path (str): The file path where data is stored.

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

    data_path = Path(path)
    file_path = data_path.parent
    file_name = data_path.name

    benchmarker = Benchmarker(
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
        path=Path(file_path),
        name=file_name,
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


def benchmarks_wrapper(*args: Any) -> Any:
    """Wrapper function to pass arguments to the run_benchmarks function.

    Args:
        *args (Any): Arguments passed into the function, corresponding to:
            - task (str): The task to be executed.
            - learners (List[str]): List of learners.
            - tuning_methods (List[str]): List of tuning methods.
            - hpo_methods (List[str]): List of hyperparameter optimization methods.
            - criteria (List[str]): List of evaluation criteria.
            - encodings (List[str]): List of encodings for categorical features.
            - sampling (Optional[str]): The sampling method.
            - factor (Optional[float]): The sampling factor.
            - n_configs (int): Number of configurations.
            - racing_folds (int): Number of folds for racing methods.
            - n_jobs (int): Number of jobs to run in parallel.
            - path (str): The file path where data is stored.

    Returns:
        Any: Result from the run_benchmarks function.
    """
    (
        task,
        learners,
        tuning_methods,
        hpo_methods,
        criteria,
        encodings,
        sampling,
        factor,
        n_configs,
        racing_folds,
        n_jobs,
        path,
    ) = args
    return run_benchmarks(
        [task],
        learners,
        tuning_methods,
        hpo_methods,
        criteria,
        encodings,
        sampling,
        factor,
        n_configs,
        racing_folds,
        n_jobs,
        path,
    )


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


def update_model_dropdown(models: Dict[str, Any]) -> dict:
    """Updates the model dropdown options based on the provided models.

    Args:
        models (Dict[str, Any]): Dictionary containing model keys and models.

    Returns:
        dict: A dictionary with updated dropdown choices and selected value.
    """
    model_keys = list(models.keys()) if models else []
    return gr.update(choices=model_keys, value=model_keys[0] if model_keys else None)


def plot_cm(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    """Generates a confusion matrix plot for the given model and test data.

    Args:
        model (Any): Trained model for generating predictions.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.

    Returns:
        plt.Figure: Confusion matrix heatmap plot.
    """
    if not model:
        return "No model available."

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    return plt.gcf()


def plot_fi(
    model: Any,
    importance_types: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    encoding: str,
) -> plt.Figure:
    """Generates a feature importance plot using FeatureImportanceEngine.

    Args:
        model (Any): Trained model for feature importance extraction.
        importance_types (List[str]): List of feature importance types to plot.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
        encoding (str): The encoding method used during preprocessing.

    Returns:
        plt.Figure: Feature importance plot.
    """
    if not model:
        return "No model available."

    fi_engine = FeatureImportanceEngine(
        [model], X_test, y_test, encoding, aggregate=True
    )
    fi_engine.evaluate_feature_importance(importance_types)
    return plt.gcf()


def plot_cluster(
    model: tuple[Any, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    encoding: str,
    n_clusters: int,
) -> Tuple[plt.Figure, plt.Figure]:
    """Performs clustering on Brier score and returns related plots.

    Args:
        model (Any): Trained model for cluster analysis.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
        encoding (str): The encoding method used during preprocessing.
        n_clusters (int): Number of clusters for Brier score analysis.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing the Brier score plot
            and the heatmap plot.
    """
    if not model:
        return "No model available."

    fi_engine = FeatureImportanceEngine(
        [model], X_test, y_test, encoding, aggregate=True
    )

    brier_plot, heatmap_plot, _ = fi_engine.analyze_brier_within_clusters(
        model=model, n_clusters=n_clusters
    )

    return brier_plot, heatmap_plot


def brier_score_wrapper(
    models: dict, selected_model: str, X_test: pd.DataFrame, y_test: pd.Series
) -> plt.Figure:
    """Wrapper function to generate Brier score plots.

    Args:
        models (dict): Dictionary of trained models where the keys are model names.
        selected_model (str): Name of the selected model from the dropdown.
        X_test (pd.DataFrame): Test dataset containing input features.
        y_test (pd.Series): Test dataset containing true labels.

    Returns:
        plt.Figure: Matplotlib figure showing the Brier score plot.
    """
    brier_score_groups(models[selected_model], X_test, y_test)
    return plt.gcf()


def plot_fi_wrapper(
    models: dict,
    selected_model: str,
    importance_types: List[str],
    X_test: Any,
    y_test: Any,
    encoding: str,
) -> Any:
    """Wrapper function to call plot_fi.

    Args:
        models (dict): Dictionary containing models.
        selected_model (str): The key to access the selected model in the dict.
        importance_types (List[str]): List of importance types.
        X_test (Any): Test features.
        y_test (Any): Test labels.
        encoding (str): The encoding method used.

    Returns:
        Any: The result from the plot_fi function.
    """
    return plot_fi(
        models[selected_model],
        importance_types,
        X_test,
        y_test,
        encoding,
    )


def plot_cluster_wrapper(
    models: dict,
    selected_model: str,
    X_test: Any,
    y_test: Any,
    encoding: str,
    n_clusters: int,
) -> Tuple[Any, Any]:
    """Wrapper function to call plot_cluster.

    Args:
        models (dict): Dictionary containing models.
        selected_model (str): The key to access the selected model in the dict.
        X_test (Any): Test features.
        y_test (Any): Test labels.
        encoding (str): The encoding method used.
        n_clusters (int): Number of clusters.

    Returns:
        Tuple[Any, Any]: The Brier score plot and heatmap plot.
    """
    return plot_cluster(
        models[selected_model],
        X_test,
        y_test,
        encoding,
        n_clusters=n_clusters,
    )


def run_single_inference(
    task: str,
    models: dict,
    selected_model: str,
    tooth: int,
    toothtype: int,
    rootnumber: int,
    mobility: int,
    restoration: int,
    percussion: int,
    sensitivity: int,
    furcation: int,
    side: int,
    pdbaseline: int,
    recbaseline: int,
    plaque: int,
    bop: int,
    age: int,
    gender: int,
    bmi: float,
    perio_history: int,
    diabetes: int,
    smokingtype: int,
    cigarettenumber: int,
    antibiotics: int,
    stresslvl: str,
    encoding: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[str, float]:
    """Run inference on a single input instance using model and encoding.

    Args:
        task (str): The task name for which the model was trained.
        models (dict): A dictionary containing the trained models.
        selected_model (str): The name of the selected model for inference.
        tooth (int): Tooth number provided as input for inference.
        toothtype (int): Type of the tooth provided for inference.
        rootnumber (int): Number of roots of the tooth.
        mobility (int): Mobility of the tooth.
        restoration (int): Restoration status of the tooth.
        percussion (int): Percussion sensitivity of the tooth.
        sensitivity (int): Tooth sensitivity level.
        furcation (int): Furcation baseline value.
        side (int): Side of the tooth for prediction.
        pdbaseline (int): Periodontal depth baseline.
        recbaseline (int): Recession baseline.
        plaque (int): Plaque level.
        bop (int): Bleeding on probing status.
        age (int): Age of the patient.
        gender (int): Gender of the patient.
        bmi (float): Body mass index of the patient.
        perio_history (int): Periodontal family history status.
        diabetes (int): Diabetes status.
        smokingtype (int): Smoking type classification.
        cigarettenumber (int): Number of cigarettes smoked per day.
        antibiotics (int): Antibiotic treatment status.
        stresslvl (str): Stress level of the patient.
        encoding (str): Encoding type ("one_hot" or "target").
        X_train (pd.DataFrame): Training features for target encoding.
        y_train (pd.Series): Training target for target encoding.

    Returns:
        Tuple[str, float]: Prediction and probability result for single input.
    """
    input_data = InferenceInput(
        tooth=tooth,
        toothtype=toothtype,
        rootnumber=rootnumber,
        mobility=mobility,
        restoration=restoration,
        percussion=percussion,
        sensitivity=sensitivity,
        furcation=furcation,
        side=side,
        pdbaseline=pdbaseline,
        recbaseline=recbaseline,
        plaque=plaque,
        bop=bop,
        age=age,
        gender=gender,
        bmi=bmi,
        perio_history=perio_history,
        diabetes=diabetes,
        smokingtype=smokingtype,
        cigarettenumber=cigarettenumber,
        antibiotics=antibiotics,
        stresslvl=stresslvl,
    )

    input_data_dict = input_data.to_dict()
    raw_data = pd.DataFrame([input_data_dict])
    engine = StaticProcessEngine(behavior=False, scale=False)
    raw_data = engine.create_tooth_features(raw_data, neighbors=False, patient_id=False)

    if encoding == "target":
        task = InputProcessor.process_tasks([task])[0]
        dataloader = ProcessedDataLoader(task, encoding)
        raw_data = dataloader.encode_categorical_columns(raw_data)
        if y_train is not None and not y_train.empty:
            classification = "binary" if y_train.nunique() == 2 else "multiclass"
        else:
            raise ValueError(
                "y_train is None or empty, cannot determine classification."
            )
        resampler = Resampler(classification, encoding)
        _, raw_data = resampler.apply_target_encoding(X_train, raw_data, y_train)

        encoded_fields = [
            "restoration",
            "periofamilyhistory",
            "diabetes",
            "toothtype",
            "furcationbaseline",
            "smokingtype",
            "stresslvl",
        ]
        for key, value in input_data_dict.items():
            if key not in encoded_fields:
                raw_data[key] = value
    else:
        input_data_dict = input_data.to_dict()
        for key, value in input_data_dict.items():
            raw_data[key] = value

    model = models[selected_model]
    predict_data = create_predict_data(raw_data, input_data, encoding, model)
    predict_data = engine.scale_numeric_columns(predict_data)
    inference_engine = ModelInference(model)

    return inference_engine.predict(predict_data)
