from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from pamod.base import Patient, Side, Tooth, create_predict_data
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

    classification = "multiclass" if "pdgrouprevaluation" in tasks else "binary"

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

    plt.figure(figsize=(8, 6))
    df_results.plot(
        x="Learner",
        y=metrics_to_plot,
        kind="bar",
        ax=plt.gca(),
    )
    plt.title("Benchmark Metrics for Each Learner")
    plt.xlabel("Learner")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metrics")
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
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

    return "Data loaded successfully", train_df, X_train, y_train, X_test, y_test


def load_data_wrapper(
    task: str, encoding: str
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wraooer to load data.

    Args:
        task (str)): Task input from UI.
        encoding (str): Encoding type.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, y_train, X_test, y_test.
    """
    return load_data(InputProcessor.process_tasks([task])[0], encoding)


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


def update_teeth_ui(
    jaw_side: str, all_teeth: list, teeth_numbers: dict, tooth_columns: dict
) -> List:
    """Updates the visibility of teeth UI components based on the selected jaw side.

    Args:
        jaw_side (str): The selected jaw side (e.g., 'Upper Right', 'Lower Left').
        all_teeth (list): List of all teeth numbers.
        teeth_numbers (dict): A dictionary mapping jaw sides to lists of tooth
            numbers.
        tooth_columns (dict): A dictionary mapping tooth numbers to the respective
            UI components for each tooth.

    Returns:
        List: A list of Gradio update objects to change the visibility of teeth UI
            components.
    """
    updates = []
    for tooth in all_teeth:
        visible = tooth in teeth_numbers[jaw_side]
        for _ in tooth_columns[tooth]:  # Access tooth_columns from the argument now
            updates.append(gr.update(visible=visible))

    return updates


def teeth_ui_wrapper(
    jaw_side: str, all_teeth: list, teeth_numbers: dict, tooth_columns: dict
) -> List:
    """Wrapper to pass additional arguments to the update_teeth_ui function.

    Args:
        jaw_side (str): The selected jaw side (e.g., 'Upper Right', 'Lower Left').
        all_teeth (list): List of all teeth numbers.
        teeth_numbers (dict): A dictionary mapping jaw sides to lists of tooth
            numbers.
        tooth_columns (dict): A dictionary mapping tooth numbers to the respective
            UI components for each tooth.

    Returns:
        List: A list of Gradio update objects to change the visibility of teeth UI
            components.
    """
    return update_teeth_ui(jaw_side, all_teeth, teeth_numbers, tooth_columns)


def create_handle_side_change_fn(
    tooth: int,
) -> Callable[
    [Dict[str, Any], str, Any, Any, Any, Any, Any],
    Tuple[Dict[str, Any], Any, Any, Any, Any, Any],
]:
    """Creates a function to handle side selection change for a specific tooth.

    Args:
        tooth (int): The tooth number.

    Returns:
        Callable: A function that handles side changes for the given tooth.
    """

    def handle_side_change(
        tooth_states_value: Dict[str, Any],
        selected_side: str,
        furcation_input_value: Any,
        pdbaseline_input_value: Any,
        recbaseline_input_value: Any,
        plaque_input_value: Any,
        bop_input_value: Any,
    ) -> Tuple[Dict[str, Any], Any, Any, Any, Any, Any]:
        """Handles the change in selected side and updates the tooth state.

        Args:
            tooth_states_value (Dict[str, Any]): The current state of all teeth.
            selected_side (str): The newly selected side.
            furcation_input_value (Any): Value of furcation input.
            pdbaseline_input_value (Any): Value of PD baseline input.
            recbaseline_input_value (Any): Value of REC baseline input.
            plaque_input_value (Any): Value of plaque input.
            bop_input_value (Any): Value of BOP input.

        Returns:
            Tuple[Dict[str, Any], Any, Any, Any, Any, Any]:
                Updated tooth states and input values for the new side.
        """
        tooth_str = str(tooth)

        tooth_state = tooth_states_value.get(
            tooth_str, {"current_side": "Side 1", "sides": {}}
        )
        previous_side = tooth_state.get("current_side", "Side 1")
        sides_data = tooth_state.get("sides", {})

        sides_data[previous_side] = {
            "furcation_input": furcation_input_value,
            "pdbaseline_input": pdbaseline_input_value,
            "recbaseline_input": recbaseline_input_value,
            "plaque_input": plaque_input_value,
            "bop_input": bop_input_value,
        }

        tooth_state["current_side"] = selected_side
        tooth_state["sides"] = sides_data  # Update sides data

        tooth_states_value[tooth_str] = tooth_state

        if selected_side in sides_data:
            data = sides_data[selected_side]
            furcation_output = data.get("furcation_input", None)
            pdbaseline_output = data.get("pdbaseline_input", "")
            recbaseline_output = data.get("recbaseline_input", "")
            plaque_output = data.get("plaque_input", None)
            bop_output = data.get("bop_input", None)
        else:
            furcation_output = None
            pdbaseline_output = ""
            recbaseline_output = ""
            plaque_output = None
            bop_output = None

        return (
            tooth_states_value,
            furcation_output,
            pdbaseline_output,
            recbaseline_output,
            plaque_output,
            bop_output,
        )

    return handle_side_change


def collect_data(
    age: Union[int, float, str],
    gender: Union[int, str],
    bmi: Union[float, str],
    perio_history: Union[int, str],
    diabetes: Union[int, str],
    smokingtype: Union[int, str],
    cigarettenumber: Union[int, str],
    antibiotics: Union[int, str],
    stresslvl: str,
    *tooth_inputs_and_state,
    teeth_components: Dict[int, Dict[str, Any]],
    all_teeth: List[int],
) -> Tuple[str, pd.DataFrame]:
    """Collects data from inputs and constructs a Patient object.

    Args:
        age (Union[int, float, str]): Age of the patient.
        gender (Union[int, str]): Gender of the patient.
        bmi (Union[float, str]): Body Mass Index of the patient.
        perio_history (Union[int, str]): Periodontal history.
        diabetes (Union[int, str]): Diabetes status.
        smokingtype (Union[int, str]): Type of smoking.
        cigarettenumber (Union[int, str]): Number of cigarettes smoked.
        antibiotics (Union[int, str]): Antibiotic treatment status.
        stresslvl (str): Stress level.
        *tooth_inputs_and_state: Flattened list of tooth input values.
        teeth_components (Dict[int, Dict[str, Any]]): Mapping of tooth numbers.
        all_teeth (List[int]): List of all tooth numbers.

    Returns:
        Tuple[str, pd.DataFrame]: Success message and DataFrame of patient data.
    """
    *tooth_inputs, tooth_states = tooth_inputs_and_state
    patient = Patient(
        age=int(age),
        gender=int(gender),
        bmi=float(bmi),
        perio_history=int(perio_history),
        diabetes=int(diabetes),
        smokingtype=int(smokingtype),
        cigarettenumber=int(cigarettenumber),
        antibiotics=int(antibiotics),
        stresslvl=stresslvl,
        teeth=[],
    )

    num_features_per_tooth = len(teeth_components[all_teeth[0]])
    num_teeth = len(all_teeth)
    total_inputs = num_features_per_tooth * num_teeth

    if len(tooth_inputs) != total_inputs:
        raise ValueError("Number of tooth inputs does not match expected total inputs.")

    for idx, tooth in enumerate(all_teeth):
        tooth_str = str(tooth)
        start_idx = idx * num_features_per_tooth
        end_idx = start_idx + num_features_per_tooth
        tooth_data = tooth_inputs[start_idx:end_idx]
        tooth_dict = {}
        for feature_key, input_value in zip(
            teeth_components[tooth].keys(), tooth_data, strict=False
        ):
            tooth_dict[feature_key] = input_value

        # Update tooth_states with current per-side inputs
        tooth_state = tooth_states.get(
            tooth_str, {"current_side": "Side 1", "sides": {}}
        )
        current_side = tooth_state.get("current_side", "Side 1")
        sides_data = tooth_state.get("sides", {})

        # Get current per-side inputs
        furcation_input = tooth_dict.get("furcation_input")
        pdbaseline_input = tooth_dict.get("pdbaseline_input")
        recbaseline_input = tooth_dict.get("recbaseline_input")
        plaque_input = tooth_dict.get("plaque_input")
        bop_input = tooth_dict.get("bop_input")

        # Update sides_data with current per-side inputs for current side
        sides_data[current_side] = {
            "furcation_input": furcation_input,
            "pdbaseline_input": pdbaseline_input,
            "recbaseline_input": recbaseline_input,
            "plaque_input": plaque_input,
            "bop_input": bop_input,
        }

        tooth_state["sides"] = sides_data
        tooth_states[tooth_str] = tooth_state

        # Now proceed to parse tooth data and sides as before

        tooth_parsed = {}
        for key, value in tooth_dict.items():
            if key == "side_dropdown":
                tooth_parsed[key] = value  # For dropdowns
            else:
                input_component = teeth_components[tooth][key]
                if isinstance(input_component, gr.Dropdown):
                    tooth_parsed[key] = int(value) if value is not None else None
                else:  # Textbox
                    if isinstance(value, str):
                        value = value.strip()
                        if value == "":
                            tooth_parsed[key] = None
                        else:
                            try:
                                tooth_parsed[key] = int(value)
                            except ValueError:
                                tooth_parsed[key] = None  # Handle invalid input
                    else:
                        tooth_parsed[key] = int(value) if value is not None else None

        tooth_has_data = any(
            v is not None for k, v in tooth_parsed.items() if k != "side_dropdown"
        )

        sides = []
        sides_data = tooth_state.get("sides", {})
        for side_name, side_values in sides_data.items():
            side_parsed = {}
            for k, v in side_values.items():
                input_component = teeth_components[tooth][k]
                if isinstance(input_component, gr.Dropdown):
                    side_parsed[k] = int(v) if v is not None else None
                else:  # Textbox
                    if isinstance(v, str):
                        v = v.strip()
                        if v == "":
                            side_parsed[k] = None
                        else:
                            try:
                                side_parsed[k] = int(v)
                            except ValueError:
                                side_parsed[k] = None  # Handle invalid input
                    else:
                        side_parsed[k] = int(v) if v is not None else None

            side_has_data = any(value is not None for value in side_parsed.values())
            if side_has_data:
                sides.append(
                    Side(
                        furcationbaseline=side_parsed.get("furcation_input"),
                        side=int(side_name.split()[-1]),
                        pdbaseline=side_parsed.get("pdbaseline_input"),
                        recbaseline=side_parsed.get("recbaseline_input"),
                        plaque=side_parsed.get("plaque_input"),
                        bop=side_parsed.get("bop_input"),
                    )
                )

        if tooth_has_data or sides:
            tooth_number = tooth
            if tooth in [11, 12, 21, 22, 31, 32, 41, 42, 13, 23, 33, 43]:
                toothtype = 0
                rootnumber = 0
            elif tooth in [14, 15, 24, 25, 34, 35, 44, 45]:
                toothtype = 1
                rootnumber = 1
            else:
                toothtype = 2
                rootnumber = 1

            tooth_obj = Tooth(
                tooth=tooth_number,
                toothtype=toothtype,
                rootnumber=rootnumber,
                mobility=tooth_parsed.get("mobility"),
                restoration=tooth_parsed.get("restoration"),
                percussion=tooth_parsed.get("percussion"),
                sensitivity=tooth_parsed.get("sensitivity"),
                sides=sides,
            )
            patient.teeth.append(tooth_obj)

    patient_df = patient_to_dataframe(patient)
    print("Collected Patient Data:")
    print(patient_df)

    return "Patient data collected successfully!", patient_df


def patient_to_dataframe(patient: Patient) -> pd.DataFrame:
    """Converts a Patient instance into a DataFrame suitable for prediction.

    Args:
        patient (Patient): The Patient dataclass instance.

    Returns:
        pd.DataFrame: DataFrame where each row represents a tooth side.
    """
    rows = []
    for tooth in patient.teeth:
        for side in tooth.sides:
            data = {
                # Patient-level data
                "age": patient.age,
                "gender": patient.gender,
                "bodymassindex": patient.bmi,
                "periofamilyhistory": patient.perio_history,
                "diabetes": patient.diabetes,
                "smokingtype": patient.smokingtype,
                "cigarettenumber": patient.cigarettenumber,
                "antibiotictreatment": patient.antibiotics,
                "stresslvl": patient.stresslvl,
                # Tooth-level data
                "tooth": tooth.tooth,
                "toothtype": tooth.toothtype,
                "rootnumber": tooth.rootnumber,
                "mobility": tooth.mobility,
                "restoration": tooth.restoration,
                "percussion-sensitivity": tooth.percussion,
                "sensitivity": tooth.sensitivity,
                # Side-level data
                "furcationbaseline": side.furcationbaseline,
                "side": side.side,
                "pdbaseline": side.pdbaseline,
                "recbaseline": side.recbaseline,
                "plaque": side.plaque,
                "bop": side.bop,
            }
            rows.append(data)
    return pd.DataFrame(rows)


def run_patient_inference(
    task: str,
    models: dict,
    selected_model: str,
    patient_data: pd.DataFrame,
    encoding: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
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
        pd.DataFrame: DataFrame containing tooth, side, prediction, and probability.
    """
    if patient_data.empty:
        raise ValueError(
            "Patient data empty. Please submit data before running inference."
        )
    print("Patient Data Received for Inference:")
    print(patient_data)

    task_processed = InputProcessor.process_tasks([task])[0]
    classification = (
        "multiclass" if task_processed == "pdgrouprevaluation" else "binary"
    )
    engine = StaticProcessEngine(behavior=False, scale=False)
    patient_data["id_patient"] = "inference_patient"
    raw_data = engine.create_tooth_features(
        patient_data, neighbors=True, patient_id=False
    )

    if encoding == "target":
        dataloader = ProcessedDataLoader(task_processed, encoding)
        raw_data = dataloader.encode_categorical_columns(raw_data)
        resampler = Resampler(classification, encoding)
        _, raw_data = resampler.apply_target_encoding(X_train, raw_data, y_train)
        print(raw_data)

        encoded_fields = [
            "restoration",
            "periofamilyhistory",
            "diabetes",
            "toothtype",
            "toothside",
            "furcationbaseline",
            "smokingtype",
            "stresslvl",
        ]

        for key in raw_data.columns:
            if key not in encoded_fields and key in patient_data.columns:
                raw_data[key] = patient_data[key].values

    else:
        raw_data = create_predict_data(
            raw_data, patient_data, encoding, models[selected_model]
        )

    model = models[selected_model]
    predict_data = create_predict_data(raw_data, patient_data, encoding, model)
    predict_data = engine.scale_numeric_columns(predict_data)
    inference_engine = ModelInference(classification, model)

    results = inference_engine.predict(predict_data)
    output_data = patient_data[["tooth", "side"]].copy()
    output_data["prediction"] = results["prediction"]
    output_data["probability"] = results.drop(columns=["prediction"]).max(axis=1)

    return predict_data, output_data, results


def run_jackknife_inference(
    task: str,
    models: dict,
    selected_model: str,
    train_df: pd.DataFrame,
    patient_data: pd.DataFrame,
    encoding: str,
    inference_results: pd.DataFrame,
    sample_fraction: float = 1.0,
    n_jobs: int = -1,
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
        sample_fraction (float, optional): Fraction of patient IDs to use in jackknife.
            Defaults to 1.0.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

    Returns:
        Tuple[pd.DataFrame, plt.Figure]: Jackknife results and the plot.
    """
    task_processed = InputProcessor.process_tasks([task])[0]
    classification = (
        "multiclass" if task_processed == "pdgrouprevaluation" else "binary"
    )
    model = models[selected_model]
    model_params = model.get_params()

    inference_engine = ModelInference(classification=classification, model=model)

    jackknife_results = inference_engine.jackknife_resampling(
        train_df=train_df,
        patient_data=patient_data,
        encoding=encoding,
        model_params=model_params,
        sample_fraction=sample_fraction,
        n_jobs=n_jobs,
    )

    ci_dict = inference_engine.jackknife_confidence_intervals(jackknife_results)

    max_plots = 5
    data_indices = patient_data.index[:max_plots]
    original_predictions = {}
    for data_index in data_indices:
        original_probs = inference_results.loc[data_index].drop("prediction").to_dict()
        original_predictions[data_index] = original_probs

    ci_plot = inference_engine.plot_jackknife_intervals(
        ci_dict, data_indices, original_preds=original_predictions
    )

    return jackknife_results, ci_plot
