"""GRadio frontend."""

from typing import Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pamod.benchmarking import InputProcessor, MultiBenchmarker


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

    # Handle None inputs for sampling and factor
    sampling = None if sampling == "None" else sampling
    factor = None if factor == "None" else float(factor) if factor else None

    # Instantiate and run the MultiBenchmarker
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

    # Capture the results
    df_results = benchmarker.run_all_benchmarks()

    if df_results.empty:
        return "No results to display", None, None

    # Round values for better readability
    df_results = df_results.round(4)

    # Plot key metrics
    plt.figure(figsize=(6, 4), dpi=300)
    df_results.plot(
        x="Learner",
        y=["F1 Score", "Accuracy", "ROC AUC Score"],
        kind="bar",
        ax=plt.gca(),
    )
    plt.title("Benchmark Metrics for Each Learner")
    plt.xlabel("Learner")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to be displayed in Gradio
    metrics_plot = plt.gcf()

    # Confusion Matrix Plot
    conf_matrix_plot = None
    if "Confusion Matrix" in df_results.columns:
        cm = df_results["Confusion Matrix"].iloc[0]  # Use the first result as example
        plt.figure(figsize=(6, 4), dpi=300)
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        conf_matrix_plot = plt.gcf()

    return df_results, metrics_plot, conf_matrix_plot


# Gradio UI components
with gr.Blocks() as app:
    gr.Markdown("## ML Periodontal Modeling")

    # Task selection
    task_input = gr.CheckboxGroup(
        label="Tasks",
        choices=["Pocket closure", "Pocket improvement", "Pocket groups"],
        value=["Pocket closure"],  # Default value can be set
    )

    # Learner selection
    learners_input = gr.CheckboxGroup(
        label="Learners",
        choices=[
            "XGBoost",
            "Random Forest",
            "Logistic Regression",
            "Multilayer Perceptron",
        ],
        value=["XGBoost"],  # Default value can be set
    )

    # Tuning method selection
    tuning_methods_input = gr.CheckboxGroup(
        label="Tuning Methods",
        choices=["Holdout", "Cross-Validation"],
        value=["Holdout"],  # Default value can be set
    )

    # HPO method selection
    hpo_methods_input = gr.CheckboxGroup(
        label="HPO Methods",
        choices=["HEBO", "Random Search"],
        value=["HEBO"],  # Default value can be set
    )

    # Criteria selection
    criteria_input = gr.CheckboxGroup(
        label="Criteria",
        choices=["F1 Score", "Brier Score", "Macro F1 Score"],
        value=["F1 Score"],  # Default value can be set
    )

    encodings_input = gr.CheckboxGroup(
        label="Encoding",
        choices=["One-hot", "Target"],
        value=["One-hot"],  # Default value can be set
    )

    # Sampling strategy
    sampling_input = gr.Dropdown(
        label="Sampling Strategy",
        choices=["None", "upsampling", "downsampling", "smote"],
        value="None",  # Default value is None
    )

    # Other inputs
    factor_input = gr.Textbox(label="Sampling Factor", value=None)
    n_configs_input = gr.Number(label="Number of Configurations", value=5)
    racing_folds_input = gr.Number(label="Racing Folds", value=5)
    n_jobs_input = gr.Number(label="Number of Jobs", value=-1)

    # Button to run the benchmark
    run_button = gr.Button("Run Benchmark")

    # Display area for the benchmark results
    results_output = gr.Dataframe(label="Benchmark Results")
    metrics_plot_output = gr.Plot(label="Metrics Comparison")
    conf_matrix_output = gr.Plot(label="Confusion Matrix")

    # Set up action to run benchmarks and plot results
    run_button.click(
        fn=run_benchmarks,
        inputs=[
            task_input,
            learners_input,
            tuning_methods_input,
            hpo_methods_input,
            criteria_input,
            encodings_input,
            sampling_input,
            factor_input,
            n_configs_input,
            racing_folds_input,
            n_jobs_input,
        ],
        outputs=[results_output, metrics_plot_output, conf_matrix_output],
    )

# Launch the Gradio app
app.launch()
