"""App."""

from typing import Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pamod.benchmarking import MultiBenchmarker


def run_benchmarks(
    tasks: str,
    learners: str,
    tuning_methods: str,
    hpo_methods: str,
    criteria: str,
    sampling: Optional[str],
    factor: Optional[float],
    n_configs: int,
    racing_folds: int,
    n_jobs: int,
) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure], Optional[plt.Figure]]:
    """Run benchmark evaluations with frontend.

    Args:
        tasks (str): Comma-separated string of tasks (e.g., 'pocketclosure, improve').
        learners (str): Comma-separated string of learners (e.g., 'XGB, RandomForest').
        tuning_methods (str): Comma-seperated string of tuning methods.
        hpo_methods (str): Comma-separated string of HPO methods (e.g., 'HEBO, RS').
        criteria (str): Comma-separated string of evaluation criteria
            (e.g., 'f1, brier_score').
        sampling (str): Sampling strategy to use ('smote', 'upsampling', 'None', etc.).
        factor (float): Resampling factor as a string or 'None'.
        n_configs (int): Number of configurations for hyperparameter tuning.
        racing_folds (int): Number of racing folds for Random Search.
        n_jobs (int): Number of parallel jobs to run.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[plt.Figure], Optional[plt.Figure]]:
            - df_results: DataFrame containing benchmark results.
            - metrics_plot: Matplotlib figure with key metrics comparison plot.
            - conf_matrix_plot: Matplotlib figure with confusion matrix plot, or None.
    """
    sampling = None if sampling == "None" else sampling
    factor = None if factor == "None" else float(factor) if factor else None

    tasks_list = [task.strip() for task in tasks.split(",")]
    learners_list = [learner.strip() for learner in learners.split(",")]
    tuning_methods_list = [tuning.strip() for tuning in tuning_methods.split(",")]
    hpo_methods_list = [hpo.strip() for hpo in hpo_methods.split(",")]
    criteria_list = [criterion.strip() for criterion in criteria.split(",")]

    # Instantiate and run the MultiBenchmarker
    benchmarker = MultiBenchmarker(
        tasks=tasks_list,
        learners=learners_list,
        tuning_methods=tuning_methods_list,
        hpo_methods=hpo_methods_list,
        criteria=criteria_list,
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
    plt.figure(figsize=(10, 6), dpi=200)
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
        plt.figure(figsize=(6, 4), dpi=200)
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        conf_matrix_plot = plt.gcf()

    return df_results, metrics_plot, conf_matrix_plot


# Gradio UI components
with gr.Blocks() as app:
    gr.Markdown("## ML Pariodontal Modeling")

    # Inputs for tasks, learners, tuning methods, HPO methods, and criteria
    task_input = gr.Textbox(
        label="Tasks (comma-separated)",
        placeholder="e.g., pocketclosure, improve, pdgrouprevaluation",
    )
    learners_input = gr.Textbox(
        label="Learners (comma-separated)",
        placeholder="e.g., XGB, RandomForest, LogisticRegression, MLP",
    )
    tuning_methods_input = gr.Textbox(
        label="Tuning Methods (comma-separated)", placeholder="e.g., holdout, cv"
    )
    hpo_methods_input = gr.Textbox(
        label="HPO Methods (comma-separated)", placeholder="e.g., HEBO, RS"
    )
    criteria_input = gr.Textbox(
        label="Criteria (comma-separated)",
        placeholder="e.g., f1, brier_score, macro_f1",
    )

    # Additional parameters
    sampling_input = gr.Textbox(label="Sampling Strategy", placeholder="e.g., None")
    factor_input = gr.Textbox(label="Resampling Factor", placeholder="e.g., None")
    n_configs_input = gr.Number(label="Number of Configurations", value=10)
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
