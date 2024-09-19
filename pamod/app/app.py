import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pamod.benchmarking import MultiBenchmarker


def run_benchmarks(
    tasks,
    learners,
    tuning_methods,
    hpo_methods,
    criteria,
    sampling,
    factor,
    n_configs,
    racing_folds,
    n_jobs,
):
    # Adjust None inputs
    sampling = None if sampling == "None" else sampling
    factor = None if factor == "None" else float(factor)

    # Instantiate and run the MultiBenchmarker
    benchmarker = MultiBenchmarker(
        tasks=tasks.split(","),
        learners=learners.split(","),
        tuning_methods=tuning_methods.split(","),
        hpo_methods=hpo_methods.split(","),
        criteria=criteria.split(","),
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

    df_results = df_results.round(4)  # Round values for better readability

    # Plot key metrics
    plt.figure(figsize=(10, 6))
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
        cm = df_results["Confusion Matrix"].iloc[
            0
        ]  # Use the first result as an example
        plt.figure(figsize=(6, 4))
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
