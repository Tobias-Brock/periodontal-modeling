"""Gradio frontend."""

import gradio as gr

from pamod.app import (
    load_and_initialize_plotter,
    load_data,
    plot_cm,
    plot_fi,
    plot_histogram_2d,
    plot_matrix,
    plot_outcome_descriptive,
    plot_pocket_comparison,
    plot_pocket_group_comparison,
    run_benchmarks,
)

with gr.Blocks() as app:
    gr.Markdown("## ML Periodontal Modeling")

    models_state = gr.State()

    with gr.Tabs():
        with gr.Tab("Descriptives"):
            load_button = gr.Button("Load Data", scale=1)
            load_output = gr.Textbox(label="Status", interactive=False, scale=2)

            with gr.Row():
                column_before_input = gr.Textbox(
                    label="Column Before", value="pdbaseline", scale=1
                )
                column_after_input = gr.Textbox(
                    label="Column After", value="pdrevaluation", scale=1
                )
                plot_hist_2d_button = gr.Button("Plot 2D Histogram", scale=1)
            hist_2d_output = gr.Plot(scale=6)

            with gr.Row():
                column1_input = gr.Textbox(
                    label="Column 1", value="pdbaseline", scale=1
                )
                column2_input = gr.Textbox(
                    label="Column 2", value="pdrevaluation", scale=1
                )
                plot_comparison_button = gr.Button("Plot Pocket Comparison", scale=1)
            pocket_comparison_output = gr.Plot(scale=6)

            with gr.Row():
                group_column_before_input = gr.Textbox(
                    label="Group Before", value="pdgroupbase", scale=1
                )
                group_column_after_input = gr.Textbox(
                    label="Group After", value="pdgrouprevaluation", scale=1
                )
                plot_group_comparison_button = gr.Button(
                    "Plot Pocket Group Comparison", scale=1
                )
            group_comparison_output = gr.Plot(scale=6)

            with gr.Row():
                vertical_input = gr.Textbox(
                    label="Vertical Column", value="pdgrouprevaluation", scale=1
                )
                horizontal_input = gr.Textbox(
                    label="Horizontal Column", value="pdgroupbase", scale=1
                )
                plot_matrix_button = gr.Button("Plot Matrix", scale=1)
            matrix_output = gr.Plot(scale=6)

            with gr.Row():
                outcome_input = gr.Textbox(
                    label="Outcome Column", value="pdgrouprevaluation", scale=1
                )
                title_input = gr.Textbox(
                    label="Plot Title", value="Distribution of Classes", scale=1
                )
                plot_outcome_button = gr.Button("Plot Outcome Descriptive", scale=1)
            outcome_output = gr.Plot(scale=6)

            load_button.click(
                fn=load_and_initialize_plotter, inputs=[], outputs=[load_output]
            )
            plot_hist_2d_button.click(
                fn=plot_histogram_2d,
                inputs=[column_before_input, column_after_input],
                outputs=hist_2d_output,
            )
            plot_comparison_button.click(
                fn=plot_pocket_comparison,
                inputs=[column1_input, column2_input],
                outputs=pocket_comparison_output,
            )
            plot_group_comparison_button.click(
                fn=plot_pocket_group_comparison,
                inputs=[group_column_before_input, group_column_after_input],
                outputs=group_comparison_output,
            )
            plot_matrix_button.click(
                fn=plot_matrix,
                inputs=[vertical_input, horizontal_input],
                outputs=matrix_output,
            )
            plot_outcome_button.click(
                fn=plot_outcome_descriptive,
                inputs=[outcome_input, title_input],
                outputs=outcome_output,
            )

        with gr.Tab("Benchmarking"):
            task_input = gr.CheckboxGroup(
                label="Tasks",
                choices=["Pocket closure", "Pocket improvement", "Pocket groups"],
                value=["Pocket closure"],
            )

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

            tuning_methods_input = gr.CheckboxGroup(
                label="Tuning Methods",
                choices=["Holdout", "Cross-Validation"],
                value=["Holdout"],  # Default value can be set
            )

            hpo_methods_input = gr.CheckboxGroup(
                label="HPO Methods",
                choices=["HEBO", "Random Search"],
                value=["HEBO"],  # Default value can be set
            )

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

            sampling_input = gr.Dropdown(
                label="Sampling Strategy",
                choices=["None", "upsampling", "downsampling", "smote"],
                value="None",  # Default value is None
            )

            factor_input = gr.Textbox(label="Sampling Factor", value=None)
            n_configs_input = gr.Number(label="Number of Configurations", value=5)
            racing_folds_input = gr.Number(label="Racing Folds", value=5)
            n_jobs_input = gr.Number(label="Number of Jobs", value=-1)

            run_button = gr.Button("Run Benchmark")

            results_output = gr.Dataframe(label="Benchmark Results")
            metrics_plot_output = gr.Plot(label="Metrics Comparison")
            models_state = gr.State()

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
                outputs=[results_output, metrics_plot_output, models_state],
            )

        with gr.Tab("Evaluation"):
            gr.Markdown("### Evaluation")

            task_input = gr.Textbox(label="Task", value="Pocket closure")
            encoding_input = gr.Dropdown(
                label="Encoding",
                choices=["one_hot", "target"],
                value="one_hot",
            )

            load_data_button = gr.Button("Load Data")
            load_status_output = gr.Textbox(label="Status", interactive=False)

            X_train_state = gr.State()
            y_train_state = gr.State()
            X_test_state = gr.State()
            y_test_state = gr.State()

            generate_confusion_matrix_button = gr.Button("Generate Confusion Matrix")
            matrix_plot = gr.Plot()

            importance_type_input = gr.CheckboxGroup(
                label="Importance Types",
                choices=["shap", "permutation", "standard"],
                value=["shap"],
            )

            generate_feature_importance_button = gr.Button(
                "Generate Feature Importance"
            )

            fi_plot = gr.Plot()

            load_data_button.click(
                fn=load_data,
                inputs=[task_input, encoding_input],
                outputs=[
                    load_status_output,
                    X_train_state,
                    y_train_state,
                    X_test_state,
                    y_test_state,
                ],
            )

            generate_confusion_matrix_button.click(
                fn=lambda models, X_test, y_test: plot_cm(models, X_test, y_test),
                inputs=[models_state, X_test_state, y_test_state],
                outputs=matrix_plot,
            )

            generate_feature_importance_button.click(
                fn=lambda models, importance_types, X_test, y_test, encoding: plot_fi(
                    models, importance_types, X_test, y_test, encoding
                ),
                inputs=[
                    models_state,
                    importance_type_input,
                    X_test_state,
                    y_test_state,
                    encoding_input,
                ],
                outputs=fi_plot,
            )

        with gr.Tab("Inference"):
            gr.Markdown("### Inference")

app.launch()
