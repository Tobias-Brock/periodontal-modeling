"""Gradio frontend."""

import gradio as gr

from pamod.app import (
    benchmarks_wrapper,
    brier_score_wrapper,
    load_and_initialize_plotter,
    load_data,
    plot_cluster_wrapper,
    plot_cm,
    plot_fi_wrapper,
    plot_histogram_2d,
    plot_matrix,
    plot_outcome_descriptive,
    plot_pocket_comparison,
    plot_pocket_group_comparison,
    run_single_inference,
    update_model_dropdown,
)
from pamod.benchmarking import InputProcessor
from pamod.config import PROCESSED_BASE_DIR, RAW_DATA_DIR

with gr.Blocks() as app:
    gr.Markdown("## ML Periodontal Modeling")

    models_state = gr.State()
    task_state = gr.State()
    X_train_state = gr.State()
    y_train_state = gr.State()
    X_test_state = gr.State()
    y_test_state = gr.State()

    with gr.Tabs():
        with gr.Tab("Descriptives"):
            with gr.Row():
                path_input = gr.Textbox(
                    label="File Path",
                    value=str(RAW_DATA_DIR) + "/" + "Periodontitis_ML_Dataset.xlsx",
                    scale=1,
                )
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
                fn=load_and_initialize_plotter,
                inputs=[path_input],
                outputs=[load_output],
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
            path_input = gr.Textbox(
                label="File Path",
                value=str(PROCESSED_BASE_DIR) + "/" + "processed_data.csv",
            )

            task_input = gr.Dropdown(
                label="Tasks",
                choices=["Pocket closure", "Pocket improvement", "Pocket groups"],
                value="Pocket closure",
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

            factor_input = gr.Textbox(label="Sampling Factor", value="")
            n_configs_input = gr.Number(label="Number of Configurations", value=3)
            racing_folds_input = gr.Number(label="Racing Folds", value=5)
            n_jobs_input = gr.Number(label="Number of Jobs", value=-1)

            run_button = gr.Button("Run Benchmark")

            results_output = gr.Dataframe(label="Benchmark Results")
            metrics_plot_output = gr.Plot(label="Metrics Comparison")

            task_input.change(fn=lambda x: x, inputs=task_input, outputs=task_state)

            run_button.click(
                fn=benchmarks_wrapper,
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
                    path_input,
                ],
                outputs=[results_output, metrics_plot_output, models_state],
            )

        with gr.Tab("Evaluation"):
            task_display = gr.Textbox(
                label="Selected Task", value="", interactive=False
            )
            model_dropdown = gr.Dropdown(
                label="Select Model", choices=[], value=None, multiselect=False
            )  # Ensure single selection
            encoding_input = gr.Dropdown(
                label="Encoding",
                choices=["one_hot", "target"],
                value="one_hot",
            )

            load_data_button = gr.Button("Load Data")
            load_status_output = gr.Textbox(label="Status", interactive=False)
            task_display.value = task_input.value  # Keep this line

            models_state.change(
                fn=update_model_dropdown,
                inputs=models_state,
                outputs=model_dropdown,
            )

            task_input.change(
                fn=lambda task: task, inputs=task_input, outputs=task_display
            )

            generate_confusion_matrix_button = gr.Button("Generate Confusion Matrix")
            matrix_plot = gr.Plot()

            generate_brier_scores_button = gr.Button("Generate Brier Scores")
            brier_score_plot = gr.Plot()

            importance_type_input = gr.CheckboxGroup(
                label="Importance Types",
                choices=["shap", "permutation", "standard"],
                value=["shap"],
            )

            generate_feature_importance_button = gr.Button(
                "Generate Feature Importance"
            )
            fi_plot = gr.Plot()

            cluster_button = gr.Button("Perform Brier Score Clustering")
            n_clusters_input = gr.Slider(
                label="Number of Clusters", minimum=2, maximum=10, step=1, value=3
            )
            cluster_brier_plot = gr.Plot()
            cluster_heatmap_plot = gr.Plot()

            load_data_button.click(
                fn=lambda task, encoding: load_data(
                    InputProcessor.process_tasks([task])[
                        0
                    ],  # Only pass task and encoding
                    encoding,
                ),
                inputs=[task_input, encoding_input],
                outputs=[
                    load_status_output,
                    X_train_state,
                    y_train_state,
                    X_test_state,
                    y_test_state,
                ],
            )

            X_train_state.change(
                fn=lambda x: x, inputs=X_train_state, outputs=X_train_state
            )

            y_train_state.change(
                fn=lambda y: y, inputs=y_train_state, outputs=y_train_state
            )

            generate_confusion_matrix_button.click(
                fn=lambda models, selected_model, X_test, y_test: plot_cm(
                    models[selected_model], X_test, y_test
                ),
                inputs=[models_state, model_dropdown, X_test_state, y_test_state],
                outputs=matrix_plot,
            )

            generate_brier_scores_button.click(
                fn=brier_score_wrapper,
                inputs=[models_state, model_dropdown, X_test_state, y_test_state],
                outputs=brier_score_plot,
            )

            generate_feature_importance_button.click(
                fn=plot_fi_wrapper,
                inputs=[
                    models_state,
                    model_dropdown,
                    importance_type_input,
                    X_test_state,
                    y_test_state,
                    encoding_input,
                ],
                outputs=fi_plot,
            )

            cluster_button.click(
                fn=plot_cluster_wrapper,
                inputs=[
                    models_state,
                    model_dropdown,
                    X_test_state,
                    y_test_state,
                    encoding_input,
                    n_clusters_input,
                ],
                outputs=[cluster_brier_plot, cluster_heatmap_plot],
            )

        with gr.Tab("Inference"):
            with gr.Row():
                tooth_input = gr.Number(label="Tooth", value=43)
                toothtype_input = gr.Number(label="Tooth Type", value=1)
                rootnumber_input = gr.Number(label="Root Number", value=0)
                mobility_input = gr.Number(label="Mobility", value=1)
                restoration_input = gr.Number(label="Restoration", value=1)
                percussion_input = gr.Number(label="Percussion Sensitivity", value=0)
                sensitivity_input = gr.Number(label="Sensitivity", value=0)
                furcation_input = gr.Number(label="Furcation Baseline", value=0)
                side_input = gr.Number(label="Side", value=3)
                pdbaseline_input = gr.Number(label="PD Baseline", value=6)
                recbaseline_input = gr.Number(label="REC Baseline", value=4)
                plaque_input = gr.Number(label="Plaque", value=0)
                bop_input = gr.Number(label="BOP", value=0)
                age_input = gr.Number(label="Age", value=30)
                gender_input = gr.Number(label="Gender", value=1)
                bmi_input = gr.Number(label="Body Mass Index", value=35.0)
                perio_history_input = gr.Number(label="Perio Family History", value=2)
                diabetes_input = gr.Number(label="Diabetes", value=1)
                smokingtype_input = gr.Number(label="Smoking Type", value=1)
                cigarettenumber_input = gr.Number(label="Cigarette Number", value=0)
                antibiotics_input = gr.Number(label="Antibiotic Treatment", value=1)
                stresslvl_input = gr.Dropdown(
                    label="Stress Level",
                    choices=["low", "medium", "high"],
                    value="high",
                )

            task_display = gr.Textbox(
                label="Selected Task", value="", interactive=False
            )

            inference_model_dropdown = gr.Dropdown(
                label="Select Model", choices=[], value=None, multiselect=False
            )

            encoding_input = gr.Dropdown(
                label="Encoding",
                choices=["one_hot", "target"],
                value="one_hot",
            )

            task_display.value = task_input.value
            task_input.change(
                fn=lambda task: task, inputs=task_input, outputs=task_display
            )

            task_input.change(
                fn=lambda task, encoding: load_data(
                    InputProcessor.process_tasks([task])[0], encoding
                )[1:],
                inputs=[task_input, encoding_input],
                outputs=[X_train_state, y_train_state, X_test_state, y_test_state],
            )

            encoding_input.change(
                fn=lambda task, encoding: load_data(
                    InputProcessor.process_tasks([task])[0], encoding
                )[1:],
                inputs=[task_input, encoding_input],
                outputs=[X_train_state, y_train_state, X_test_state, y_test_state],
            )

            predict_button = gr.Button("Run Single Prediction")
            prediction_output = gr.Textbox(label="Prediction Output", interactive=False)
            probability_output = gr.Textbox(
                label="Probability Output", interactive=False
            )

            models_state.change(
                fn=update_model_dropdown,
                inputs=models_state,
                outputs=inference_model_dropdown,
            )

            predict_button.click(
                fn=run_single_inference,
                inputs=[
                    task_input,
                    models_state,
                    inference_model_dropdown,
                    tooth_input,
                    toothtype_input,
                    rootnumber_input,
                    mobility_input,
                    restoration_input,
                    percussion_input,
                    sensitivity_input,
                    furcation_input,
                    side_input,
                    pdbaseline_input,
                    recbaseline_input,
                    plaque_input,
                    bop_input,
                    age_input,
                    gender_input,
                    bmi_input,
                    perio_history_input,
                    diabetes_input,
                    smokingtype_input,
                    cigarettenumber_input,
                    antibiotics_input,
                    stresslvl_input,
                    encoding_input,
                    X_train_state,
                    y_train_state,
                ],
                outputs=[prediction_output, probability_output],
            )

app.launch()
