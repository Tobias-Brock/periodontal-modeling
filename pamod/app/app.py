"""Gradio frontend."""

from functools import partial
from typing import Dict, List, Union

import gradio as gr

from pamod.app import (
    all_teeth,
    app_inference,
    benchmarks_wrapper,
    brier_score_wrapper,
    collect_data,
    handle_tooth_selection,
    load_and_initialize_plotter,
    load_data_wrapper,
    plot_cluster_wrapper,
    plot_cm,
    plot_fi_wrapper,
    plot_histogram_2d,
    plot_matrix,
    plot_outcome_descriptive,
    plot_pocket_comparison,
    plot_pocket_group_comparison,
    run_jackknife_inference,
    update_model_dropdown,
    update_side_state,
    update_tooth_state,
)
from pamod.config import PROCESSED_BASE_DIR, RAW_DATA_DIR

with gr.Blocks() as app:
    gr.Markdown("## ML Periodontal Modeling")

    models_state = gr.State()
    task_state = gr.State()
    train_df_state = gr.State()
    X_train_state = gr.State()
    y_train_state = gr.State()
    X_test_state = gr.State()
    y_test_state = gr.State()
    side_data_state = gr.State({})

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
                label="Task",
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

            sampling_input = gr.CheckboxGroup(
                label="Sampling Strategy",
                choices=["None", "upsampling", "downsampling", "smote"],
                value=["None"],  # Default value is None
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
                fn=load_data_wrapper,
                inputs=[task_input, encoding_input],
                outputs=[
                    load_status_output,
                    train_df_state,
                    X_train_state,
                    y_train_state,
                    X_test_state,
                    y_test_state,
                ],
            )

            train_df_state.change(
                fn=lambda x: x, inputs=train_df_state, outputs=train_df_state
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
                age_input = gr.Number(
                    label="Age", value=30, minimum=0, maximum=120, step=1
                )
                gender_input = gr.Radio(label="Gender", choices=[0, 1], value=1)
                bmi_input = gr.Number(label="Body Mass Index", value=35.0, minimum=0)
                perio_history_input = gr.Radio(
                    label="Perio Family History", choices=[0, 1, 2], value=2
                )
                diabetes_input = gr.Radio(
                    label="Diabetes", choices=[0, 1, 2, 3], value=1
                )
                smokingtype_input = gr.Radio(
                    label="Smoking Type", choices=[0, 1, 2, 3, 4], value=1
                )
                cigarettenumber_input = gr.Number(
                    label="Cigarette Number", value=0, minimum=0, step=1
                )
                antibiotics_input = gr.Radio(
                    label="Antibiotic Treatment", choices=[0, 1], value=1
                )
                stresslvl_input = gr.Radio(
                    label="Stress Level", choices=[0, 1, 2], value=2
                )

            tooth_features = [
                ("Mobility", "mobility", "radio", [0, 1]),
                ("Restoration", "restoration", "radio", [0, 1, 2]),
                ("Percussion", "percussion", "radio", [0, 1]),
                ("Sensitivity", "sensitivity", "radio", [0, 1]),
            ]

            side_features = [
                (
                    "Furcation",
                    "furcationbaseline",
                    "radio",
                    [0, 1, 2],
                ),
                ("PD Baseline", "pdbaseline", "textbox", None),
                ("REC Baseline", "recbaseline", "textbox", None),
                ("Plaque", "plaque", "radio", [0, 1]),
                ("BOP", "bop", "radio", [0, 1]),
            ]

            tooth_selector = gr.Radio(
                label="Select Tooth",
                choices=[str(tooth) for tooth in all_teeth],
                value=str(all_teeth[0]),
            )

            tooth_choices: Union[str, List[int], None]
            tooth_states = gr.State({})
            tooth_components: Dict[str, gr.components.Component] = {}
            with gr.Row():
                for (
                    feature_label,
                    feature_key,
                    input_type,
                    tooth_choices,
                ) in tooth_features:
                    if input_type == "radio":
                        input_component = gr.Radio(
                            label=feature_label,
                            choices=tooth_choices,
                            value=None,
                        )
                    else:
                        input_component = gr.Dropdown(
                            label=feature_label,
                            choices=tooth_choices,
                            value=None,
                        )
                    tooth_components[feature_key] = input_component

            sides_components: Dict[int, Dict[str, gr.components.Component]] = {}

            with gr.Row():
                gr.Markdown("### ")
                for side_num in range(1, 7):
                    gr.Markdown(f"**Side {side_num}**")

            side_choices: Union[List[int], None]
            for feature_label, feature_key, input_type, side_choices in side_features:
                with gr.Row():
                    gr.Markdown(f"{feature_label}")
                    for side_num in range(1, 7):
                        side_components = sides_components.setdefault(side_num, {})
                        if input_type == "radio":
                            input_component = gr.Radio(
                                label="",
                                choices=(
                                    side_choices if side_choices is not None else []
                                ),
                                value=None,
                            )
                        elif input_type == "textbox":
                            input_component = gr.Textbox(
                                label="",
                                value="",
                                placeholder="Enter number",
                            )
                        else:
                            input_component = gr.Dropdown(
                                label="",
                                choices=(
                                    side_choices if side_choices is not None else []
                                ),
                                value=None,
                            )
                        side_components[feature_key] = input_component

            input_components = list(tooth_components.values())
            for side_num in range(1, 7):
                input_components.extend(sides_components[side_num].values())

            tooth_selector.change(
                fn=partial(
                    handle_tooth_selection,
                    tooth_components=tooth_components,
                    sides_components=sides_components,
                ),
                inputs=[tooth_selector, tooth_states],
                outputs=input_components,
            )

            for input_name, component in tooth_components.items():
                component.change(
                    fn=partial(update_tooth_state, input_name=input_name),
                    inputs=[tooth_states, tooth_selector, component],
                    outputs=tooth_states,
                )

            for side_num in range(1, 7):
                side_components = sides_components[side_num]
                for input_name, component in side_components.items():
                    component.change(
                        fn=partial(
                            update_side_state, side_num=side_num, input_name=input_name
                        ),
                        inputs=[tooth_states, tooth_selector, component],
                        outputs=tooth_states,
                    )

            submit_button = gr.Button("Submit")
            output_message = gr.Textbox(label="Output")
            patient_data = gr.Dataframe(visible=False)

            patient_inputs = [
                age_input,
                gender_input,
                bmi_input,
                perio_history_input,
                diabetes_input,
                smokingtype_input,
                cigarettenumber_input,
                antibiotics_input,
                stresslvl_input,
            ]

            submit_button.click(
                fn=collect_data,
                inputs=patient_inputs + [tooth_states],
                outputs=[output_message, patient_data],
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

            results = gr.DataFrame(visible=False)

            task_display.value = task_input.value
            task_input.change(
                fn=lambda task: task, inputs=task_input, outputs=task_display
            )

            task_input.change(
                fn=load_data_wrapper,
                inputs=[task_input, encoding_input],
                outputs=[
                    load_status_output,
                    train_df_state,
                    X_train_state,
                    y_train_state,
                    X_test_state,
                    y_test_state,
                ],
            )

            encoding_input.change(
                fn=load_data_wrapper,
                inputs=[task_input, encoding_input],
                outputs=[
                    load_status_output,
                    train_df_state,
                    X_train_state,
                    y_train_state,
                    X_test_state,
                    y_test_state,
                ],
            )

            prediction_data = gr.Dataframe(visible=False)
            inference_button = gr.Button("Run Inference")
            prediction_output = gr.Dataframe(label="Prediction Results")

            models_state.change(
                fn=update_model_dropdown,
                inputs=models_state,
                outputs=inference_model_dropdown,
            )

            inference_button.click(
                fn=app_inference,
                inputs=[
                    task_input,
                    models_state,
                    inference_model_dropdown,
                    patient_data,
                    encoding_input,
                    X_train_state,
                    y_train_state,
                ],
                outputs=[prediction_data, prediction_output, results],
            )

            load_data_button = gr.Button("Load Data")
            load_status_output = gr.Textbox(label="Status")

            sample_fraction_input = gr.Slider(
                label="Sample Fraction for Jackknife Resampling",
                minimum=0.1,
                maximum=1.0,
                step=0.1,
                value=1.0,
            )

            n_jobs_input = gr.Number(
                label="Number of Parallel Jobs (n_jobs)",
                value=-1,
                precision=0,
            )

            load_data_button.click(
                fn=load_data_wrapper,
                inputs=[task_input, encoding_input],
                outputs=[
                    load_status_output,
                    train_df_state,
                    X_train_state,
                    y_train_state,
                    X_test_state,
                    y_test_state,
                ],
            )

            jackknife_button = gr.Button("Run Jackknife Inference")
            jackknife_plot = gr.Plot(label="Confidence Intervals Plot")

            jackknife_button.click(
                fn=run_jackknife_inference,
                inputs=[
                    task_input,
                    models_state,
                    inference_model_dropdown,
                    train_df_state,
                    prediction_data,
                    encoding_input,
                    results,
                    sample_fraction_input,
                    n_jobs_input,
                ],
                outputs=[jackknife_plot],
            )

app.launch()
