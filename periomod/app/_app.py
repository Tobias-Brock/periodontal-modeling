"""Gradio frontend for periodontal modeling.

Contains streamlined methods for plotting, benchmarking, evaluation and inference.

Example:
    ```
    from periomod.app import perioapp

    perioapp.launch()
    ```
"""

from functools import partial
from importlib.resources import files
from typing import Dict, List, Union

import gradio as gr

from periomod.app import (
    _app_inference,
    _baseline_wrapper,
    _benchmarks_wrapper,
    _brier_score_wrapper,
    _collect_data,
    _display_data,
    _handle_tooth_selection,
    _initialize_benchmark,
    _load_data_engine,
    _load_data_wrapper,
    _plot_bss,
    _plot_calibration,
    _plot_cluster_wrapper,
    _plot_cm,
    _plot_fi_wrapper,
    _plot_histogram_2d,
    _plot_matrix,
    _plot_outcome_descriptive,
    _plot_pocket_comparison,
    _plot_pocket_group_comparison,
    _process_data,
    _run_jackknife_inference,
    _save_data,
    _update_criteria_fields,
    _update_hpo_method_fields,
    _update_learners_fields,
    _update_model_dropdown,
    _update_side_state,
    _update_task_fields,
    _update_tooth_state,
    _update_tuning_method_fields,
    all_teeth,
)

logo_path = files("periomod.app.images").joinpath("logo_app.png")

with gr.Blocks(
    css="""
    .hide-label > label {
        display: none !important;
    }
    .no-box {
        border: none !important;
        box-shadow: none !important;
    }
"""
) as perioapp:
    gr.Image(
        logo_path,
        elem_id="logo",
        label="",
        show_download_button=False,
        show_fullscreen_button=False,
        elem_classes=["no-box", "hide-label"],
    )

    models_state = gr.State()
    task_state = gr.State()
    encoding_state = gr.State()
    processed_data_state = gr.State()
    train_df_state = gr.State()
    X_train_state = gr.State()
    y_train_state = gr.State()
    X_test_state = gr.State()
    y_test_state = gr.State()
    side_data_state = gr.State({})

    with gr.Tabs():
        with gr.Tab("Data"):
            with gr.Row():
                path_input = gr.Textbox(
                    label="File Path",
                    value="data/raw/raw_data.xlsx",
                    scale=1,
                    info="Specify the path to the raw data file for processing.",
                )
            load_button = gr.Button("Load Data", scale=1)
            load_output = gr.Textbox(label="Status", interactive=False, scale=2)

            display_button = gr.Button("Display Data", scale=1)
            display_output = gr.Dataframe(
                label="Data Preview", interactive=False, scale=6
            )

            process_button = gr.Button("Process Data", scale=1)
            process_output = gr.Textbox(
                label="Process Output", interactive=False, scale=6, lines=8
            )

            save_path_input = gr.Textbox(
                label="Save Path",
                value="data/processed/processed_data.csv",
                scale=1,
            )
            save_button = gr.Button("Save Data", scale=1)
            save_output = gr.Textbox(label="Save Status", interactive=False, scale=2)

            with gr.Row():
                column_before_input = gr.Dropdown(
                    label="Column Before",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column representing initial values before therapy.",
                )
                column_after_input = gr.Dropdown(
                    label="Column After",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column representing values after therapy.",
                )
                x_2dh_input = gr.Textbox(
                    label="x-Axis Label",
                    scale=1,
                    value="Pocket depth before therapy [mm]",
                    info="Enter label for the x-axis of the 2D histogram plot.",
                )
                y_2dh_input = gr.Textbox(
                    label="y-Axis Label",
                    scale=1,
                    value="Pocket after before therapy [mm]",
                    info="Enter label for the y-axis of the 2D histogram plot.",
                )
                plot_hist_2d_button = gr.Button("Plot 2D Histogram", scale=1)
            hist_2d_output = gr.Plot(scale=6)

            with gr.Row():
                column1_input = gr.Dropdown(
                    label="Column 1",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column representing initial values before therapy.",
                )
                column2_input = gr.Dropdown(
                    label="Column 2",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column representing values after therapy.",
                )
                title1_pocket_input = gr.Textbox(
                    label="Title 1",
                    scale=1,
                    value="Pocket depth before therapy",
                    info="Set title for the first comparison plot.",
                )
                title2_pocket_input = gr.Textbox(
                    label="Title 2",
                    scale=1,
                    value="Pocket after before therapy",
                    info="Set title for the second comparison plot.",
                )
                plot_comparison_button = gr.Button("Plot Pocket Comparison", scale=1)
            pocket_comparison_output = gr.Plot(scale=6)

            with gr.Row():
                group_column_before_input = gr.Dropdown(
                    label="Group Before",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column representing initial values before therapy.",
                )
                group_column_after_input = gr.Dropdown(
                    label="Group After",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column representing values after therapy.",
                )
                title1_pocket_group_input = gr.Textbox(
                    label="Title 1",
                    scale=1,
                    value="Pocket depth before therapy",
                    info="Set title for the first comparison plot.",
                )
                title2_pocket_group_input = gr.Textbox(
                    label="Title 2",
                    scale=1,
                    value="Pocket after before therapy",
                    info="Set the title for the second comparison plot.",
                )
                plot_group_comparison_button = gr.Button(
                    "Plot Pocket Group Comparison", scale=1
                )
            group_comparison_output = gr.Plot(scale=6)

            with gr.Row():
                vertical_input = gr.Dropdown(
                    label="Vertical Column",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column for the vertical axis in the matrix plot.",
                )
                horizontal_input = gr.Dropdown(
                    label="Horizontal Column",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select column for the horizontal axis in the matrix plot.",
                )
                x_matrix_input = gr.Textbox(
                    label="x-Axis Label",
                    scale=1,
                    value="Pocket depth before therapy",
                    info="Enter label for the x-axis in the matrix plot.",
                )
                y_matrix_input = gr.Textbox(
                    label="y-Axis Label",
                    scale=1,
                    value="Pocket after before therapy",
                    info="Enter label for the y-axis in the matrix plot.",
                )
                plot_matrix_button = gr.Button("Plot Matrix", scale=1)
            matrix_output = gr.Plot(scale=6)

            with gr.Row():
                outcome_input = gr.Dropdown(
                    label="Outcome Column",
                    choices=[],
                    scale=1,
                    value=None,
                    info="Select the column to analyze distribution.",
                )
                title_input = gr.Textbox(
                    label="Plot Title",
                    value="Distribution of Classes",
                    scale=1,
                    info="Enter the title of the  plot.",
                )
                plot_outcome_button = gr.Button("Plot Outcome Descriptive", scale=1)
            outcome_output = gr.Plot(scale=6)

            load_button.click(
                fn=_load_data_engine,
                inputs=[path_input],
                outputs=[
                    load_output,
                    column_before_input,
                    column_after_input,
                    column1_input,
                    column2_input,
                    group_column_before_input,
                    group_column_after_input,
                    horizontal_input,
                    vertical_input,
                    outcome_input,
                ],
            )
            display_button.click(
                fn=_display_data,
                inputs=None,
                outputs=[display_output],
            )
            process_button.click(
                fn=_process_data,
                inputs=None,
                outputs=[
                    process_output,
                    column_before_input,
                    column_after_input,
                    column1_input,
                    column2_input,
                    group_column_before_input,
                    group_column_after_input,
                    vertical_input,
                    horizontal_input,
                    outcome_input,
                ],
            )
            save_button.click(
                fn=_save_data,
                inputs=[save_path_input],
                outputs=[save_output],
            )
            plot_hist_2d_button.click(
                fn=_plot_histogram_2d,
                inputs=[
                    column_before_input,
                    column_after_input,
                    x_2dh_input,
                    y_2dh_input,
                ],
                outputs=hist_2d_output,
            )
            plot_comparison_button.click(
                fn=_plot_pocket_comparison,
                inputs=[
                    column1_input,
                    column2_input,
                    title1_pocket_input,
                    title2_pocket_input,
                ],
                outputs=pocket_comparison_output,
            )
            plot_group_comparison_button.click(
                fn=_plot_pocket_group_comparison,
                inputs=[
                    group_column_before_input,
                    group_column_after_input,
                    title1_pocket_group_input,
                    title2_pocket_group_input,
                ],
                outputs=group_comparison_output,
            )
            plot_matrix_button.click(
                fn=_plot_matrix,
                inputs=[
                    vertical_input,
                    horizontal_input,
                    x_matrix_input,
                    y_matrix_input,
                ],
                outputs=matrix_output,
            )
            plot_outcome_button.click(
                fn=_plot_outcome_descriptive,
                inputs=[outcome_input, title_input],
                outputs=outcome_output,
            )

        with gr.Tab("Benchmarking"):
            path_input = gr.Textbox(
                label="File Path",
                value="data/processed/processed_data.csv",
                info="Specify the path to the processed data file for benchmarking.",
            )

            with gr.Row():
                task_input = gr.Dropdown(
                    label="Task",
                    choices=[
                        "Pocket closure",
                        "Pocket closure PdBaseline > 3",
                        "Pocket improvement",
                        "Pocket groups",
                    ],
                    value="Pocket closure",
                    info="Select the task for benchmarking.",
                )
                learners_input = gr.CheckboxGroup(
                    label="Learners",
                    choices=[
                        "XGBoost",
                        "Random Forest",
                        "Logistic Regression",
                        "Multilayer Perceptron",
                    ],
                    value=["XGBoost"],
                    info="Select machine learning algorithms for benchmarking.",
                )

            with gr.Row():
                tuning_methods_input = gr.Radio(
                    label="Tuning Methods",
                    choices=["Holdout", "Cross-Validation"],
                    value="Holdout",
                    info="Choose the validation strategy to tune models.",
                )
                hpo_methods_input = gr.Radio(
                    label="HPO Methods",
                    choices=["HEBO", "Random Search"],
                    value="HEBO",
                    info="Select hyperparameter optimization method(s).",
                )
                criteria_input = gr.Radio(
                    label="Criteria",
                    choices=["F1 Score", "Brier Score"],
                    value="F1 Score",
                    info="Choose evaluation metrics to assess model performance.",
                )

            with gr.Row():
                encoding_input = gr.Radio(
                    label="Encoding",
                    choices=["One-hot", "Target"],
                    value="One-hot",
                    info="Select encoding type(s) for categorical features.",
                )
                sampling_input = gr.CheckboxGroup(
                    label="Sampling Strategy",
                    choices=["None", "upsampling", "downsampling", "smote"],
                    value=["None"],
                    info="Choose a sampling strategy to address class imbalance.",
                )
                factor_input = gr.Textbox(
                    label="Sampling Factor",
                    value="",
                    info="Specify a factor for resampling methods if applicable.",
                )

            with gr.Row():
                n_configs_input = gr.Number(
                    label="Num Configs",
                    value=3,
                    info="Enter number of iterations for hyperparameter tuning.",
                )
                cv_folds_input = gr.Number(
                    label="CV Folds",
                    value=None,
                    interactive=False,
                    info="Specify number of folds for cross-validation.",
                )
                racing_folds_input = gr.Number(
                    label="Racing Folds",
                    value=None,
                    interactive=False,
                    info="Enter number of folds for racing in hyperparameter tuning.",
                )
                n_jobs_input = gr.Number(
                    label="Num Jobs",
                    value=-1,
                    info="Set number of parallel jobs. Use -1 to utilize all CPUs.",
                )

            with gr.Row():
                test_seed_input = gr.Number(
                    label="Test Seed",
                    value=0,
                    info="Specify random seed for test set splitting.",
                )
                cv_seed_input = gr.Number(
                    label="CV Seed",
                    value=None,
                    interactive=False,
                    info="Set random seed for cross-validation splitting.",
                )
                test_size_input = gr.Number(
                    label="Test Set Size",
                    value=0.2,
                    minimum=0.0,
                    maximum=1.0,
                    info="Define proportion of data to allocate to the test set.",
                )
                val_size_input = gr.Number(
                    label="Val Set Size",
                    value=0.2,
                    interactive=True,
                    minimum=0.0,
                    maximum=1.0,
                    info="Define proportion of training data for validation.",
                )

            with gr.Row():
                mlp_flag_input = gr.Checkbox(
                    label="Enable MLP Training with Early Stopping",
                    value=None,
                    interactive=False,
                    info="Enable or disable early stopping for MLP training.",
                )
                threshold_tuning_input = gr.Checkbox(
                    label="Enable Threshold Tuning",
                    value=True,
                    interactive=True,
                    info="Enable or disable threshold tuning for classification tasks.",
                )

            run_baseline = gr.Button("Run Baseline")
            baseline_output = gr.Dataframe(label="Baseline Results")
            run_benchmark = gr.Button("Run Benchmark")

            benchmark_output = gr.Dataframe(label="Benchmark Results")
            metrics_plot_output = gr.Plot(label="Metrics Comparison")

            tuning_methods_input.change(
                fn=_update_tuning_method_fields,
                inputs=tuning_methods_input,
                outputs=[cv_folds_input, cv_seed_input, val_size_input],
            )

            tuning_methods_input.change(
                fn=_update_hpo_method_fields,
                inputs=[tuning_methods_input, hpo_methods_input],
                outputs=racing_folds_input,
            )

            hpo_methods_input.change(
                fn=_update_hpo_method_fields,
                inputs=[tuning_methods_input, hpo_methods_input],
                outputs=racing_folds_input,
            )

            learners_input.change(
                fn=_update_learners_fields,
                inputs=learners_input,
                outputs=mlp_flag_input,
            )

            criteria_input.change(
                fn=_update_criteria_fields,
                inputs=criteria_input,
                outputs=threshold_tuning_input,
            )

            task_input.change(
                fn=_update_task_fields,
                inputs=task_input,
                outputs=criteria_input,
            )

            path_input.change(
                fn=lambda x: x, inputs=path_input, outputs=processed_data_state
            )
            task_input.change(fn=lambda x: x, inputs=task_input, outputs=task_state)
            encoding_input.change(
                fn=lambda x: x, inputs=encoding_input, outputs=encoding_state
            )

            perioapp.load(
                _initialize_benchmark,
                inputs=[
                    tuning_methods_input,
                    hpo_methods_input,
                    learners_input,
                    criteria_input,
                    task_input,
                ],
                outputs=[
                    cv_folds_input,
                    cv_seed_input,
                    val_size_input,
                    racing_folds_input,
                    mlp_flag_input,
                    threshold_tuning_input,
                    criteria_input,
                ],
            )

            run_baseline.click(
                fn=_baseline_wrapper,
                inputs=[
                    task_input,
                    encoding_input,
                    path_input,
                ],
                outputs=[baseline_output],
            )

            run_benchmark.click(
                fn=_benchmarks_wrapper,
                inputs=[
                    task_input,
                    learners_input,
                    tuning_methods_input,
                    hpo_methods_input,
                    criteria_input,
                    encoding_input,
                    sampling_input,
                    factor_input,
                    n_configs_input,
                    cv_folds_input,
                    racing_folds_input,
                    test_seed_input,
                    test_size_input,
                    val_size_input,
                    cv_seed_input,
                    mlp_flag_input,
                    threshold_tuning_input,
                    n_jobs_input,
                    path_input,
                ],
                outputs=[benchmark_output, metrics_plot_output, models_state],
            )

        with gr.Tab("Evaluation"):
            with gr.Row():
                task_display = gr.Textbox(
                    label="Selected Task",
                    value="",
                    interactive=False,
                    info="Displays task currently selected for evaluation.",
                )
                model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=[],
                    value=None,
                    multiselect=False,
                    info="Select a model from the available trained models.",
                )
                encoding_display = gr.Textbox(
                    label="Encoding",
                    value="",
                    interactive=False,
                    info="Displays encoding method used for categorical variables.",
                )
            with gr.Row():
                processed_data_display = gr.Textbox(
                    label="File Path",
                    value="",
                    scale=1,
                    info="Specify the path to the processed data file for evaluation.",
                )

            load_data_button = gr.Button("Load Data")
            load_status_output = gr.Textbox(label="Status", interactive=False)
            processed_data_display.value = path_input.value
            task_display.value = task_input.value
            encoding_display.value = encoding_input.value

            models_state.change(
                fn=_update_model_dropdown,
                inputs=models_state,
                outputs=model_dropdown,
            )

            path_input.change(
                fn=lambda x: x, inputs=path_input, outputs=processed_data_display
            )
            task_input.change(
                fn=lambda task: task, inputs=task_input, outputs=task_display
            )

            encoding_input.change(
                fn=lambda encoding: encoding,
                inputs=encoding_input,
                outputs=encoding_display,
            )

            generate_confusion_matrix_button = gr.Button("Generate Confusion Matrix")
            matrix_plot = gr.Plot()

            generate_brier_scores_button = gr.Button("Generate Brier Scores")
            brier_score_plot = gr.Plot()

            generate_calibration_button = gr.Button("Generate Calibration Plot")
            calibration_plot = gr.Plot()

            generate_bss_button = gr.Button("Generate Brier Skill Score Plot")
            bss_plot = gr.Plot()

            with gr.Row():
                importance_type_input = gr.Radio(
                    label="Importance Types",
                    choices=["shap", "permutation", "standard"],
                    value="shap",
                )
                aggregate_fi_input = gr.Checkbox(
                    label="Aggregate Features",
                    value=True,
                    info="Aggregate encoded Multi-Category Features",
                )

            generate_feature_importance_button = gr.Button(
                "Generate Feature Importance"
            )
            fi_plot = gr.Plot()

            with gr.Row():
                n_clusters_input = gr.Slider(
                    label="Number of Clusters", minimum=2, maximum=10, step=1, value=3
                )
                aggregate_cluster_input = gr.Checkbox(
                    label="Aggregate Features",
                    value=True,
                    info="Aggregate encoded Multi-Category Features",
                )

            cluster_button = gr.Button("Perform Brier Score Clustering")
            cluster_brier_plot = gr.Plot()
            cluster_heatmap_plot = gr.Plot()

            load_data_button.click(
                fn=_load_data_wrapper,
                inputs=[task_input, encoding_input, path_input],
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
                fn=_plot_cm,
                inputs=[
                    models_state,
                    model_dropdown,
                    X_test_state,
                    y_test_state,
                    task_input,
                ],
                outputs=matrix_plot,
            )

            generate_brier_scores_button.click(
                fn=_brier_score_wrapper,
                inputs=[
                    models_state,
                    model_dropdown,
                    X_test_state,
                    y_test_state,
                    task_input,
                ],
                outputs=brier_score_plot,
            )

            generate_calibration_button.click(
                fn=_plot_calibration,
                inputs=[
                    models_state,
                    model_dropdown,
                    X_test_state,
                    y_test_state,
                    task_input,
                ],
                outputs=calibration_plot,
            )

            generate_bss_button.click(
                fn=_plot_bss,
                inputs=[
                    models_state,
                    model_dropdown,
                    X_test_state,
                    y_test_state,
                    task_input,
                    encoding_input,
                    path_input,
                ],
                outputs=bss_plot,
            )

            generate_feature_importance_button.click(
                fn=_plot_fi_wrapper,
                inputs=[
                    models_state,
                    model_dropdown,
                    importance_type_input,
                    X_test_state,
                    y_test_state,
                    encoding_input,
                    aggregate_fi_input,
                ],
                outputs=fi_plot,
            )

            cluster_button.click(
                fn=_plot_cluster_wrapper,
                inputs=[
                    models_state,
                    model_dropdown,
                    X_test_state,
                    y_test_state,
                    encoding_input,
                    aggregate_cluster_input,
                    n_clusters_input,
                ],
                outputs=[cluster_brier_plot, cluster_heatmap_plot],
            )

        with gr.Tab("Inference"):
            with gr.Row():
                gender_input = gr.Radio(
                    label="Gender",
                    choices=["Male", "Female"],
                    value="Male",
                    info="Select the patient's gender.",
                )
                age_input = gr.Number(
                    label="Age",
                    value=30,
                    minimum=0,
                    maximum=120,
                    step=1,
                    info="Enter the patient's age.",
                )
                bmi_input = gr.Number(
                    label="Body Mass Index",
                    value=35.0,
                    minimum=0,
                    info="Enter the patient's Body Mass Index (BMI).",
                )
                smokingtype_input = gr.Radio(
                    label="Smoking Type",
                    choices=["no", "Cigarette", "Pipe", "Cigarillo", "all"],
                    value="no",
                    info="Specify the type of smoking habit, if any.",
                )
                cigarettenumber_input = gr.Number(
                    label="Cigarette Number",
                    value=0,
                    minimum=0,
                    step=1,
                    info="Enter the average number of cigarettes smoked per day.",
                )
            with gr.Row():
                perio_history_input = gr.Radio(
                    label="Perio Family History",
                    choices=["yes", "no", "unknown"],
                    value="no",
                    info="Is there a family history of periodontal disease?",
                )
                diabetes_input = gr.Radio(
                    label="Diabetes",
                    choices=["no", "Type I", "Type II", "Type II med."],
                    value="no",
                    info="Select the type of diabetes, if diagnosed.",
                )
                antibiotics_input = gr.Radio(
                    label="Antibiotic Treatment",
                    choices=["yes", "no"],
                    value="no",
                    info="Patient recieved adjunct antibiotic treatment.",
                )
                stresslvl_input = gr.Radio(
                    label="Stress Level",
                    choices=["low", "mid", "high"],
                    value="low",
                    info="Specify the patient's perceived stress level.",
                )

            tooth_features = [
                (
                    "Mobility",
                    "mobility",
                    "radio",
                    ["yes", "no"],
                    "Can the tooth be moved?",
                ),
                (
                    "Restoration",
                    "restoration",
                    "radio",
                    ["no", "Filling", "Crown"],
                    "Is a type of restoration present?",
                ),
                (
                    "Percussion",
                    "percussion",
                    "radio",
                    ["yes", "no"],
                    "Is the tooth percussion sensitive?",
                ),
                (
                    "Sensitivity",
                    "sensitivity",
                    "radio",
                    ["yes", "no"],
                    "Has the tooth vitality/sensitivity",
                ),
            ]

            side_features = [
                (
                    "Furcation:\n\n What furcation involvement does the tooth exhibit?",
                    "furcationbaseline",
                    "radio",
                    ["no", "Palpable", "1-3 mm", ">3 mm"],
                ),
                (
                    "PD Baseline:\n\n Pocket depth at baseline examination in mm.",
                    "pdbaseline",
                    "textbox",
                    None,
                ),
                (
                    "REC Baseline:\n\n Recession at baseline examination in mm.",
                    "recbaseline",
                    "textbox",
                    None,
                ),
                (
                    "Plaque:\n\n Was plaque found at the toothside?",
                    "plaque",
                    "radio",
                    ["yes", "no"],
                ),
                ("BOP: \n\n Bleeding on Probing", "bop", "radio", ["yes", "no"]),
            ]

            tooth_selector = gr.Radio(
                label="Select Tooth",
                choices=[str(tooth) for tooth in all_teeth],
                value=str(all_teeth[0]),
                info="Select the tooth of interest",
            )

            tooth_choices: Union[str, List[str], None]
            tooth_states = gr.State({})
            tooth_components: Dict[str, gr.components.Component] = {}
            with gr.Row():
                for (
                    feature_label,
                    feature_key,
                    input_type,
                    tooth_choices,
                    info,
                ) in tooth_features:
                    if input_type == "radio":
                        input_component = gr.Radio(
                            label=feature_label,
                            choices=tooth_choices,
                            value=None,
                            info=info,
                        )
                    else:
                        input_component = gr.Dropdown(
                            label=feature_label,
                            choices=tooth_choices,
                            value=None,
                            info=info,
                        )
                    tooth_components[feature_key] = input_component

            sides_components: Dict[int, Dict[str, gr.components.Component]] = {}

            with gr.Row():
                gr.Markdown("### ")
                for side_num in range(1, 7):
                    gr.Markdown(f"**Side {side_num}**")

            side_choices: Union[str, List[str], None]
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
                    _handle_tooth_selection,
                    tooth_components=tooth_components,
                    sides_components=sides_components,
                ),
                inputs=[tooth_selector, tooth_states],
                outputs=input_components,
            )

            for input_name, component in tooth_components.items():
                component.change(
                    fn=partial(_update_tooth_state, input_name=input_name),
                    inputs=[tooth_states, tooth_selector, component],
                    outputs=tooth_states,
                )

            for side_num in range(1, 7):
                side_components = sides_components[side_num]
                for input_name, component in side_components.items():
                    component.change(
                        fn=partial(
                            _update_side_state, side_num=side_num, input_name=input_name
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
                fn=_collect_data,
                inputs=patient_inputs + [tooth_states],
                outputs=[output_message, patient_data],
            )
            with gr.Row():
                task_display = gr.Textbox(
                    label="Selected Task",
                    value="",
                    interactive=False,
                    info="Displays the selected task for inference.",
                )
                inference_model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=[],
                    value=None,
                    multiselect=False,
                    info="Choose the model to use for inference.",
                )
                encoding_display = gr.Textbox(
                    label="Encoding",
                    value="",
                    interactive=False,
                    info="Displays encoding method used for categorical variables.",
                )
            with gr.Row():
                processed_data_display = gr.Textbox(
                    label="File Path",
                    value="",
                    scale=1,
                    info="Specify the path to the processed data file for evaluation.",
                )

            results = gr.DataFrame(visible=False)
            processed_data_display.value = path_input.value
            task_display.value = task_input.value
            encoding_display.value = encoding_input.value
            path_input.change(
                fn=lambda x: x, inputs=path_input, outputs=processed_data_display
            )
            task_input.change(
                fn=lambda task: task, inputs=task_input, outputs=task_display
            )
            encoding_input.change(
                fn=lambda encoding: encoding,
                inputs=encoding_input,
                outputs=encoding_display,
            )

            task_input.change(
                fn=_load_data_wrapper,
                inputs=[task_input, encoding_input, path_input],
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
                fn=_load_data_wrapper,
                inputs=[task_input, encoding_input, path_input],
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
                fn=_update_model_dropdown,
                inputs=models_state,
                outputs=inference_model_dropdown,
            )

            inference_button.click(
                fn=_app_inference,
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

            with gr.Row():
                n_jobs_input = gr.Number(
                    label="Number of Parallel Jobs (n_jobs)",
                    value=-1,
                    precision=0,
                    info="Set number of parallel jobs (-1 uses all available cores).",
                )
                alpha_input = gr.Number(
                    label="Confidence Level",
                    value=0.05,
                    minimum=0.0,
                    maximum=1.0,
                    info="Specify the confidence level for jackknife intervals.",
                )

            load_data_button.click(
                fn=_load_data_wrapper,
                inputs=[task_input, encoding_input, path_input],
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
                fn=_run_jackknife_inference,
                inputs=[
                    task_input,
                    models_state,
                    inference_model_dropdown,
                    train_df_state,
                    prediction_data,
                    encoding_input,
                    results,
                    alpha_input,
                    sample_fraction_input,
                    n_jobs_input,
                ],
                outputs=[jackknife_plot],
            )

        with gr.Tab("Achknowledgments"):
            gr.Markdown(
                """
            ## License

            © 2024 Tobias Brock, Elias Walter

            This project is licensed under the Creative Commons
            Attribution-NonCommercial-ShareAlike 4.0 International License.
            Read the full license at [Creative Commons](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

            ## Correspondence

            Tobias Brock: t.brock@campus.lmu.de  \n
            Elias Walter: elias.walter@med.uni-muenchen.de
            """
            )


perioapp.launch(server_port=7890, server_name="0.0.0.0")
