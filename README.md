# periodontal-modeling

A Python package for comprehensive periodontal data processing and modeling. This package provides tools for preprocessing, automatic hyperparameter tuning, resampling, model evaluation, inference, and descriptive analysis with an interactive Gradio frontend. It was developed for Python 3.11.

## Features

- **Preprocessing Pipeline**: Flexible preprocessing of periodontal data, including encoding, scaling, and transformation.
- **Descriptive Analysis**: Generate descriptive statistics and plots such as confusion matrices and bar plots.
- **Automatic Model Tuning**: Supports multiple learners and tuning strategies for optimized model training.
- **Resampling and Handling Class Imbalance**: Resampling strategies such as SMOTE and upsampling/downsampling to balance dataset classes.
- **Model Evaluation**: Cross-validation and holdout evaluation with support for criteria such as F1, Brier score, and Macro-F1.
- **Inference and Descriptive Statistics**: Patient-level inference, jackknife resampling, and 2D histogram generation.
- **Interactive Frontend with Gradio**: A simple Gradio interface for streamlined model benchmarking, evaluation and inference.

## Installation

Ensure you have Python 3.11 installed. Install the package via pip:

```bash
pip install periodontal-modeling
```

## Usage

### App Module

The periomod app provides a streamlined gradio interface for plotting descriptives, performing benchmarks, model evaluation and inference. The app can be launched in a straighforward manner.

```python
from periomod.app import app

app.launch()
```

The app can also be launched using docker. Run the following commands in the root of the repository:

```bash
docker build -f docker/app.dockerfile -t periomod-image .
docker run -p 7880:7880 periomod-image
```
By default the app will be launched on port 7880 and can be accessed via `http://localhost:7880`.

Alternatively, the `make` commands can be used to build and run the docker image:

```bash
make docker-build
make docker-run
```

### Data Module

Use the `StaticProcessEngine` class to preprocess your data. This class handles data transformations and imputation.

```python
from periomod.data import StaticProcessEngine

# do not include behavior columns for processing data
# activate verbose logging during processing
engine = StaticProcessEngine(behavior=False, verbose=True)
df = engine.load_data(path="data/raw", name="Periodontitis_ML_Dataset.xlsx")
df = engine.process_data(df)
engine.save_data(df=df, path="data/processed", name="processed_data.csv")
```

The `ProcessedDataLoader` requires a fully imputed dataset. It contains methods for scaling and encoding. As encoding types, 'one_hot' and 'target' can be selected. The scale argument scales numerical columns. One out of three periodontal task can be selected, either "pocketclosure", "pdgrouprevaluation" or "improvement".

```python
from periomod.data import ProcessedDataLoader

# instantiate with one-hot encoding and scale numerical variables
dataloader = ProcessedDataLoader(
    task="pocketclosure", encoding="one_hot", encode=True, scale=True
)
df = dataloader.load_data(path="data/processed", name="processed_data.csv")
df = dataloader.transform_data(df=df)
dataloader.save_data(df=df, path="data/training", name="training_data.csv")
```

### Descriptives Module

`DesctiptivesPlotter` can be used to plot descriptive plots for target columns before and after treatment.

```python
from periomod.descriptives import DescriptivesPlotter

# instantiate plotter with dataframe
plotter = DescriptivesPlotter(df)
plotter.plt_matrix(vertical="depth_before", horizontal="depth_after")
plotter.pocket_comparison(col1="depth_before", col2="depth_after")
plotter.histogram_2d(col_before="depth_before", col_after="depth_after")
```

### Resampling Module

The `Resampler` class allows for straightforward grouped splitting operations. It also includes different sampling strategies to treat the minority classes.

```python
from periomod.resampling import Resampler

resampler = Resampler(classification="binary", encoding="one_hot")
train_df, test_df = resampler.split_train_test_df(df=df, seed=42, test_size=0.3)

# upsample minority class by a factor of 2.
X_train, y_train, X_test, y_test = resampler.split_x_y(
    train_df, test_df, sampling="upsampling", factor=2
)
# performs grouped cross-validation with "smote" sampling on the training folds
outer_splits, cv_folds_indices = resampler.cv_folds(
    df, sampling="smote", factor=2.0, seed=42, n_folds=5
)
```

### Training Module

`Trainer` contains different training methods that are used during hyperparameter tuning and benchmarking. It further includes methods for threshold tuning.

```python
from periomod.training import Trainer
from sklearn.ensemble import RandomForestClassifier

trainer = Trainer(classification="binary", criterion="f1", tuning="cv", hpo="hebo")

score, trained_model, threshold = trainer.train(
    model=RandomForestClassifier,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
)
print(f"Score: {score}, Optimal Threshold: {threshold}")
```

The `trainer.train_mlp` function uses the `partial_fit` method of the 'MLPClassifier' to leverage early stopping during the training process.

```python
from sklearn.neural_network import MLPClassifier

score, trained_mlp, threshold = trainer.train_mlp(
    mlp_model=MLPClassifier,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    final=True,
)
print(f"MLP Validation Score: {score}, Optimal Threshold: {threshold}")
```

### Tuning Module

The tuning module contains the `HEBOTuner`and `RandomSearchTuner` classes that can be used for hyperparameter tuning.
`HEBOTuner` leverages Bayesian optimization to obtain the optimal set of hyperparameters.

```python
from periomod.training import Trainer
from periomod.tuning import HEBOTuner

tuner = HEBOTuner(
    classification="binary",
    criterion="f1",
    tuning="holdout",
    hpo="hebo",
    n_configs=10,
    n_jobs=-1,
    verbose=True,
    trainer=Trainer(
        classification="binary",
        criterion="f1",
        tuning="holdout",
        hpo="hebo",
        mlp_training=True,
        threshold_tuning=True,
    ),
    mlp_training=True,
    threshold_tuning=True,
)

best_params, best_threshold = tuner.holdout(
    learner="rf", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
)

# Using cross-validation
best_params, best_threshold = tuner.cv(learner="rf", outer_splits=cross_val_splits)
```
`RandomSearchTuner` implements random search tuning by sampling parameters at random from specified ranges. Also, allows for racing when cross-validation is used as tuning technique.

```python
from periomod.training import Trainer
from periomod.tuning import RandomSearchTuner

tuner = RandomSearchTuner(
    classification="binary",
    criterion="f1",
    tuning="cv",
    hpo="rs",
    n_configs=15,
    n_jobs=4,
    verbose=True,
    trainer=Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="rs",
        mlp_training=True,
        threshold_tuning=True,
    ),
    mlp_training=True,
    threshold_tuning=True,
)

# Running holdout-based tuning
best_params, best_threshold = tuner.holdout(
    learner="rf", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
)

# Running cross-validation tuning
best_params, best_threshold = tuner.cv(learner="rf", outer_splits=cross_val_splits)
```

### Evaluation Module

`ModelEvaluator` contains method for model evaluation after training. It includes prediction analysis and feature importance.

```python
from periomod.evaluation import ModelEvaluator

evaluator = ModelEvaluator(
    X=X_test, y=y_test, model=trained_rf_model, encoding="one_hot"
)

# Plots confusion matrix of target column
evaluator.plot_confusion_matrix()

# plot feature importances
evaluator.evaluate_feature_importance(fi_types=["shap", "permutation"])

# perform feature clustering
brier_plot, heatmap_plot, clustered_data = evaluator.analyze_brier_within_clusters()
```

### Inference Module

The inference module includes methods for single but also patient-level predictions. Jackknife resampling and confidence intervals are also included.

```python
from periomod.inference import ModelInference

model_inference = ModelInference(
    classification="binary", model=trained_model, verbose=True
)

# Prepare data for inference
prepared_data, patient_data = model_inference.prepare_inference(
    task="classification_task",
    patient_data=patient_df,
    encoding="one_hot",
    X_train=train_df,
    y_train=target_series,
)

# Run inference on patient data
inference_results = model_inference.patient_inference(
    predict_data=prepared_data, patient_data=patient_data
)

# Perform jackknife inference with confidence interval plotting
jackknife_results, ci_plot = model_inference.jackknife_inference(
    model=trained_model,
    train_df=train_df,
    patient_data=patient_df,
    encoding="target",
    inference_results=inference_results,
    alpha=0.05,
    sample_fraction=0.8,
    n_jobs=4,
)
```

### Benchmarking Module

The benchmarking module contains methods to run single or multiple experiments with a specified tuning setup. For a single experiment the `Experiment`class can be used.

```python
from periomod.benchmarking import Experiment
from periomod.data import ProcessedDataLoader

# Load a dataframe with the correct target and encoding selected
dataloader = ProcessedDataLoader(
    task="pocketclosure", encoding="one_hot", encode=True, scale=True
)
df = dataloader.load_data(path="data/processed", name="processed_data.csv")
df = dataloader.transform_data(df=df)

experiment = Experiment(
    df=df,
    task="pocketclosure",
    learner="rf",
    criterion="f1",
    encoding="one_hot",
    tuning="cv",
    hpo="rs",
    sampling="upsample",
    factor=1.5,
    n_configs=20,
    racing_folds=3,
    n_jobs=-1,
    cv_folds=10,
    test_seed=42,
    test_size=0.2,
    val_size=0.1,
    cv_seed=10,
    mlp_flag=True,
    threshold_tuning=True,
    verbose=True,
)

# Perform the evaluation based on cross-validation
final_metrics = experiment.perform_evaluation()
print(final_metrics)
```

For running multiple experiments, the `Benchmarker`class can be used. It will output a dictionary based on the best 4 models for a respective tuning criterion and the full experiment runs in a dataframe.

```python
from periomod.benchmarking import Benchmarker

benchmarker = Benchmarker(
    task="pocketclosure",
    learners=["xgb", "rf", "lr"],
    tuning_methods=["holdout", "cv"],
    hpo_methods=["hebo", "rs"],
    criteria=["f1", "brier_score"],
    encodings=["one_hot", "target"],
    sampling=[None, "upsampling", "downsampling"],
    factor=2,
    n_configs=20,
    n_jobs=-1,
    cv_folds=5,
    test_seed=42,
    test_size=0.2,
    verbose=True,
    path="/data/processed",
    name="processed_data.csv",
)

# Running all benchmarks
results_df, top_models = benchmarker.run_all_benchmarks()
print(results_df)
print(top_models)
```

### Wrapper Module

The wrapper module wraps benchmark and evaluation methods to provide a streamlined setup that requires a minimal amount of code while making use of all the submodules contained in the `periomod` package.

```python
from periomod.wrapper import BenchmarkWrapper

# Initialize the BenchmarkWrapper
benchmarker = BenchmarkWrapper(
    task="pocketclosure",
    encodings=["one_hot", "target"],
    learners=["rf", "xgb", "lr"],
    tuning_methods=["holdout", "cv"],
    hpo_methods=["rs", "hebo"],
    criteria=["f1", "brier_score"],
    sampling=["upsampling"],
    factor=None,
    n_configs=10,
    n_jobs=4,
    verbose=True,
    path="data/processed",
    name="processed_data.csv",
)

# Run baseline benchmarking
baseline_df = benchmarker.baseline()

# Run full benchmark and retrieve results
benchmark_results, learners_used = benchmarker.wrapped_benchmark()

# Save the benchmark results
benchmarker.save_benchmark(baseline_df, path=Path("reports"))

# Save the trained learners
benchmarker.save_learners(learners_dict=learners_used, path=Path("models"))
```

```python
# Initialize the evaluator with required parameters
evaluator = EvaluatorWrapper(
    learners_dict=my_learners, criterion="f1", aggregate=True, verbose=True
)

# Evaluate the model and generate plots
evaluator.wrapped_evaluation(cm=True, brier_groups=True)

# Calculate feature importance
evaluator.evaluate_feature_importance(fi_types=["shap", "permutation"])

# Train and average over multiple random splits
avg_metrics_df = evaluator.average_over_splits(num_splits=5, n_jobs=-1)

# Run inference on a specific patient's data
predict_data, output, results = evaluator.wrapped_patient_inference(patient=my_patient)

# Execute jackknife resampling for robust inference
jackknife_results, ci_plots = evaluator.wrapped_jackknife(
    patient=my_patient, results=results_df, sample_fraction=0.8, n_jobs=-1
)
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
