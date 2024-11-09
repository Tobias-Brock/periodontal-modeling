# periodontal-modeling Documentation

## Overview

Welcome to the **periodontal-modeling** documentation, a fully automated benchmarking and evaluation package for short-term periodontal modeling. The package provides flexible and efficient preprocessing, model tuning, evaluation, inference, and descriptive analysis with an interactive Gradio frontend, allowing users to perform comprehensive periodontal data modeling.

This documentation includes detailed information on the functionality, setup, and usage of each module and provides code examples for streamlined integration into your projects.

## Key Features

- **Preprocessing Pipeline**: Flexible preprocessing for periodontal data, including encoding, scaling, and transformation.
- **Descriptive Analysis**: Generate detailed statistics and visualizations, including confusion matrices and bar plots.
- **Automated Model Tuning**: Support for various learners and tuning strategies for optimized model training.
- **Resampling**: Balanced class handling via SMOTE, upsampling, and downsampling techniques.
- **Model Evaluation**: Cross-validation, holdout evaluation, and support for criteria such as F1, Brier score, and Macro-F1.
- **Inference**: Patient-level inference, jackknife resampling, and 2D histogram generation.
- **Interactive Gradio Interface**: A user-friendly interface for seamless model benchmarking, evaluation, and inference.

## Installation

Ensure you have Python 3.11 installed. Install the package via pip:

```bash
pip install periodontal-modeling
```

## Commands and Usage

The `Makefile` provides key entry points for common tasks, making project setup and management easier. You can run tasks such as building Docker images, running benchmarks, or serving the Gradio app directly from the command line.

### Running the Gradio Interface
Launch the Gradio app for interactive model evaluation and benchmarking either directly or through Docker:

```python
from periomod.app import perioapp

perioapp.launch()
```

If you download the repository and install the package in editable mode, the following `make` command starts the app:

```bash
pip install -e .
make app
```

The app can also be launched using docker. Run the following commands in the root of the repository:

```bash
docker build -f docker/app.dockerfile -t periomod-image .
docker run -p 7890:7890 periomod-image
```
By default, the app will be launched on port 7890 and can be accessed at `http://localhost:7890`.

Alternatively, the following `make` commands are available to build and run the docker image:

```bash
make docker-build
make docker-run
```80:7880 periomod-image
```

## Core Modules
The following sections summarize each core module within `periodontal-modeling`, with links to detailed documentation and examples for each function.

### App Module

The **App** module hosts the Gradio-based interface, providing a unified platform for tasks such as plotting descriptives, conducting benchmarks, evaluating models, and performing inference.

[Read more on the App Module](reference/app/index.md)

### Data Module

The **Data** module provides tools for loading, transforming, and saving processed periodontal data. Classes like `StaticProcessEngine` handle data preparation, while `ProcessedDataLoader` is tailored for fully preprocessed datasets, supporting encoding and scaling options.

[Read more on the Data Module](reference/data/index.md)

### Descriptive Analysis

The **Descriptives** module enables users to perform statistical analysis and visualize periodontal data pre- and post-therapy using tools like `DescriptivesPlotter`. It includes methods for generating matrices, histograms, and pocket depth comparisons.

[Read more on the Descriptives Module](reference/descriptives/index.md)


### Training Module

The **Training** module provides core training and model-building functionalities, including threshold tuning. The `Trainer` classes allow you to streamline model fitting.

[Read more on the Training Module](reference/training/index.md)

### Tuning Module

The **Tuning** module offers classes like `HEBOTuner` and `RandomSearchTuner` for hyperparameter optimization, supporting both Bayesian optimization and random search.

[Read more on the Tuning Module](reference/tuning/index.md)

### Evaluation Module

The **Evaluation** module supplies methods for post-model training analysis, including prediction assessment, feature importance calculations, and feature clustering. The `ModelEvaluator` class enables confusion matrix plotting and other feature-based evaluations.

[Read more on the Evaluation Module](reference/evaluation/index.md)

### Inference Module

The **Inference** module handles patient-specific predictions, confidence intervals through jackknife resampling, and both single and batch inference. The `ModelInference` class supports robust inference operations.

[Read more on the Inference Module](reference/inference/index.md)

### Benchmarking Module

The **Benchmarking** module provides tools for running and comparing multiple experiments across different tuning setups. Use `Experiment` for single benchmark execution or `Benchmarker` to perform full comparative analyses across models and configurations.

[Read more on the Benchmarking Module](reference/benchmarking/index.md)

### Wrapper Module

The **Wrapper** module simplifies the benchmarking and evaluation setup, consolidating various submodules for a more concise codebase. `BenchmarkWrapper` enables straightforward benchmarking, while `EvaluatorWrapper` supports model evaluation, feature importance, and patient-level inference.

[Read more on the Wrapper Module](reference/wrapper/index.md)
