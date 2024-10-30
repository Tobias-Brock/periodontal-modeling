# pa-modeling

A Python package for comprehensive periodontal data processing and modeling. This package provides tools for preprocessing, automatic hyperparameter tuning, resampling, model evaluation, inference, and descriptive analysis with an interactive Gradio frontend. It is designed for Python 3.11.

## Features

- **Preprocessing Pipeline**: Flexible preprocessing of periodontal data, including encoding, scaling, and transformation.
- **Automatic Model Tuning**: Supports multiple learners and tuning strategies for optimized model training.
- **Resampling and Handling Class Imbalance**: Resampling strategies such as SMOTE and upsampling/downsampling to balance dataset classes.
- **Model Evaluation**: Cross-validation and holdout evaluation with support for criteria such as F1, Brier score, and Macro-F1.
- **Inference and Descriptive Statistics**: Patient-level inference, jackknife resampling, and 2D histogram generation.
- **Interactive Frontend with Gradio**: A simple Gradio interface for streamlined model benchmarking and inference.
- **Descriptive Analysis**: Generate descriptive statistics and plots such as confusion matrices and bar plots for model insights.

## Installation

Ensure you have Python 3.11 installed. Install the package via pip:

```bash
pip install pa-modeling
```

## Usage

### 1. Prprocessing

Use the StaticProcessEngine class to preprocess your data. This class handles data transformation, encoding, and other preprocessing tasks to prepare data for modeling.

```python
from pamod.data import StaticProcessEngine

engine = StaticProcessEngine(df=my_data)
processed_data = engine.transform_data()
```

### 2. Training

```python
from pamod.training import Trainer

trainer = Trainer(classification="binary", criterion="f1", tuning="cv", hpo="hebo")
trained_model = trainer.train_final_model(df=processed_data)
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
