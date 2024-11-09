import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from ..base import Patient, patient_to_df
from ..benchmarking import BaseBenchmark, Baseline, Benchmarker
from ..wrapper import BaseEvaluatorWrapper


def load_benchmark(
    path: Path = Path("reports"),
    file_name: Optional[str] = None,
    folder_name: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Loads the benchmark DataFrame from a CSV file.

    Args:
        path (Path): Path from where the benchmark report is loaded.
            Defaults to "reports".
        file_name (Optional[str]): Name of the CSV file to load.
            Defaults to 'benchmark.csv'.
        folder_name (Optional[str]): Folder name to load the CSV from.
            Defaults to a subfolder within "reports" named after the task.
        verbose (bool): Prints loaded models. Defaults to False.

    Returns:
        pd.DataFrame: Loaded benchmark DataFrame.
    """
    load_path = path / (folder_name if folder_name else "")
    if not load_path.exists():
        raise FileNotFoundError(f"The directory {load_path} does not exist.")

    csv_file_name = file_name if file_name else "benchmark.csv"
    csv_file_path = load_path / csv_file_name

    if not csv_file_path.exists():
        raise FileNotFoundError(f"The file {csv_file_path} does not exist.")

    benchmark_df = pd.read_csv(csv_file_path)
    if verbose:
        print(f"Loaded benchmark report from {csv_file_path}")

    return benchmark_df


def load_learners(
    path: Path = Path("models"),
    folder_name: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Loads the learners from a specified directory.

    Args:
        path (Path): Path from where models are loaded. Defaults to "models".
        folder_name (Optional[str]): Folder name to load models from.
            Defaults to a subfolder within "models" named after the task.
        verbose (bool): Prints loaded models. Defaults to False.

    Returns:
        dict: Dictionary containing loaded learners.
    """
    load_path = path / (folder_name if folder_name else "")
    if not load_path.exists():
        raise FileNotFoundError(f"The directory {load_path} does not exist.")

    learners_dict = {}
    for model_file in load_path.glob("*.pkl"):
        model_name = model_file.stem
        model = joblib.load(model_file)
        learners_dict[model_name] = model
        if verbose:
            print(f"Loaded model {model_name} from {model_file}")

    return learners_dict


class BenchmarkWrapper(BaseBenchmark):
    """Wrapper class for model benchmarking, baseline evaluation, and result storage.

    Inherits:
        - `BaseBenchmark`: Initializes parameters for benchmarking models and provides
          configuration for task, learners, tuning methods, HPO, and criteria.

    Args:
        task (str): Task for evaluation (pocketclosure', 'pocketclosureinf',
            'improvement', or 'pdgrouprevaluation'.).
        learners (List[str]): List of learners to benchmark ('xgb', 'rf', 'lr' or
            'mlp').
        tuning_methods (List[str]): Tuning methods for each learner ('holdout', 'cv').
        hpo_methods (List[str]): HPO methods ('hebo' or 'rs').
        criteria (List[str]): List of evaluation criteria ('f1', 'macro_f1',
            'brier_score').
        encodings (List[str]): List of encodings ('one_hot' or 'target').
        sampling (Optional[List[str]]): Sampling strategies to handle class imbalance.
            Includes None, 'upsampling', 'downsampling', and 'smote'.
        factor (Optional[float]): Factor to apply during resampling.
        n_configs (int): Number of configurations for hyperparameter tuning.
            Defaults to 10.
        n_jobs (int): Number of parallel jobs for processing.
        cv_folds (int): Number of folds for cross-validation. Defaults to 10.
        racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
            Defaults to 5.
        test_seed (int): Random seed for test splitting. Defaults to 0.
        test_size (float): Proportion of data used for testing. Defaults to
            0.2.
        val_size (float): Size of validation set in holdout tuning. Defaults to 0.2.
        cv_seed (int): Random seed for cross-validation. Defaults to 0
        mlp_flag (bool): Enables MLP training with early stopping. Defaults to True.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        verbose (bool): If True, enables detailed logging during benchmarking.
            Defaults to False.
        path (Path): Path to the directory containing processed data files.
        name (str): File name for the processed data file. Defaults to
            "processed_data.csv".

    Attributes:
        classification (str): 'binary' or 'multiclass' based on the task.

    Methods:
        baseline: Evaluates baseline models for each encoding and returns metrics.
        wrapped_benchmark: Runs benchmarks with various learners, encodings, and
            tuning methods.
        save_benchmark: Saves benchmark results to a specified directory.
        save_learners: Saves learners to a specified directory as serialized files.

    Example:
        ```
        # Initialize the BenchmarkWrapper
        benchmarker = BenchmarkWrapper(
            task="pocketclosure",
            encodings=["one_hot", "target"],
            learners=["rf", "xgb", "lr"],
            tuning_methods=["holdout", "cv"],
            hpo_methods=["rs", "hebo"],
            criteria=["f1", "brier_score"],
            sampling=["upsampling"],
            factor=2,
            n_configs=25,
            n_jobs=-1,
            verbose=True,
        )

        # Run baseline benchmarking
        baseline_df = benchmarker.baseline()

        # Run full benchmark and retrieve results
        benchmark_results, learners = benchmarker.wrapped_benchmark()

        # Save the benchmark results
        benchmarker.save_benchmark(baseline_df, path=Path("reports"))

        # Save the trained learners
        benchmarker.save_learners(learners_dict=learners_used, path=Path("models"))
        ```
    """

    def __init__(
        self,
        task: str,
        encodings: List[str],
        learners: List[str],
        tuning_methods: List[str],
        hpo_methods: List[str],
        criteria: List[str],
        sampling: Optional[List[Union[str, None]]] = None,
        factor: Optional[float] = None,
        n_configs: int = 10,
        n_jobs: int = 1,
        verbose: bool = False,
        cv_folds: int = 10,
        racing_folds: Optional[int] = 5,
        test_seed: int = 0,
        test_size: float = 0.2,
        val_size: float = 0.2,
        cv_seed: int = 0,
        mlp_flag: bool = True,
        threshold_tuning: bool = True,
        path: Path = Path("data/processed"),
        name: str = "processed_data.csv",
    ) -> None:
        """Initializes the BenchmarkWrapper."""
        super().__init__(
            task=task,
            learners=learners,
            tuning_methods=tuning_methods,
            hpo_methods=hpo_methods,
            criteria=criteria,
            encodings=encodings,
            sampling=sampling,
            factor=factor,
            n_configs=n_configs,
            n_jobs=n_jobs,
            cv_folds=cv_folds,
            racing_folds=racing_folds,
            test_seed=test_seed,
            test_size=test_size,
            val_size=val_size,
            cv_seed=cv_seed,
            mlp_flag=mlp_flag,
            threshold_tuning=threshold_tuning,
            verbose=verbose,
            path=path,
            name=name,
        )
        self.classification = "multiclass" if task == "pdgrouprevaluation" else "binary"

    def baseline(self) -> pd.DataFrame:
        """Runs baseline benchmark for each encoding type.

        Returns:
            pd.DataFrame: Combined baseline benchmark dataframe with encoding info.
        """
        baseline_dfs = []

        for encoding in self.encodings:
            baseline_df = Baseline(
                task=self.task, encoding=encoding, path=self.path, name=self.name
            ).baseline()
            baseline_df["Encoding"] = encoding
            baseline_dfs.append(baseline_df)

        combined_baseline_df = pd.concat(baseline_dfs, ignore_index=True)
        column_order = ["Model", "Encoding"] + [
            col
            for col in combined_baseline_df.columns
            if col not in ["Model", "Encoding"]
        ]
        combined_baseline_df = combined_baseline_df[column_order]

        return combined_baseline_df

    def wrapped_benchmark(self) -> Tuple[pd.DataFrame, dict]:
        """Runs baseline and benchmarking tasks.

        Returns:
            Tuple[pd.DataFrame, dict]: Benchmark and learners used for evaluation.
        """
        benchmarker = Benchmarker(
            task=self.task,
            learners=self.learners,
            tuning_methods=self.tuning_methods,
            hpo_methods=self.hpo_methods,
            criteria=self.criteria,
            encodings=self.encodings,
            sampling=self.sampling,
            factor=self.factor,
            n_configs=self.n_configs,
            n_jobs=self.n_jobs,
            cv_folds=self.cv_folds,
            test_size=self.test_size,
            val_size=self.val_size,
            test_seed=self.test_seed,
            cv_seed=self.cv_seed,
            mlp_flag=self.mlp_flag,
            threshold_tuning=self.threshold_tuning,
            verbose=self.verbose,
            path=self.path,
            name=self.name,
        )

        return benchmarker.run_all_benchmarks()

    def save_benchmark(
        self,
        benchmark_df: pd.DataFrame,
        path: Path = Path("reports"),
        file_name: Optional[str] = None,
        folder_name: Optional[str] = None,
    ) -> None:
        """Saves the benchmark DataFrame to a specified directory as a CSV file.

        Args:
            benchmark_df (pd.DataFrame): The benchmark DataFrame to save.
            path (Path): Path to save the benchmark report. Defaults to REPORTS_DIR.
            file_name (Optional[str]): Name of CSV file. Defaults to 'benchmark.csv'.
            folder_name (Optional[str]): Folder name for storing the CSV file.
                Defaults to a subfolder within "reports" named after the task.
        """
        save_path = path / (folder_name if folder_name else self.task)
        os.makedirs(save_path, exist_ok=True)
        csv_file_name = (
            file_name
            if file_name and file_name.endswith(".csv")
            else f"{file_name or 'benchmark'}.csv"
        )
        csv_file_path = save_path / csv_file_name
        benchmark_df.to_csv(csv_file_path, index=False)
        print(f"Saved benchmark report to {csv_file_path}")

    def save_learners(
        self,
        learners_dict: dict,
        path: Path = Path("models"),
        folder_name: Optional[str] = None,
    ) -> None:
        """Saves the learners to a specified directory.

        Args:
            learners_dict (dict): Dictionary containing learners to save.
            path: (Path): Path to save models. Defaults to "models".
            folder_name (Optional[str]): Folder name for storing models.
                Defaults to a subfolder within "models" named after the task.
        """
        save_path = path / (folder_name if folder_name else self.task)
        os.makedirs(save_path, exist_ok=True)
        for model_name, model in learners_dict.items():
            model_file_name = f"{model_name}.pkl"
            model_path = save_path / model_file_name
            joblib.dump(model, model_path)
            print(f"Saved model {model_name} to {model_path}")


class EvaluatorWrapper(BaseEvaluatorWrapper):
    """Wrapper class for model evaluation, feature importance, and inference.

    Extends the base evaluation functionality to enable comprehensive model
    evaluation, feature importance analysis, patient inference, and jackknife
    resampling for confidence interval estimation.

    Inherits:
        - `BaseEvaluatorWrapper`: Provides foundational methods and attributes for
          model evaluation, data preparation, and inference.

    Args:
        learners_dict (dict): Dictionary containing trained models and their metadata.
        criterion (str): The criterion used to select the best model ('f1', 'macro_f1',
            'brier_score').
        aggregate (bool): Whether to aggregate one-hot encoding. Defaults
            to True.
        verbose (bool): If True, enables verbose logging during evaluation
            and inference. Defaults to False.

    Attributes:
        learners_dict (dict): Contains metadata about trained models.
        criterion (str): Criterion used for model selection.
        aggregate (bool): Flag for aggregating one-hot encoded metrics.
        verbose (bool): Controls verbose in evaluation processes.
        model (object): Best-ranked model based on the criterion.
        encoding (str): Encoding method ('one_hot' or 'target').
        learner (str): Type of model (learner) used in training.
        task (str): Task associated with the model (e.g., 'pocketclosure', 'improve').
        factor (Optional[float]): Resampling factor if applicable.
        sampling (Optional[str]): Resampling strategy ('upsampling', 'smote', etc.).
        classification (str): Classification type ('binary' or 'multiclass').
        dataloader (ProcessedDataLoader): Data loader and transformer.
        resampler (Resampler): Resampling strategy for training and testing.
        df (pd.DataFrame): Loaded dataset.
        train_df (pd.DataFrame): Training data after splitting.
        _test_df (pd.DataFrame): Test data after splitting.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        base_target (Optional[np.ndarray]): Baseline target for evaluations.
        evaluator (ModelEvaluator): Evaluator for model metrics and feature importance.
        inference_engine (ModelInference): Model inference manager.
        trainer (Trainer): Trainer for model evaluation and optimization.

    Methods:
        wrapped_evaluation: Runs comprehensive evaluation with optional
            plots for metrics such as confusion matrix and Brier scores.
        evaluate_cluster: Performs clustering and calculates Brier scores.
            Allows subsetting of test set.
        evaluate_feature_importance: Computes feature importance using
            specified methods (e.g., SHAP, permutation importance). Allows subsetting
            of test set.
        average_over_splits: Aggregates metrics across multiple data
            splits for robust evaluation.
        wrapped_patient_inference: Conducts inference on individual patient data.
        wrapped_jackknife: Executes jackknife resampling on patient data to
            estimate confidence intervals.

    Inherited Properties:
        - `criterion (str):` Retrieves or sets current evaluation criterion for model
            selection. Supports 'f1', 'brier_score', and 'macro_f1'.
        - `model (object):` Retrieves best-ranked model dynamically based on the current
            criterion. Recalculates when criterion is updated.

    Examples:
        ```
        benchmarker = BenchmarkWrapper(...)

        # Results of BenchmarkWrapper
        benchmark_results, learners = benchmarker.wrapped_benchmark()

        # Initialize the evaluator with required parameters
        evaluator = EvaluatorWrapper(
            learners_dict=learners,
            criterion="f1",
            aggregate=True,
            verbose=True
        )

        # Evaluate the model and generate plots
        evaluator.wrapped_evaluation(cm=True, brier_groups=True)

        # Calculate feature importance
        evaluator.evaluate_feature_importance(fi_types=["shap", "permutation"])

        # Train and average over multiple random splits
        avg_metrics_df = evaluator.average_over_splits(num_splits=5, n_jobs=4)

        # Run inference on a specific patient's data
        patient_inference_df = evaluator.wrapped_patient_inference(patient=my_patient)

        # Execute jackknife resampling for robust inference
        jackknife_results_df = evaluator.wrapped_jackknife(
            patient=my_patient,
            results=results_df,
            sample_fraction=0.8,
            n_jobs=2
        )
        ```
    """

    def __init__(
        self,
        learners_dict: dict,
        criterion: str,
        aggregate: bool = True,
        verbose: bool = False,
        random_state: int = 0,
    ) -> None:
        """Initializes EvaluatorWrapper with model, evaluation, and inference setup."""
        super().__init__(
            learners_dict=learners_dict,
            criterion=criterion,
            aggregate=aggregate,
            verbose=verbose,
            random_state=random_state,
        )

    def wrapped_evaluation(
        self,
        cm: bool = True,
        cm_base: bool = True,
        brier_groups: bool = True,
    ) -> None:
        """Runs evaluation on the best-ranked model.

        Args:
            cm (bool): Plot the confusion matrix. Defaults to True.
            cm_base (bool): Plot confusion matrix vs value before treatment.
                Defaults to True.
            brier_groups (bool): Calculate Brier score groups. Defaults to True.

        Returns:
            None
        """
        if cm:
            self.evaluator.plot_confusion_matrix()
        if cm_base:
            if self.task in [
                "pocketclosure",
                "pocketclosureinf",
                "pdgrouprevaluation",
            ]:
                self.evaluator.plot_confusion_matrix(
                    col=self.base_target, y_label="Pocket Closure"
                )
        if brier_groups:
            self.evaluator.brier_score_groups()

    def evaluate_cluster(
        self,
        base: Optional[str] = None,
        revaluation: Optional[str] = None,
        n_cluster: int = 3,
        true_preds: bool = False,
        brier_threshold: Optional[float] = None,
    ) -> None:
        """Performs cluster analysis with Brier scores, optionally applying subsetting.

        Args:
            base (Optional[str]): Baseline variable for comparison. Defaults to None.
            revaluation (Optional[str]): Revaluation variable. Defaults to None.
            n_cluster (int): Number of clusters for Brier score clustering analysis.
                Defaults to 3.
            true_preds (bool): Whether to further subset by correct predictions.
                Defaults to False.
            brier_threshold (Optional[float]): Threshold for Brier score filtering.
                Defaults to None.
        """
        self.evaluator.X, self.evaluator.y = self._test_filters(
            base=base,
            revaluation=revaluation,
            true_preds=true_preds,
            brier_threshold=brier_threshold,
        )
        self.evaluator.analyze_brier_within_clusters(n_clusters=n_cluster)

    def evaluate_feature_importance(
        self,
        fi_types: List[str],
        base: Optional[str] = None,
        revaluation: Optional[str] = None,
        true_preds: bool = False,
    ) -> None:
        """Evaluates feature importance using the evaluator, with optional subsetting.

        Args:
            fi_types (List[str]): List of feature importance types to evaluate.
            base (Optional[str]): Baseline variable for comparison. Defaults to None.
            revaluation (Optional[str]): Revaluation variable. Defaults to None.
            true_preds (bool): If True, further subsets to cases where model predictions
                match the true labels. Defaults to False.
        """
        self.evaluator.X, self.evaluator.y = self._test_filters(
            base=base,
            revaluation=revaluation,
            true_preds=true_preds,
            brier_threshold=None,
        )
        self.evaluator.evaluate_feature_importance(fi_types=fi_types)

    def average_over_splits(
        self, num_splits: int = 5, n_jobs: int = -1
    ) -> pd.DataFrame:
        """Trains the final model over multiple splits with different seeds.

        Args:
            num_splits (int): Number of random seeds/splits to train the model on.
            n_jobs (int): Number of parallel jobs. Defaults to -1 (use all processors).

        Returns:
            pd.DataFrame: DataFrame containing average performance metrics.
        """
        seeds = range(num_splits)
        metrics_list = Parallel(n_jobs=n_jobs)(
            delayed(self._train_and_get_metrics)(seed, self.learner) for seed in seeds
        )
        avg_metrics = {}
        for metric in metrics_list[0]:
            if metric == "Confusion Matrix":
                continue
            values = [d[metric] for d in metrics_list if d[metric] is not None]
            avg_metrics[metric] = sum(values) / len(values) if values else None

        avg_confusion_matrix = None
        if self.classification == "binary" and "Confusion Matrix" in metrics_list[0]:
            avg_confusion_matrix = (
                np.mean([d["Confusion Matrix"] for d in metrics_list], axis=0)
                .astype(int)
                .tolist()
            )

        results = {
            "Task": self.task,
            "Learner": self.learner,
            "Criterion": self.criterion,
            "Sampling": self.sampling,
            "Factor": self.factor,
            **{
                metric: round(value, 4) if isinstance(value, (int, float)) else value
                for metric, value in avg_metrics.items()
            },
        }

        if avg_confusion_matrix is not None:
            results["Confusion Matrix"] = avg_confusion_matrix

        df_results = pd.DataFrame([results])
        return df_results

    def wrapped_patient_inference(
        self,
        patient: Patient,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs inference on the patient's data using the best-ranked model.

        Args:
            patient (Patient): A `Patient` dataclass instance containing patient-level,
                tooth-level, and side-level information.

        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities for each side
            of the patient's teeth.
        """
        patient_data = patient_to_df(patient=patient)
        predict_data, patient_data = self.inference_engine.prepare_inference(
            task=self.task,
            patient_data=patient_data,
            encoding=self.encoding,
            X_train=self.X_train,
            y_train=self.y_train,
        )

        return self.inference_engine.patient_inference(
            predict_data=predict_data, patient_data=patient_data
        )

    def wrapped_jackknife(
        self,
        patient: Patient,
        results: pd.DataFrame,
        sample_fraction: float = 1.0,
        n_jobs: int = -1,
        max_plots: int = 192,
    ) -> pd.DataFrame:
        """Runs jackknife resampling for inference on a given patient's data.

        Args:
            patient (Patient): `Patient` dataclass instance containing patient-level
                information, tooth-level, and side-level details.
            results (pd.DataFrame): DataFrame to store results from jackknife inference.
            sample_fraction (float, optional): The fraction of patient data to use for
                jackknife resampling. Defaults to 1.0.
            n_jobs (int, optional): The number of parallel jobs to run. Defaults to -1.
            max_plots (int): Maximum number of plots for jackknife intervals.

        Returns:
            pd.DataFrame: The results of jackknife inference.
        """
        patient_data = patient_to_df(patient=patient)
        patient_data, _ = self.inference_engine.prepare_inference(
            task=self.task,
            patient_data=patient_data,
            encoding=self.encoding,
            X_train=self.X_train,
            y_train=self.y_train,
        )
        return self.inference_engine.jackknife_inference(
            model=self.model,
            train_df=self.train_df,
            patient_data=patient_data,
            encoding=self.encoding,
            inference_results=results,
            sample_fraction=sample_fraction,
            n_jobs=n_jobs,
            max_plots=max_plots,
        )
