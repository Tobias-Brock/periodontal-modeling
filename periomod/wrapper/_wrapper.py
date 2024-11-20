import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from ..base import Patient, patient_to_df
from ..benchmarking import BaseBenchmark, Baseline, Benchmarker
from ..wrapper import BaseEvaluatorWrapper


def load_benchmark(path: Union[str, Path]) -> pd.DataFrame:
    """Loads the benchmark DataFrame from a specified CSV file.

    Args:
        path (Union[str, Path]): Path to the benchmark CSV file.

    Returns:
        pd.DataFrame: Loaded benchmark DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    return pd.read_csv(path)


def load_learners(path: Union[str, Path], verbose: bool = False) -> dict:
    """Loads the learners from a specified directory.

    Args:
        path (Union[str, Path]): Path to the directory where models are stored.
        verbose (bool): Prints loaded models. Defaults to False.

    Returns:
        dict: Dictionary containing loaded learners.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"The directory {path} does not exist.")

    learners_dict = {}
    for model_file in path.glob("*.pkl"):
        model = joblib.load(model_file)
        learners_dict[model_file.stem] = model
        if verbose:
            print(f"Loaded model {model_file.stem} from {model_file}")

    return learners_dict


class BenchmarkWrapper(BaseBenchmark):
    """Wrapper class for model benchmarking, baseline evaluation, and result storage.

    Inherits:
        - `BaseBenchmark`: Initializes parameters for benchmarking models and provides
          configuration for task, learners, tuning methods, HPO, and criteria.

    Args:
        task (str): Task for evaluation ('pocketclosure', 'pocketclosureinf',
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
            Defaults to None.
        test_seed (int): Random seed for test splitting. Defaults to 0.
        test_size (float): Proportion of data used for testing. Defaults to
            0.2.
        val_size (Optional[float]): Size of validation set in holdout tuning.
            Defaults to 0.2.
        cv_seed (int): Random seed for cross-validation. Defaults to 0
        mlp_flag (Optional[bool]): Enables MLP training with early stopping.
            Defaults to True.
        threshold_tuning (Optional[bool]): Enables threshold tuning for binary
            classification. Defaults to None.
        verbose (bool): If True, enables detailed logging during benchmarking.
            Defaults to False.
        path (Path): Path to the directory containing processed data files.
            Defaults to Path("data/processed").

    Attributes:
        task (str): The specified task for evaluation.
        learners (List[str]): List of learners to evaluate.
        tuning_methods (List[str]): Tuning methods for model evaluation.
        hpo_methods (List[str]): HPO methods for hyperparameter tuning.
        criteria (List[str]): List of evaluation metrics.
        encodings (List[str]): Encoding types for categorical features.
        sampling (List[str]): Resampling strategies for class balancing.
        factor (float): Resampling factor for balancing.
        n_configs (int): Number of configurations for hyperparameter tuning.
        n_jobs (int): Number of parallel jobs for model training.
        cv_folds (int): Number of cross-validation folds.
        racing_folds (int): Number of racing folds for random search.
        test_seed (int): Seed for reproducible train-test splits.
        test_size (float): Size of the test split.
        val_size (float): Size of the validation split in holdout tuning.
        cv_seed (int): Seed for cross-validation splits.
        mlp_flag (bool): Indicates if MLP training with early stopping is used.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        verbose (bool): Enables detailed logging during benchmarking.
        path (Path): Directory path for processed data.
        classification (str): 'binary' or 'multiclass' based on the task.

    Methods:
        baseline: Evaluates baseline models for each encoding and returns metrics.
        wrapped_benchmark: Runs benchmarks with various learners, encodings, and
            tuning methods.
        save_benchmark: Saves benchmark results to a specified directory.
        save_learners: Saves learners to a specified directory as serialized files.

    Example:
        ```
        from periomod.wrapper import BenchmarkWrapper

        # Initialize the BenchmarkWrapper
        benchmarker = BenchmarkWrapper(
            task="pocketclosure",
            encodings=["one_hot", "target"],
            learners=["rf", "xgb", "lr", "mlp"],
            tuning_methods=["holdout", "cv"],
            hpo_methods=["rs", "hebo"],
            criteria=["f1", "brier_score"],
            sampling=["upsampling"],
            factor=2,
            path="/data/processed/processed_data.csv",
        )

        # Run baseline benchmarking
        baseline_df = benchmarker.baseline()

        # Run full benchmark and retrieve results
        benchmark, learners = benchmarker.wrapped_benchmark()

        # Save the benchmark results
        benchmarker.save_benchmark(
            benchmark_df=benchmark,
            path="reports/experiment/benchmark.csv",
        )

        # Save the trained learners
        benchmarker.save_learners(learners_dict=learners, path="models/experiment")
        ```
    """

    def __init__(
        self,
        task: str,
        learners: List[str],
        tuning_methods: List[str],
        hpo_methods: List[str],
        criteria: List[str],
        encodings: List[str],
        sampling: Optional[List[Union[str, None]]] = None,
        factor: Optional[float] = None,
        n_configs: int = 10,
        n_jobs: int = 1,
        cv_folds: Optional[int] = 10,
        racing_folds: Optional[int] = None,
        test_seed: int = 0,
        test_size: float = 0.2,
        val_size: Optional[float] = 0.2,
        cv_seed: Optional[int] = 0,
        mlp_flag: Optional[bool] = None,
        threshold_tuning: Optional[bool] = None,
        verbose: bool = False,
        path: Path = Path("data/processed/processed_data.csv"),
    ) -> None:
        """Initializes the BenchmarkWrapper.

        Args:
            task (str): Task for evaluation ('pocketclosure', 'pocketclosureinf',
                'improvement', or 'pdgrouprevaluation'.).
            learners (List[str]): List of learners to benchmark ('xgb', 'rf', 'lr' or
                'mlp').
            tuning_methods (List[str]): Tuning methods for each learner
                ('holdout', 'cv').
            hpo_methods (List[str]): HPO methods ('hebo' or 'rs').
            criteria (List[str]): List of evaluation criteria ('f1', 'macro_f1',
                'brier_score').
            encodings (List[str]): List of encodings ('one_hot' or 'target').
            sampling (Optional[List[str]]): Sampling strategies to handle class
                imbalance. Includes None, 'upsampling', 'downsampling', and 'smote'.
            factor (Optional[float]): Factor to apply during resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
                Defaults to 10.
            n_jobs (int): Number of parallel jobs for processing.
            cv_folds (int): Number of folds for cross-validation. Defaults to 10.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
                Defaults to None.
            test_seed (int): Random seed for test splitting. Defaults to 0.
            test_size (float): Proportion of data used for testing. Defaults to
                0.2.
            val_size (Optional[float]): Size of validation set in holdout tuning.
                Defaults to 0.2.
            cv_seed (int): Random seed for cross-validation. Defaults to 0
            mlp_flag (Optional[bool]): Enables MLP training with early stopping.
                Defaults to True.
            threshold_tuning (Optional[bool]): Enables threshold tuning for binary
                classification. Defaults to None.
            verbose (bool): If True, enables detailed logging during benchmarking.
                Defaults to False.
            path (Path): Path to the directory containing processed data files.
                Defaults to Path("data/processed").
        """
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
                task=self.task,
                encoding=encoding,
                path=self.path,
                random_state=self.test_seed,
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
            Tuple: Benchmark and learners used for evaluation.
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
        )

        return benchmarker.run_benchmarks()

    @staticmethod
    def save_benchmark(benchmark_df: pd.DataFrame, path: Union[str, Path]) -> None:
        """Saves the benchmark DataFrame to the specified path as a CSV file.

        Args:
            benchmark_df (pd.DataFrame): The benchmark DataFrame to save.
            path (Union[str, Path]): Path (including filename) where CSV file will be
                saved.

        Raises:
            ValueError: If the benchmark DataFrame is empty.
            FileNotFoundError: If the parent directory of the path does not exist.
        """
        path = Path(path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if benchmark_df.empty:
            raise ValueError("Benchmark DataFrame is empty and cannot be saved.")

        if not path.suffix == ".csv":
            path = path.with_suffix(".csv")

        os.makedirs(path.parent, exist_ok=True)
        benchmark_df.to_csv(path, index=False)
        print(f"Saved benchmark report to {path}")

    @staticmethod
    def save_learners(learners_dict: dict, path: Union[str, Path]) -> None:
        """Saves the learners to the specified directory.

        Args:
            learners_dict (dict): Dictionary containing learners to save.
            path (Union[str, Path]): Path to the directory where models will be saved.

        Raises:
            ValueError: If the learners dictionary is empty.
        """
        path = Path(path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not learners_dict:
            raise ValueError("Learners dictionary is empty and cannot be saved.")

        os.makedirs(path, exist_ok=True)

        for model_name, model in learners_dict.items():
            model_file_name = f"{model_name}.pkl"
            model_path = path / model_file_name
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
        learners_dict (Dict): Dictionary containing trained models and their metadata.
        criterion (str): The criterion used to select the best model ('f1', 'macro_f1',
            'brier_score').
        aggregate (bool): Whether to aggregate one-hot encoding. Defaults
            to True.
        verbose (bool): If True, enables verbose logging during evaluation
            and inference. Defaults to False.
        random_state (int): Random state for resampling. Defaults to 0.
        path (Path): Path to the directory containing processed data files.
            Defaults to Path("data/processed/processed_data.csv").

    Attributes:
        learners_dict (Dict): Contains metadata about trained models.
        criterion (str): Criterion used for model selection.
        aggregate (bool): Flag for aggregating one-hot encoded metrics.
        verbose (bool): Controls verbose in evaluation processes.
        model (object): Best-ranked model based on the criterion.
        encoding (str): Encoding method ('one_hot' or 'target').
        learner (str): Type of model (learner) used in training.
        task (str): Task associated with the extracted model.
        factor (Optional[float]): Resampling factor if applicable.
        sampling (Optional[str]): Resampling strategy ('upsampling', 'smote', etc.).
        classification (str): Classification type ('binary' or 'multiclass').
        dataloader (ProcessedDataLoader): Data loader and transformer.
        resampler (Resampler): Resampling strategy for training and testing.
        df (pd.DataFrame): Loaded dataset.
        df_processed (pd.DataFrame): Processed dataset.
        train_df (pd.DataFrame): Training data after splitting.
        test_df (pd.DataFrame): Test data after splitting.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        base_target (Optional[np.ndarray]): Baseline target for evaluations.
        baseline (Baseline): Basline class for model analysis.
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
        from periomod.base import Patient, patient_to_dataframe
        from periomod.wrapper import EvaluatorWrapper, load_benchmark, load_learners

        benchmark = load_benchmark(path="reports/experiment/benchmark.csv")
        learners = load_learners(path="models/experiments")

        # Initialize evaluator with learners from BenchmarkWrapper and f1 criterion
        evaluator = EvaluatorWrapper(
            learners_dict=learners,
            criterion="f1",
            path="data/processed/processed_data.csv"
        )

        # Evaluate the model and generate plots
        evaluator.wrapped_evaluation()

        # Cluster analysis on predictions with brier score smaller than threshold
        evaluator.evaluate_cluster(brier_threshold=0.15)

        # Calculate feature importance
        evaluator.evaluate_feature_importance(fi_types=["shap", "permutation"])

        # Train and average over multiple random splits
        avg_metrics_df = evaluator.average_over_splits(num_splits=5, n_jobs=-1)

        # Define a patient instance
        patient = Patient()
        patient_df = patient_to_df(patient=patient)

        # Run inference on a specific patient's data
        predict_data, output, results = evaluator.wrapped_patient_inference(
            patient=patient
            )

        # Execute jackknife resampling for robust inference
        jackknife_results, ci_plots = evaluator.wrapped_jackknife(
            patient=my_patient, results=results_df, sample_fraction=0.8, n_jobs=-1
        )
        ```
    """

    def __init__(
        self,
        learners_dict: Dict,
        criterion: str,
        aggregate: bool = True,
        verbose: bool = False,
        random_state: int = 0,
        path: Path = Path("data/processed/processed_data.csv"),
    ) -> None:
        """Initializes EvaluatorWrapper with model, evaluation, and inference setup.

        Args:
            learners_dict (Dict): Dictionary containing trained models.
            criterion (str): The criterion used to select the best model ('f1',
                'macro_f1', 'brier_score').
            aggregate (bool): Whether to aggregate one-hot encoding. Defaults
                to True.
            verbose (bool): If True, enables verbose logging during evaluation
                and inference. Defaults to False.
            random_state (int): Random state for resampling. Defaults to 0.
            path (Path): Path to the directory containing processed data files.
                Defaults to Path("data/processed/processed_data.csv").

        """
        super().__init__(
            learners_dict=learners_dict,
            criterion=criterion,
            aggregate=aggregate,
            verbose=verbose,
            random_state=random_state,
            path=path,
        )

    def wrapped_evaluation(
        self,
        cm: bool = True,
        cm_base: bool = True,
        brier_groups: bool = True,
        calibration: bool = True,
        tight_layout: bool = False,
    ) -> None:
        """Runs evaluation on the best-ranked model.

        Args:
            cm (bool): Plot the confusion matrix. Defaults to True.
            cm_base (bool): Plot confusion matrix vs value before treatment.
                Defaults to True.
            brier_groups (bool): Calculate Brier score groups. Defaults to True.
            calibration (bool): Plots model calibration. Defaults to True.
            tight_layout (bool): If True, applies tight layout to the plot.
                Defaults to False.
        """
        if cm:
            self.evaluator.plot_confusion_matrix(
                tight_layout=tight_layout, task=self.task
            )
        if cm_base:
            if self.task in [
                "pocketclosure",
                "pocketclosureinf",
                "pdgrouprevaluation",
            ]:
                self.evaluator.plot_confusion_matrix(
                    col=self.base_target,
                    y_label="Pocket Closure",
                    tight_layout=tight_layout,
                    task=self.task,
                )
        if brier_groups:
            self.evaluator.brier_score_groups(tight_layout=tight_layout, task=self.task)
        if calibration:
            self.evaluator.calibration_plot(task=self.task, tight_layout=tight_layout)

    def compare_bss(
        self,
        base: Optional[str] = None,
        revaluation: Optional[str] = None,
        true_preds: bool = False,
        brier_threshold: Optional[float] = None,
        tight_layout: bool = False,
    ) -> None:
        """Compares Brier Skill Score of model with baseline on test set.

        Args:
            base (Optional[str]): Baseline variable for comparison. Defaults to None.
            revaluation (Optional[str]): Revaluation variable. Defaults to None.
            true_preds (bool): Subset by correct predictions. Defaults to False.
            brier_threshold (Optional[float]): Filters observations ny Brier score
                threshold. Defaults to None.
            tight_layout (bool): If True, applies tight layout to the plot.
                Defaults to False.
        """
        baseline_models, _, _ = self.baseline.train_baselines()
        self.evaluator.X, self.evaluator.y, patients = self._test_filters(
            X=self.evaluator.X,
            y=self.evaluator.y,
            base=base,
            revaluation=revaluation,
            true_preds=true_preds,
            brier_threshold=brier_threshold,
        )
        self.evaluator.bss_comparison(
            baseline_models=baseline_models,
            classification=self.classification,
            num_patients=patients,
            tight_layout=tight_layout,
        )
        self.evaluator.X, self.evaluator.y = self.X_test, self.y_test

    def evaluate_cluster(
        self,
        n_cluster: int = 3,
        base: Optional[str] = None,
        revaluation: Optional[str] = None,
        true_preds: bool = False,
        brier_threshold: Optional[float] = None,
        tight_layout: bool = False,
    ) -> None:
        """Performs cluster analysis with Brier scores, optionally applying subsetting.

        This method allows detailed feature analysis by offering multiple subsetting
        options for the test set. The base and revaluation columns allow filtering of
        observations that have not changed after treatment. With true_preds, only
        observations that were correctly predicted are considered. The brier_threshold
        enables filtering of observations that achieved a smaller Brier score at
        prediction time than the threshold.

        Args:
            n_cluster (int): Number of clusters for Brier score clustering analysis.
                Defaults to 3.
            base (Optional[str]): Baseline variable for comparison. Defaults to None.
            revaluation (Optional[str]): Revaluation variable. Defaults to None.
            true_preds (bool): Subset by correct predictions. Defaults to False.
            brier_threshold (Optional[float]): Filters observations ny Brier score
                threshold. Defaults to None.
            tight_layout (bool): If True, applies tight layout to the plot.
                Defaults to False.
        """
        self.evaluator.X, self.evaluator.y, patients = self._test_filters(
            X=self.evaluator.X,
            y=self.evaluator.y,
            base=base,
            revaluation=revaluation,
            true_preds=true_preds,
            brier_threshold=brier_threshold,
        )
        print(f"Number of patients in test set: {patients}")
        print(f"Number of tooth sites: {len(self.evaluator.y)}")
        self.evaluator.analyze_brier_within_clusters(
            n_clusters=n_cluster, tight_layout=tight_layout
        )
        self.evaluator.X, self.evaluator.y = self.X_test, self.y_test

    def evaluate_feature_importance(
        self,
        fi_types: List[str],
        base: Optional[str] = None,
        revaluation: Optional[str] = None,
        true_preds: bool = False,
        brier_threshold: Optional[float] = None,
    ) -> None:
        """Evaluates feature importance using the evaluator, with optional subsetting.

        This method allows detailed feature analysis by offering multiple subsetting
        options for the test set. The base and revaluation columns allow filtering of
        observations that have not changed after treatment. With true_preds, only
        observations that were correctly predicted are considered. The brier_threshold
        enables filtering of observations that achieved a smaller Brier score at
        prediction time than the threshold.

        Args:
            fi_types (List[str]): List of feature importance types to evaluate.
            base (Optional[str]): Baseline variable for comparison. Defaults to None.
            revaluation (Optional[str]): Revaluation variable. Defaults to None.
            true_preds (bool): Subset by correct predictions. Defaults to False.
            brier_threshold (Optional[float]): Filters observations ny Brier score
                threshold. Defaults to None.
        """
        self.evaluator.X, self.evaluator.y, patients = self._test_filters(
            X=self.evaluator.X,
            y=self.evaluator.y,
            base=base,
            revaluation=revaluation,
            true_preds=true_preds,
            brier_threshold=brier_threshold,
        )
        print(f"Number of patients in test set: {patients}")
        print(f"Number of tooth sites: {len(self.evaluator.y)}")
        self.evaluator.evaluate_feature_importance(fi_types=fi_types)
        self.evaluator.X, self.evaluator.y = self.X_test, self.y_test

    def average_over_splits(
        self, num_splits: int = 5, n_jobs: int = -1
    ) -> pd.DataFrame:
        """Trains the final model over multiple splits with different seeds.

        Args:
            num_splits (int): Number of random seeds/splits to train the model on.
                Defaults to 5.
            n_jobs (int): Number of parallel jobs. Defaults to -1 (use all processors).

        Returns:
            DataFrame: DataFrame containing average performance metrics.
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

        return pd.DataFrame([results])

    def wrapped_patient_inference(
        self,
        patient: Patient,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs inference on the patient's data using the best-ranked model.

        Args:
            patient (Patient): A `Patient` dataclass instance containing patient-level,
                tooth-level, and side-level information.

        Returns:
            DataFrame: DataFrame with predictions and probabilities for each side
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
            DataFrame: The results of jackknife inference.
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
