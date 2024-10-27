import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from ..base import Patient, patient_to_dataframe
from ..benchmarking import BaseBenchmark, Baseline, Benchmarker
from ..config import MODELS_DIR, PROCESSED_BASE_DIR, REPORTS_DIR
from ..wrapper import BaseEvaluatorWrapper


def load_benchmark(
    path: Path = REPORTS_DIR,
    file_name: Optional[str] = None,
    folder_name: Optional[str] = None,
    verbosity: bool = False,
) -> pd.DataFrame:
    """Loads the benchmark DataFrame from a CSV file.

    Args:
        path (Path): Path from where the benchmark report is loaded.
            Defaults to REPORTS_DIR.
        file_name (Optional[str]): Name of the CSV file to load.
            Defaults to 'benchmark.csv'.
        folder_name (Optional[str]): Folder name to load the CSV from.
            Defaults to a subfolder within REPORTS_DIR named after the task.
        verbosity (bool): Prints loaded models. Defaults to False.

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
    if verbosity:
        print(f"Loaded benchmark report from {csv_file_path}")

    return benchmark_df


def load_learners(
    path: Path = MODELS_DIR, folder_name: Optional[str] = None, verbosity: bool = False
) -> dict:
    """Loads the learners from a specified directory.

    Args:
        path (Path): Path from where models are loaded. Defaults to MODELS_DIR.
        folder_name (Optional[str]): Folder name to load models from.
            Defaults to a subfolder within MODELS_DIR named after the task.
        verbosity (bool): Prints loaded models. Defaults to False.

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
        if verbosity:
            print(f"Loaded model {model_name} from {model_file}")

    return learners_dict


class BenchmarkWrapper(BaseBenchmark):
    """Benchmarking and Evaluation Wrapper."""

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
        n_jobs: Optional[int] = None,
        verbosity: bool = False,
        cv_folds: Optional[int] = None,
        racing_folds: Optional[int] = None,
        test_seed: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        cv_seed: Optional[int] = None,
        mlp_flag: Optional[bool] = None,
        threshold_tuning: bool = True,
        path: Path = PROCESSED_BASE_DIR,
        name: str = "processed_data.csv",
    ) -> None:
        """Initializes the BenchmarkWrapper.

        Args:
            task (str): Task for evaluation. Can be "pocketclosure",
                "pdgrouprevaluation", or "improvement".
            encodings (List[str]): Type of encoding. Can be "one_hot" or "target".
            learners (List[str]): List of learners for benchmarking.
            tuning_methods (List[str]): Tuning methods for learners.
            hpo_methods (List[str]): Hyperparameter optimization methods.
            criteria (List[str]): Evaluation criteria for benchmarking.
            sampling (Optional[List[str]]): Sampling strategy to use. Defaults to None.
            factor (Optional[float]): Factor for resampling. Defaults to None.
            n_configs (int, optional): Number of configurations to test in HPO.
                Defaults to 10.
            n_jobs (Optional[int], optional): Number of parallel jobs. Defaults to None.
            cv_folds (Optional[int], optional): Number of folds for cross-validation.
                Defaults to None, in which case the class's `n_folds` will be used.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
            test_seed (Optional[int], optional): Random seed for splitting.
                Defaults to None.
            test_size (Optional[float]): Size of grouped train test split.
            val_size (Optional[float]): Size of grouped train test split for holdout.
            cv_seed (int): Seed for splitting CV folds.
            mlp_flag (bool): Flag for MLP training with early stopping.
            threshold_tuning (bool): Perform threshold tuning for binary classification
                if the criterion is "f1". Defaults to True.
            verbosity (bool): Enables verbose output if set to True.
            path (str): Directory path for the processed data.
            name (str): File name for the processed data.
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
            verbosity=verbosity,
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
            baseline_df = Baseline(task=self.task, encoding=encoding).baseline()
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
            verbosity=self.verbosity,
            path=self.path,
            name=self.name,
        )

        return benchmarker.run_all_benchmarks()

    def save_benchmark(
        self,
        benchmark_df: pd.DataFrame,
        path: Path = REPORTS_DIR,
        file_name: Optional[str] = None,
        folder_name: Optional[str] = None,
    ) -> None:
        """Saves the benchmark DataFrame to a specified directory as a CSV file.

        Args:
            benchmark_df (pd.DataFrame): The benchmark DataFrame to save.
            path (Path): Path to save the benchmark report. Defaults to REPORTS_DIR.
            file_name (Optional[str]): Name of CSV file. Defaults to 'benchmark.csv'.
            folder_name (Optional[str]): Folder name for storing the CSV file.
                Defaults to a subfolder within REPORTS_DIR named after the task.
        """
        save_path = path / (folder_name if folder_name else self.task)
        os.makedirs(save_path, exist_ok=True)
        csv_file_name = file_name if file_name else "benchmark.csv"
        csv_file_path = save_path / csv_file_name
        benchmark_df.to_csv(csv_file_path, index=False)
        print(f"Saved benchmark report to {csv_file_path}")

    def save_learners(
        self,
        learners_dict: dict,
        path: Path = MODELS_DIR,
        folder_name: Optional[str] = None,
    ) -> None:
        """Saves the learners to a specified directory.

        Args:
            learners_dict (dict): Dictionary containing learners to save.
            path: (Path): Path to save models. Defaults to MODELS_DIR.
            folder_name (Optional[str]): Folder name for storing models.
                Defaults to a subfolder within MODELS_DIR named after the task.
        """
        save_path = path / (folder_name if folder_name else self.task)
        os.makedirs(save_path, exist_ok=True)
        for model_name, model in learners_dict.items():
            model_file_name = f"{model_name}.pkl"
            model_path = save_path / model_file_name

            joblib.dump(model, model_path)
            print(f"Saved model {model_name} to {model_path}")


class EvaluatorWrapper(BaseEvaluatorWrapper):
    """Wrapper class for model evaluation and inference, including jackknife resampling.

    Args:
        learners_dict (dict): Dictionary containing trained models and their metadata.
        criterion (str): The criterion used to select the best model (e.g., 'f1').
        aggregate (bool, optional): Whether to aggregate one-hot encoding.
            Defaults to True.
        verbosity (bool, optional): If True, enables verbose logging during evaluation
            and inference. Defaults to False.
    """

    def __init__(
        self,
        learners_dict: dict,
        criterion,
        aggregate: bool = True,
        verbosity: bool = False,
    ) -> None:
        """Initializes EvaluatorWrapper with model, evaluation, and inference setup."""
        super().__init__(
            learners_dict=learners_dict,
            criterion=criterion,
            aggregate=aggregate,
            verbosity=verbosity,
        )

    def wrapped_evaluation(
        self,
        cm: bool = True,
        cm_base: bool = True,
        brier_groups: bool = True,
        cluster: bool = True,
        n_cluster: int = 3,
    ) -> None:
        """Runs evaluation on best-ranked model based on criterion."""
        if cm:
            self.evaluator.plot_confusion_matrix()
        if cm_base:
            if self.task in ["pocketclosure", "pdgrouprevaluation"]:
                self.evaluator.plot_confusion_matrix(
                    col=self.base_target, y_label="Pocket Closure"
                )
        if brier_groups:
            self.evaluator.brier_score_groups()
        if cluster:
            self.evaluator.analyze_brier_within_clusters(n_clusters=n_cluster)

    def evaluate_feature_importance(self, fi_types: List[str]):
        """Evaluates feature importance using the provided evaluator.

        Args:
            fi_types (List[str]): List of feature importance types.
        """
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
        avg_metrics = {
            metric: sum(d[metric] for d in metrics_list) / num_splits
            for metric in metrics_list[0]
            if metric != "Confusion Matrix"
        }

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
    ):
        """Runs inference on the patient's data using the best-ranked model.

        Args:
            patient (Patient): A `Patient` dataclass instance containing patient-level,
                tooth-level, and side-level information.

        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities for each side
            of the patient's teeth.
        """
        patient_data = patient_to_dataframe(patient=patient)
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
        patient_data = patient_to_dataframe(patient=patient)
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
