from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from pamod.benchmarking import BaseBenchmark, Baseline, Benchmarker
from pamod.config import PROCESSED_BASE_DIR
from pamod.data import ProcessedDataLoader
from pamod.evaluation import Evaluator
from pamod.resampling import Resampler


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
            verbosity (bool): Enables verbose output if set to True.
            path (str): Directory path for the processed data.
            name (str): File name for the processed data.
        """
        super().__init__(
            task,
            learners,
            tuning_methods,
            hpo_methods,
            criteria,
            encodings,
            sampling,
            factor,
            n_configs,
            n_jobs,
            cv_folds,
            racing_folds,
            test_seed,
            test_size,
            val_size,
            cv_seed,
            mlp_flag,
            verbosity,
            path,
            name,
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
            verbosity=self.verbosity,
            path=self.path,
            name=self.name,
        )

        return benchmarker.run_all_benchmarks()

    def wrapped_evaluator(self, learners_dict: dict) -> None:
        """Runs evaluation on the best-ranked model from learners_dict.

        Args:
            learners_dict (dict): Dictionary containing models and their metadata.
        """
        best_model_key = next((key for key in learners_dict if "rank1" in key), None)

        if not best_model_key:
            raise ValueError("No model with rank1 found in learners_dict")

        best_model = learners_dict[best_model_key]
        encoding = "one_hot" if best_model_key.split("_")[-6] == "one" else "target"

        if encoding not in ["one_hot", "target"]:
            raise ValueError(f"Invalid encoding extracted: {encoding}")

        dataloader = ProcessedDataLoader(task=self.task, encoding=encoding)
        resampler = Resampler(classification=self.classification, encoding=encoding)
        df = dataloader.load_data()

        if self.task in ["pocketclosure", "pdgrouprevaluation"]:
            target_before = self._generate_target_before(df)

        df = dataloader.transform_data(df)
        train_df, test_df = resampler.split_train_test_df(df)

        if self.task in ["pocketclosure", "pdgrouprevaluation"]:
            test_patient_ids = test_df[self.group_col]
            target_before = target_before[
                df[self.group_col].isin(test_patient_ids)
            ].values

        _, _, X_test, y_test = resampler.split_x_y(train_df, test_df)

        evaluator = Evaluator(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            encoding=encoding,
        )

        evaluator.plot_confusion_matrix()

        if self.task in ["pocketclosure", "pdgrouprevaluation"]:
            evaluator.plot_confusion_matrix(col=target_before, y_label="Pocket Closure")

        evaluator.brier_score_groups()
        evaluator.analyze_brier_within_clusters()
        evaluator.evaluate_feature_importance(importance_types=["standard"])

    def _generate_target_before(self, df: pd.DataFrame) -> pd.Series:
        """Generates the target column before treatment based on the task.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.Series: The target before column for evaluation.
        """
        if self.task == "pocketclosure":
            return df.apply(
                lambda row: (
                    0
                    if row["pdbaseline"] == 4
                    and row["bop"] == 2
                    or row["pdbaseline"] > 4
                    else 1
                ),
                axis=1,
            )
        elif self.task == "pdgrouprevaluation":
            return df["pdbaseline"]
        else:
            raise ValueError(f"Task '{self.task}' is not recognized.")
