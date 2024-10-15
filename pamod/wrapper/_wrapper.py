from typing import List, Optional, Tuple

import pandas as pd

from pamod.base import BaseHydra
from pamod.benchmarking import Baseline, Benchmarker
from pamod.data import ProcessedDataLoader
from pamod.evaluation import Evaluator
from pamod.resampling import Resampler


class BenchmarkWrapper(BaseHydra):
    """Benchmarking and Evaluation Wrapper.

    This class streamlines benchmarking and model evaluation for specific tasks,
    such as 'pocketclosure' or 'pdgrouprevaluation'.

    Attributes:
        task (str): Task for evaluation. Can be "pocketclosure",
            "pdgrouprevaluation", or "improvement".
        encoding (str): Type of encoding. Can be "one_hot" or "target".
        learners (List[str]): List of learners for benchmarking.
        tuning (List[str]): Tuning methods for learners.
        hpo (List[str]): Hyperparameter optimization methods.
        criteria (List[str]): Evaluation criteria for benchmarking.
        n_configs (int): Number of configurations to test in HPO.
        n_jobs (Optional[int]): Number of jobs for parallel processing.
        verbosity (bool): Whether to print progress during benchmarking.
        classification (str): Classification type ("multiclass" or "binary").
    """

    def __init__(
        self,
        task: str,
        encoding: str,
        learners: List[str],
        tuning_methods: List[str],
        hpo_methods: List[str],
        criteria: List[str],
        sampling: Optional[str] = None,
        factor: Optional[float] = None,
        n_configs: int = 10,
        n_jobs: Optional[int] = None,
        verbosity: bool = False,
        cv_folds: Optional[int] = None,
        test_seed: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        cv_seed: Optional[int] = None,
        mlp_flag: Optional[bool] = None,
    ) -> None:
        """Initializes the BenchmarkWrapper.

        Args:
            task (str): Task for evaluation. Can be "pocketclosure",
                "pdgrouprevaluation", or "improvement".
            encoding (str): Type of encoding. Can be "one_hot" or "target".
            learners (List[str]): List of learners for benchmarking.
            tuning_methods (List[str]): Tuning methods for learners.
            hpo_methods (List[str]): Hyperparameter optimization methods.
            criteria (List[str]): Evaluation criteria for benchmarking.
            sampling (Optional[str]): Sampling strategy to use. Defaults to None.
            factor (Optional[float]): Factor for resampling. Defaults to None.
            n_configs (int, optional): Number of configurations to test in HPO.
                Defaults to 10.
            n_jobs (Optional[int], optional): Number of parallel jobs. Defaults to None.
            cv_folds (Optional[int], optional): Number of folds for cross-validation.
                Defaults to None, in which case the class's `n_folds` will be used.
            test_seed (Optional[int], optional): Random seed for splitting.
                Defaults to None.
            test_size (Optional[float]): Size of grouped train test split.
            val_size (Optional[float]): Size of grouped train test split for holdout.
            cv_seed (int): Seed for splitting CV folds.
            mlp_flag (bool): Flag for MLP training with early stopping.
            verbosity (bool): Enables verbose output if set to True.
        """
        super().__init__()
        self.task = task
        self.encoding = encoding
        self.learners = learners
        self.tuning = tuning_methods
        self.hpo = hpo_methods
        self.criteria = criteria
        self.sampling = sampling
        self.factor = factor
        self.n_configs = n_configs
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.cv_folds = cv_folds
        self.test_seed = test_seed
        self.test_size = test_size
        self.val_size = val_size
        self.cv_seed = cv_seed
        self.mlp_flag = mlp_flag
        self.classification = "multiclass" if task == "pdgrouprevaluation" else "binary"

    def wrapped_benchmark(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Runs baseline and benchmarking tasks.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, dict]: The baseline, benchmark,
            and learners used for evaluation.
        """
        baseline = Baseline(task=self.task, encoding=self.encoding).baseline()

        benchmarker = Benchmarker(
            tasks=[self.task],
            learners=self.learners,
            tuning_methods=self.tuning,
            hpo_methods=self.hpo,
            criteria=self.criteria,
            encodings=[self.encoding],
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
        )

        benchmark, learners = benchmarker.run_all_benchmarks()

        return baseline, benchmark, learners

    def wrapped_evaluator(self, model: object) -> None:
        """Runs evaluation on a specific model.

        Args:
            model (object): The model to be evaluated.
        """
        dataloader = ProcessedDataLoader(target=self.task, encoding=self.encoding)
        resampler = Resampler(
            classification=self.classification, encoding=self.encoding
        )
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
            model=model,
            X_test=X_test,
            y_test=y_test,
            encoding=self.encoding,
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
