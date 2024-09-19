import itertools
from typing import List, Optional

import pandas as pd

from pamod.base import BaseEvaluator
from pamod.data import ProcessedDataLoader
from pamod.resampling import Resampler
from pamod.training import Trainer
from pamod.tuning import HEBOTuner, RandomSearchTuner


class Benchmarker(BaseEvaluator):
    def __init__(
        self,
        task: str,
        criterion: str,
        tuning: Optional[str],
        hpo: Optional[str],
        sampling: Optional[str],
        factor: Optional[float],
        n_configs: int,
        racing_folds: int,
        n_jobs: int,
        verbosity: bool = True,
    ) -> None:
        """Initialize the Benchmarker class with tuning parameters.

        Args:
            task (str): The task name used to determine classification type.
            criterion (str): Criterion for optimization ('macro_f1' or 'brier_score').
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (Optional[str]): The hyperparameter optimization method. Can be None.
            sampling (str): Sampling strategy.
            factor (float): Factor for resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
            racing_folds (int): Number of racing folds for Random Search (RS).
            n_jobs (int): Number of parallel jobs to run for evaluation.
            verbosity (bool): Enables verbose output if set to True.
        """
        # Automatically determine classification type based on the task
        classification = self._determine_classification(task)
        super().__init__(classification, criterion, tuning, hpo)
        dataloader = ProcessedDataLoader(task)  # Use task as the dataset name
        df = dataloader.load_data()  # Load the dataset
        self.df = dataloader.transform_target(df)
        self.sampling = sampling
        self.factor = factor
        self.n_configs = n_configs
        self.racing_folds = racing_folds
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.resampler = Resampler(self.classification)
        self.trainer = Trainer(
            self.classification, self.criterion, tuning=None, hpo=None
        )
        self.tuner = self._initialize_tuner()

    def _determine_classification(self, task: str) -> str:
        """Determine classification type based on the task name.

        Args:
            task (str): The task name.

        Returns:
            str: The classification type ('binary' or 'multiclass').
        """
        if task in ["pocketclosure", "improve"]:
            return "binary"
        elif task == "pdgrouprevaluation":
            return "multiclass"
        else:
            raise ValueError(
                f"Unknown task: {task}. Unable to determine classification."
            )

    def _initialize_tuner(self):
        """Initialize the appropriate tuner based on the hpo method."""
        if self.hpo == "RS":
            return RandomSearchTuner(self.classification, self.criterion, self.tuning)
        elif self.hpo == "HEBO":
            return HEBOTuner(self.classification, self.criterion, self.tuning)
        else:
            raise ValueError(f"Unsupported HPO method: {self.hpo}")

    def perform_evaluation(self, learner: str) -> dict:
        """Perform model evaluation and return final metrics.

        Args:
            learner (str): The machine learning model to evaluate.

        Returns:
            dict: A dictionary containing the trained model and its evaluation metrics.
        """
        train_df, _ = self.resampler.split_train_test_df(self.df)

        if self.tuning == "holdout":
            return self._evaluate_holdout(train_df, learner)
        elif self.tuning == "cv":
            return self._evaluate_cv(learner)
        else:
            raise ValueError(f"Unsupported tuning method: {self.tuning}")

    def _evaluate_holdout(self, train_df: pd.DataFrame, learner: str) -> dict:
        """Perform holdout validation and return the final model metrics.

        Args:
            train_df (pd.DataFrame): train df for holdout tuning.
            learner (str): The machine learning model to evaluate.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        train_df_h, test_df_h = self.resampler.split_train_test_df(train_df)
        X_train_h, y_train_h, X_val, y_val = self.resampler.split_x_y(
            train_df_h, test_df_h, self.sampling, self.factor
        )
        best_params, best_threshold = self.tuner.holdout(
            learner,
            X_train_h,
            y_train_h,
            X_val,
            y_val,
            self.n_configs,
            self.n_jobs,
            self.verbosity,
        )
        final_model = (learner, best_params, best_threshold)

        return self.trainer.train_final_model(
            self.df,
            self.resampler,
            final_model,
            self.sampling,
            self.factor,
            self.n_jobs,
        )

    def _evaluate_cv(self, learner: str) -> dict:
        """Perform cross-validation and return the final model metrics.

        Args:
            learner (str): The machine learning model to evaluate.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        outer_splits, _ = self.resampler.cv_folds(self.df, self.sampling, self.factor)
        best_params, best_threshold = self.tuner.cv(
            learner,
            outer_splits,
            self.n_configs,
            self.racing_folds,
            self.n_jobs,
            self.verbosity,
        )
        final_model = (learner, best_params, best_threshold)

        return self.trainer.train_final_model(
            self.df,
            self.resampler,
            final_model,
            self.sampling,
            self.factor,
            self.n_jobs,
        )


class MultiBenchmarker:
    def __init__(
        self,
        tasks: List[str],
        learners: List[str],
        tuning_methods: List[str],
        hpo_methods: List[str],
        criteria: List[str],
        sampling: Optional[str],
        factor: Optional[float],
        n_configs: int,
        racing_folds: int,
        n_jobs: int,
        verbosity: bool = True,
    ) -> None:
        """Initialize the MultiBenchmarker with different tasks, learners, etc.

        Args:
            tasks (List[str]): List of tasks (e.g., 'pocketclosure', 'improve').
            learners (List[str]): List of learners to benchmark (e.g., 'XGB', etc.).
            tuning_methods (List[str]): List of tuning methods ('holdout', 'cv').
            hpo_methods (List[str]): List of HPO methods ('HEBO', 'RS').
            criteria (List[str]): List of evaluation criteria ('f1', 'brier_score').
            sampling (str): Sampling strategy to use.
            factor (float): Factor for resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
            racing_folds (int): Number of racing folds for Random Search.
            n_jobs (int): Number of parallel jobs to run.
            verbosity (bool): Enables verbose output if True.
        """
        self.tasks = tasks
        self.learners = learners
        self.tuning_methods = tuning_methods
        self.hpo_methods = hpo_methods
        self.criteria = criteria
        self.sampling = sampling
        self.factor = factor
        self.n_configs = n_configs
        self.racing_folds = racing_folds
        self.n_jobs = n_jobs
        self.verbosity = verbosity

    def run_all_benchmarks(self) -> pd.DataFrame:
        """Benchmark all combinations of tasks, learners, tuning, HPO, and criteria."""
        results = []

        # Loop through all combinations
        for task, learner, tuning, hpo, criterion in itertools.product(
            self.tasks,
            self.learners,
            self.tuning_methods,
            self.hpo_methods,
            self.criteria,
        ):
            # Skip invalid combinations
            if criterion == "macro_f1" and task != "pdgrouprevaluation":
                continue

            print(
                f"\nRunning benchmark for Task: {task}, Learner: {learner}, "
                f"Tuning: {tuning}, HPO: {hpo}, Criterion: {criterion}"
            )

            # Instantiate and run the Benchmarker for the current combination
            benchmarker = Benchmarker(
                task=task,
                criterion=criterion,
                tuning=tuning,
                hpo=hpo,
                sampling=self.sampling,
                factor=self.factor,
                n_configs=self.n_configs,
                racing_folds=self.racing_folds,
                n_jobs=self.n_jobs,
                verbosity=self.verbosity,
            )

            try:
                # Perform evaluation and collect the results
                metrics = benchmarker.perform_evaluation(learner)
                unpacked_metrics = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics["metrics"].items()
                }

                results.append(
                    {
                        "Task": task,
                        "Learner": learner,
                        "Tuning": tuning,
                        "HPO": hpo,
                        "Criterion": criterion,
                        **unpacked_metrics,  # Unpack rounded metrics here
                    }
                )

            except Exception as e:
                print(f"Error running benchmark for {task}, {learner}: {e}\n")

        df_results = pd.DataFrame(results)
        self._print_results_table(df_results)
        return df_results

    def _print_results_table(self, df_results: pd.DataFrame) -> None:
        """Print the benchmark results in a tabular format.

        Args:
            df_results (pd.DataFrame): DataFrame of benchmark results.
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print("\nBenchmark Results Summary:")
        print(df_results)


def main():
    tasks = ["pocketclosure"]
    # learners = ["XGB", "RandomForest", "LogisticRegression", "MLP"]
    learners = ["XGB"]
    tuning_methods = ["holdout"]
    hpo_methods = ["HEBO", "RS"]
    criteria = ["f1"]

    multi_benchmarker = MultiBenchmarker(
        tasks=tasks,
        learners=learners,
        tuning_methods=tuning_methods,
        hpo_methods=hpo_methods,
        criteria=criteria,
        sampling=None,
        factor=None,
        n_configs=3,
        racing_folds=3,
        n_jobs=-1,
        verbosity=True,
    )

    multi_benchmarker.run_all_benchmarks()


if __name__ == "__main__":
    main()
