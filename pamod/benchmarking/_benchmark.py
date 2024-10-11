import itertools
from pathlib import Path
import traceback
from typing import List, Optional, Tuple

import pandas as pd

from pamod.base import BaseEvaluator
from pamod.config import PROCESSED_BASE_DIR
from pamod.data import ProcessedDataLoader
from pamod.resampling import Resampler
from pamod.training import Trainer
from pamod.tuning import HEBOTuner, RandomSearchTuner


class Experiment(BaseEvaluator):
    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        learner: str,
        criterion: str,
        encoding: str,
        tuning: Optional[str],
        hpo: Optional[str],
        sampling: Optional[str],
        factor: Optional[float],
        n_configs: int,
        racing_folds: Optional[int],
        n_jobs: int,
        cv_folds: Optional[int] = None,
        test_seed: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        cv_seed: Optional[int] = None,
        mlp_flag: Optional[bool] = None,
        verbosity: bool = True,
    ) -> None:
        """Initialize the Experiment class with tuning parameters.

        Args:
            df (pd.DataFrame): The preloaded data.
            task (str): The task name used to determine classification type.
            learner (str): The machine learning model to evaluate.
            criterion (str): Criterion for optimization ('macro_f1' or 'brier_score').
            encoding (str): Encoding type ('one_hot' or 'binary')
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (Optional[str]): The hyperparameter optimization method. Can be None.
            sampling (str): Sampling strategy.
            factor (float): Factor for resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
            n_jobs (int): Number of parallel jobs to run for evaluation.
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
        self.df = df
        self.task = task
        classification = self._determine_classification()
        super().__init__(classification, criterion, tuning, hpo)
        self.learner = learner
        self.encoding = encoding
        self.sampling = sampling
        self.factor = factor
        self.n_configs = n_configs
        self.racing_folds = racing_folds
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds if cv_folds is not None else self.n_folds
        self.test_seed = test_seed if test_seed is not None else self.random_state_split
        self.test_size = test_size if test_size is not None else self.test_set_size
        self.val_size = val_size if val_size is not None else self.val_set_size
        self.cv_seed = cv_seed if cv_seed is not None else self.random_state_cv
        self.verbosity = verbosity
        self.mlp_flag = mlp_flag if mlp_flag is not None else self.mlp_training
        self.resampler = Resampler(self.classification, self.encoding)
        self.trainer = Trainer(
            self.classification,
            self.criterion,
            tuning=None,
            hpo=None,
            mlp_training=self.mlp_flag,
        )
        self.tuner = self._initialize_tuner()

    def _determine_classification(self) -> str:
        """Determine classification type based on the task name.

        Returns:
            str: The classification type ('binary' or 'multiclass').
        """
        if self.task in ["pocketclosure", "improve"]:
            return "binary"
        elif self.task == "pdgrouprevaluation":
            return "multiclass"
        else:
            raise ValueError(
                f"Unknown task: {self.task}. Unable to determine classification."
            )

    def _initialize_tuner(self):
        """Initialize the appropriate tuner based on the hpo method."""
        if self.hpo == "rs":
            return RandomSearchTuner(self.classification, self.criterion, self.tuning)
        elif self.hpo == "hebo":
            return HEBOTuner(self.classification, self.criterion, self.tuning)
        else:
            raise ValueError(f"Unsupported HPO method: {self.hpo}")

    def perform_evaluation(self) -> dict:
        """Perform model evaluation and return final metrics.

        Returns:
            dict: A dictionary containing the trained model and its evaluation metrics.
        """
        train_df, _ = self.resampler.split_train_test_df(
            self.df, self.test_seed, self.test_size
        )

        if self.tuning == "holdout":
            return self._evaluate_holdout(train_df)
        elif self.tuning == "cv":
            return self._evaluate_cv()
        else:
            raise ValueError(f"Unsupported tuning method: {self.tuning}")

    def _evaluate_holdout(self, train_df: pd.DataFrame) -> dict:
        """Perform holdout validation and return the final model metrics.

        Args:
            train_df (pd.DataFrame): train df for holdout tuning.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        train_df_h, test_df_h = self.resampler.split_train_test_df(
            train_df, self.test_seed, self.val_size
        )
        X_train_h, y_train_h, X_val, y_val = self.resampler.split_x_y(
            train_df_h, test_df_h, self.sampling, self.factor
        )
        best_params, best_threshold = self.tuner.holdout(
            self.learner,
            X_train_h,
            y_train_h,
            X_val,
            y_val,
            self.n_configs,
            self.n_jobs,
            self.verbosity,
        )
        final_model = (self.learner, best_params, best_threshold)

        return self.trainer.train_final_model(
            self.df,
            self.resampler,
            final_model,
            self.sampling,
            self.factor,
            self.n_jobs,
            self.test_seed,
            self.test_size,
        )

    def _evaluate_cv(self) -> dict:
        """Perform cross-validation and return the final model metrics.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        outer_splits, _ = self.resampler.cv_folds(
            self.df, self.sampling, self.factor, self.cv_seed, self.cv_folds
        )
        best_params, best_threshold = self.tuner.cv(
            self.learner,
            outer_splits,
            self.n_configs,
            self.racing_folds,
            self.n_jobs,
            self.verbosity,
        )
        final_model = (self.learner, best_params, best_threshold)

        return self.trainer.train_final_model(
            self.df,
            self.resampler,
            final_model,
            self.sampling,
            self.factor,
            self.n_jobs,
            self.test_seed,
            self.test_size,
        )


class Benchmarker:
    def __init__(
        self,
        tasks: List[str],
        learners: List[str],
        tuning_methods: List[str],
        hpo_methods: List[str],
        criteria: List[str],
        encodings: List[str],
        sampling: Optional[str],
        factor: Optional[float],
        n_configs: int,
        racing_folds: int,
        n_jobs: int,
        cv_folds: Optional[int] = None,
        test_seed: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        cv_seed: Optional[int] = None,
        mlp_flag: Optional[bool] = None,
        verbosity: bool = True,
        path: Path = PROCESSED_BASE_DIR,
        name: str = "processed_data.csv",
    ) -> None:
        """Initialize the Experiment with different tasks, learners, etc.

        Args:
            tasks (List[str]): List of tasks (e.g., 'pocketclosure', 'improve').
            learners (List[str]): List of learners to benchmark (e.g., 'xgb', etc.).
            tuning_methods (List[str]): List of tuning methods ('holdout', 'cv').
            hpo_methods (List[str]): List of HPO methods ('hebo', 'rs').
            criteria (List[str]): List of evaluation criteria ('f1', 'brier_score').
            encodings (List[str]): List of encodings ('one_hot' or 'target').
            sampling (str): Sampling strategy to use.
            factor (float): Factor for resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
            racing_folds (int): Number of racing folds for Random Search.
            n_jobs (int): Number of parallel jobs to run.
            cv_folds (Optional[int], optional): Number of folds for cross-validation.
                Defaults to None, in which case the class's `n_folds` will be used.
            test_seed (Optional[int], optional): Random seed for splitting.
                Defaults to None.
            test_size (Optional[float]): Size of grouped train test split.
            val_size (Optional[float]): Size of grouped train test split for holdout.
            cv_seed (int): Seed for splitting CV folds.
            mlp_flag (Optional[bool]): Flag for MLP training with early stopping.
            verbosity (bool): Enables verbose output if True.
            path (str): Directory path for the processed data.
            name (str): File name for the processed data.
        """
        self.tasks = tasks
        self.learners = learners
        self.tuning_methods = tuning_methods
        self.hpo_methods = hpo_methods
        self.criteria = criteria
        self.encodings = encodings
        self.sampling = sampling
        self.factor = factor
        self.n_configs = n_configs
        self.racing_folds = racing_folds
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.cv_folds = cv_folds
        self.test_seed = test_seed
        self.test_size = test_size
        self.val_size = val_size
        self.cv_seed = cv_seed
        self.mlp_flag = mlp_flag
        self.path = path
        self.name = name
        self.data_cache = self._load_data_for_tasks()

    def _load_data_for_tasks(self) -> dict:
        """Load and transform data for each task and encoding combination once.

        Returns:
            dict: A dictionary containing transformed data for each task-encoding pair.
        """
        data_cache = {}
        for task, encoding in itertools.product(self.tasks, self.encodings):
            cache_key = (task, encoding)  # Use task and encoding as key for caching

            if cache_key not in data_cache:
                dataloader = ProcessedDataLoader(task, encoding)
                df = dataloader.load_data(self.path, self.name)
                transformed_df = dataloader.transform_data(df)
                data_cache[cache_key] = transformed_df

        return data_cache

    def run_all_benchmarks(self) -> Tuple[pd.DataFrame, dict]:
        """Benchmark all combinations of tasks, learners, tuning, HPO, and criteria."""
        results = []
        learners_dict = {}

        for task, learner, tuning, hpo, criterion, encoding in itertools.product(
            self.tasks,
            self.learners,
            self.tuning_methods,
            self.hpo_methods,
            self.criteria,
            self.encodings,
        ):
            if (criterion == "macro_f1" and task != "pdgrouprevaluation") or (
                criterion == "f1" and task == "pdgrouprevaluation"
            ):
                print(f"Criterion '{criterion}' and task '{task}' not valid.")
                continue

            print(
                f"\nRunning benchmark for Task: {task}, Learner: {learner}, "
                f"Tuning: {tuning}, HPO: {hpo}, Criterion: {criterion}"
            )

            df = self.data_cache[(task, encoding)]

            exp = Experiment(
                df=df,
                task=task,
                learner=learner,
                criterion=criterion,
                encoding=encoding,
                tuning=tuning,
                hpo=hpo,
                sampling=self.sampling,
                factor=self.factor,
                n_configs=self.n_configs,
                racing_folds=self.racing_folds,
                n_jobs=self.n_jobs,
                cv_folds=self.cv_folds,
                test_seed=self.test_seed,
                test_size=self.test_size,
                val_size=self.val_size,
                cv_seed=self.cv_seed,
                mlp_flag=self.mlp_flag,
                verbosity=self.verbosity,
            )

            try:
                result = exp.perform_evaluation()
                metrics = result["metrics"]
                trained_model = result["model"]

                unpacked_metrics = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()
                }

                results.append(
                    {
                        "Task": task,
                        "Learner": learner,
                        "Tuning": tuning,
                        "HPO": hpo,
                        "Criterion": criterion,
                        **unpacked_metrics,
                    }
                )

                learners_dict[
                    f"{task}_{learner}_{tuning}_{hpo}_{criterion}_{encoding}"
                ] = trained_model

            except Exception as e:
                print(f"Error running benchmark for {task}, {learner}: {e}\n")
                traceback.print_exc()

        df_results = pd.DataFrame(results)
        self._print_results_table(df_results)
        return df_results, learners_dict

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
    tasks = ["pdgrouprevaluation"]
    learners = ["xgb", "rf", "lr", "mlp"]
    # learners = ["xgb"]
    tuning_methods = ["holdout"]
    hpo_methods = ["hebo"]
    criteria = ["macro_f1"]
    encoding = ["one_hot"]

    benchmarker = Benchmarker(
        tasks=tasks,
        learners=learners,
        tuning_methods=tuning_methods,
        hpo_methods=hpo_methods,
        criteria=criteria,
        encodings=encoding,
        sampling=None,
        factor=None,
        n_configs=3,
        racing_folds=3,
        n_jobs=-1,
        verbosity=True,
    )

    benchmarker.run_all_benchmarks()


if __name__ == "__main__":
    main()
